"""Train a restoration model on data/train, evaluate PSNR on data/valid.

Examples:
    python train.py --arch dual --steps 3000 --out runs/dual_quick          # quick comparison run
    python train.py --arch dual --steps 60000 --out runs/dual_full          # full training
    python train.py --arch dual --steps 10000 --loss mse --lr 5e-5 \
        --crop 256 --init runs/dual_full/best.pth --out runs/dual_mse      # MSE fine-tune
"""

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PairedDataset
from model import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    """Mean per-image PSNR on 0..255 after clip + round, like the NPU output path."""
    model.eval()
    psnrs = []
    for lr, hr in loader:
        sr = model(lr.to(device)).clamp(0, 1)
        sr = torch.round(sr * 255.0)
        hr = torch.round(hr.to(device) * 255.0)
        mse = ((sr - hr) ** 2).mean(dim=(1, 2, 3))
        psnrs.extend((10 * torch.log10(255.0 ** 2 / mse)).tolist())
    return float(np.mean(psnrs))


@torch.no_grad()
def ema_update(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(p.detach(), alpha=1 - decay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="dual", choices=["plain", "dual", "unet"])
    p.add_argument("--model-kwargs", default="{}", help='JSON, e.g. \'{"n_deep": 16}\'')
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--loss", default="l1", choices=["l1", "mse", "charbonnier"])
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--init", default=None, help="checkpoint to start from (fine-tuning)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    model_kwargs = json.loads(args.model_kwargs)

    model = build_model(args.arch, **model_kwargs).to(device)
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Initialized from {args.init} (val PSNR {ckpt.get('val_psnr', float('nan')):.3f})")
    ema = copy.deepcopy(model).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"arch={args.arch} kwargs={model_kwargs} params={n_params:.2f}M device={device}")

    train_ds = PairedDataset(os.path.join(args.data, "train"), crop=args.crop, augment=True)
    val_ds = PairedDataset(os.path.join(args.data, "valid"))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                          num_workers=args.workers, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=4)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0)
    warmup = min(500, args.steps // 20)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        t = (step - warmup) / max(1, args.steps - warmup)
        return 1e-7 + 0.5 * (args.lr - 1e-7) * (1 + math.cos(math.pi * t))

    def loss_fn(sr, hr):
        if args.loss == "l1":
            return F.l1_loss(sr, hr)
        if args.loss == "mse":
            return F.mse_loss(sr, hr)
        return torch.sqrt((sr - hr) ** 2 + 1e-6).mean()

    history, best, step, t0 = [], -1.0, 0, time.time()
    start_psnr = evaluate(ema, val_dl, device)
    # Models start as identity, so a fresh run prints the "do nothing" baseline here.
    print(f"step 0  val PSNR {start_psnr:.3f} dB (model at init)")

    while step < args.steps:
        for lr_img, hr_img in train_dl:
            if step >= args.steps:
                break
            model.train()
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            loss = loss_fn(model(lr_img), hr_img)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # EMA warms up so early evals aren't dominated by the init weights.
            ema_update(ema, model, min(args.ema, (1 + step) / (10 + step)))
            step += 1

            if step % args.eval_every == 0 or step == args.steps:
                psnr = evaluate(ema, val_dl, device)
                elapsed = (time.time() - t0) / 60
                history.append({"step": step, "loss": loss.item(), "val_psnr": psnr, "minutes": elapsed})
                tag = ""
                if psnr > best:
                    best = psnr
                    torch.save({"arch": args.arch, "model_kwargs": model_kwargs,
                                "model_state_dict": ema.state_dict(), "val_psnr": psnr, "step": step},
                               os.path.join(args.out, "best.pth"))
                    tag = "  *best*"
                print(f"step {step:6d}  loss {loss.item():.4f}  lr {lr_at(step):.2e}  "
                      f"val PSNR {psnr:.3f} dB  ({elapsed:.1f} min){tag}", flush=True)

    torch.save({"arch": args.arch, "model_kwargs": model_kwargs,
                "model_state_dict": ema.state_dict(), "val_psnr": history[-1]["val_psnr"], "step": step},
               os.path.join(args.out, "last.pth"))
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"args": vars(args), "params_M": n_params, "best_val_psnr": best,
                   "history": history}, f, indent=1)
    print(f"Done. best val PSNR {best:.3f} dB -> {args.out}/best.pth")


if __name__ == "__main__":
    main()
