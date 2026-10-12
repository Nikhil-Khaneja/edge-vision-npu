"""WINDOWS/NPU-SIDE: run model.mxq on the validation set and report mean PSNR.

    python npu_infer.py --mxq model.mxq --data data/valid

Feeds raw LR images as [256, 256, 3] uint8 (scaling is baked into the MXQ).
"""

import argparse
import os

import maccel
import numpy as np
from PIL import Image


def to_hwc_255(out):
    """NPU output -> HWC float on the 0..255 scale, whatever layout it comes in."""
    y = np.asarray(out, dtype=np.float32).squeeze()
    if y.ndim == 1:
        y = y.reshape(256, 256, 3)
    elif y.shape[0] == 3 and y.shape[-1] != 3:
        y = np.transpose(y, (1, 2, 0))  # CHW -> HWC
    return y * 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mxq", default="model.mxq")
    p.add_argument("--data", default="data/valid", help="folder containing LR2/ and HR2/")
    p.add_argument("--save-dir", default=None, help="optionally save restored images")
    args = p.parse_args()

    acc = maccel.Accelerator()
    model = maccel.Model(args.mxq)
    model.launch(acc)

    lr_dir, hr_dir = os.path.join(args.data, "LR2"), os.path.join(args.data, "HR2")
    names = sorted(f for f in os.listdir(hr_dir) if not f.startswith("."))
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    psnrs_npu, psnrs_in = [], []
    for i, n in enumerate(names):
        lr = np.array(Image.open(os.path.join(lr_dir, n)).convert("RGB"), dtype=np.uint8)
        hr = np.array(Image.open(os.path.join(hr_dir, n)).convert("RGB"), dtype=np.float64)

        out = model.infer(lr)
        if out is None:
            raise RuntimeError("NPU inference returned no output")
        sr = np.round(np.clip(to_hwc_255(out[0]), 0, 255))
        if i == 0:
            print(f"raw output shape {np.asarray(out[0]).shape}, "
                  f"range {np.min(out[0]):.3f}..{np.max(out[0]):.3f}")

        psnrs_npu.append(10 * np.log10(255 ** 2 / np.mean((sr - hr) ** 2)))
        psnrs_in.append(10 * np.log10(255 ** 2 / np.mean((lr - hr) ** 2)))
        print(f"[{i + 1}/{len(names)}] {n}  NPU PSNR {psnrs_npu[-1]:.2f} dB  (input {psnrs_in[-1]:.2f})")
        if args.save_dir:
            Image.fromarray(sr.astype(np.uint8)).save(os.path.join(args.save_dir, n.replace(".jpg", ".png")))

    print(f"\nMean PSNR of LR input : {np.mean(psnrs_in):.3f} dB")
    print(f"Mean PSNR on NPU      : {np.mean(psnrs_npu):.3f} dB   <-- leaderboard metric")


if __name__ == "__main__":
    main()
