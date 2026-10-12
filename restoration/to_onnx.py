"""Export a trained checkpoint to ONNX and verify it against PyTorch.

    python to_onnx.py --checkpoint runs/dual_full/best.pth --out model.onnx

Contract: input [1, 3, 256, 256] float 0..1 (NCHW), output [1, 3, 256, 256].
Qubee bakes the /255 + HWC handling in at compile time (see sr_preproc.yaml),
so on the NPU the model consumes raw [256, 256, 3] uint8 images.
"""

import argparse
import collections
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

from dataset import PairedDataset
from model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="model.onnx")
    p.add_argument("--data", default="data")
    p.add_argument("--opset", type=int, default=13)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(ckpt["arch"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Legacy TorchScript exporter: the dynamo exporter only emits opset >= 18,
    # which the Qubee 0.12 compiler cannot consume.
    torch.onnx.export(model, torch.rand(1, 3, 256, 256), args.out,
                      input_names=["input"], output_names=["output"],
                      opset_version=args.opset, dynamic_axes=None, dynamo=False)
    onnx.checker.check_model(args.out)
    ops = collections.Counter(n.op_type for n in onnx.load(args.out).graph.node)
    print(f"Exported {args.out}  ops: {dict(ops)}")

    # Compare ONNX Runtime vs PyTorch on the validation set (PSNR vs HR).
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    val = PairedDataset(os.path.join(args.data, "valid"))
    max_diff, psnr_pt, psnr_onnx = 0.0, [], []
    for i in range(len(val)):
        lr, hr = val[i]
        x = lr.unsqueeze(0)
        with torch.no_grad():
            y_pt = model(x).numpy()
        y_onnx = sess.run(None, {"input": x.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(y_pt - y_onnx).max()))
        hr255 = np.round(hr.numpy() * 255)
        for y, lst in ((y_pt, psnr_pt), (y_onnx, psnr_onnx)):
            sr255 = np.round(np.clip(y[0], 0, 1) * 255)
            lst.append(10 * np.log10(255 ** 2 / np.mean((sr255 - hr255) ** 2)))
    print(f"max |pytorch - onnx| = {max_diff:.2e}")
    print(f"val PSNR  pytorch {np.mean(psnr_pt):.3f} dB   onnx {np.mean(psnr_onnx):.3f} dB "
          f"(checkpoint says {ckpt.get('val_psnr', float('nan')):.3f})")


if __name__ == "__main__":
    main()
