import argparse
import os

import torch

from model import ResNet20


def main():
    parser = argparse.ArgumentParser(
        description="Export trained ResNet20 to ONNX for Qubee compilation"
    )
    parser.add_argument("--checkpoint", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoint.pth"))
    parser.add_argument("--out", default="model.onnx")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = ResNet20(num_classes=20)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # INPUT SHAPE CONTRACT: [1, 3, 240, 240] (NCHW), output 20 logits.
    # We use the LEGACY exporter (dynamo=False) with a conservative opset so
    # the lowering to opset 13 is done by the mature TorchScript path; the new
    # dynamo-based exporter only supports opset >=18, which older Qubee
    # compilers cannot consume. No onnxscript/version-converter needed here.
    dummy = torch.randn(1, 3, 240, 240)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        opset_version=13,
        dynamo=False,
    )
    print(f"Exported ONNX to {args.out}")

    # Sanity check the checkpoint's reported accuracy.
    if "best_val_acc" in ckpt:
        print(f"(checkpoint best val acc was {ckpt['best_val_acc']*100:.2f}%)")


if __name__ == "__main__":
    main()
