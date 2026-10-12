"""DOCKER-SIDE: compile ONNX -> MXQ with Qubee (submission file: ONNX-to-MXQ script).

    python compile_mxq.py --onnx model.onnx --calib calib_data --out model.mxq

Quantization settings matter a lot for PSNR. If the NPU PSNR drops noticeably
below the ONNX PSNR, try another --percentile (0.9999 / 0.999 / 0.99) or
--mode 0 (plain max) and compare on the validation set with npu_infer.py.
"""

import argparse
import os

from qubee import mxq_compile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="model.onnx")
    p.add_argument("--calib", default="calib_data")
    p.add_argument("--out", default="model.mxq")
    p.add_argument("--mode", type=int, default=2, help="quantization_mode: 2 = max percentile")
    p.add_argument("--percentile", type=float, default=0.9999)
    args = p.parse_args()

    calib_path = os.path.join(os.path.abspath(args.calib), "sr_cali")
    print(f"onnx={args.onnx} calib={calib_path} out={args.out} "
          f"mode={args.mode} percentile={args.percentile}")

    mxq_compile(
        model=args.onnx,
        calib_data_path=calib_path,
        quantization_method=1,       # per-channel weights
        quantization_mode=args.mode,
        percentile=args.percentile,
        quantization_output=0,       # per-layer output quantization
        save_path=args.out,
        backend="onnx",
    )
    print(f"Done: {args.out}")


if __name__ == "__main__":
    main()
