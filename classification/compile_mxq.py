import argparse
import os

# Runs INSIDE the Qubee Docker container (mobilint/qbcompiler:0.12-...).
# After compiling, the produced .mxq runs on the NPU via MACCEL (Windows).
from qubee import mxq_compile


def main():
    parser = argparse.ArgumentParser(
        description="Compile ONNX model to MXQ with Qubee (run in Docker)"
    )
    parser.add_argument("--onnx", default="model.onnx")
    parser.add_argument("--calib", default="calib_data",
                        help="calibration dataset folder from prepare_calib.py")
    parser.add_argument("--out", default="model.mxq")
    args = parser.parse_args()

    calib_path = os.path.join(os.path.abspath(args.calib), "resnet20_cali")

    print("=== Qubee compilation ===")
    print(f"  onnx        : {args.onnx}")
    print(f"  calib data  : {calib_path}")
    print(f"  output mxq  : {args.out}")
    print()

    # Quantization options follow the Qubee classification tutorial:
    #  - per-channel weights, max-percentile activation calibration
    mxq_compile(
        model=args.onnx,
        calib_data_path=calib_path,
        quantization_method=1,   # per channel quantization
        quantization_mode=2,     # max percentile
        percentile=0.999,
        quantization_output=0,   # per-layer output quantization
        save_path=args.out,
        backend="onnx",
    )

    print(f"\nDone. Compiled model: {args.out}")
    print("Output .mxq is in the mounted folder; ready for MACCEL on Windows.")


if __name__ == "__main__":
    main()