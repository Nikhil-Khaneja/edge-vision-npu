"""DOCKER-SIDE: build Qubee calibration data from calib_images/ (training LR images).

Must run INSIDE the Qubee Docker container:
    python prepare_calib.py --images-dir calib_images --calib-dir calib_data
"""

import argparse
import os
import sys

try:
    from qubee.calibration import make_calib
except ImportError:
    sys.exit("[error] qubee is not installed. Run this inside the Qubee Docker container:\n"
             "  pip install /NPU_Projects/qubee-0.12.0.0+aries2-py3-none-any.whl")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default=os.path.join(root, "calib_images"))
    p.add_argument("--calib-dir", default=os.path.join(root, "calib_data"))
    p.add_argument("--preproc-yaml", default=os.path.join(root, "sr_preproc.yaml"))
    args = p.parse_args()

    n = len([f for f in os.listdir(args.images_dir) if not f.startswith(".")])
    make_calib(
        args_pre=os.path.abspath(args.preproc_yaml),
        data_dir=os.path.abspath(args.images_dir),
        save_dir=os.path.abspath(args.calib_dir),
        save_name="sr_cali",
        max_size=n,
    )
    print(f"Calibration data ({n} training LR images) ready in {args.calib_dir}/sr_cali")


if __name__ == "__main__":
    main()
