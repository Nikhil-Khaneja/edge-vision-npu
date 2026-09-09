"""DOCKER-SIDE script: build Qubee calibration data from the training split.

IMPORTANT: This imports `qubee` and must run INSIDE the Qubee Docker
container on the NPU machine (see NPU_RUNBOOK.md). It expects a
flat folder of raw calibration images (produced by collect_calib_images.py,
which you can run anywhere).

Usage (inside Docker, /NPU_Projects):
    python prepare_calib.py --train-txt imagenet_train20.txt \
                            --train-dir imagenet_train20a \
                            --images-dir calib_images --num-calib 128
"""

import argparse
import os
import shutil
import sys

try:
    from qubee.calibration import make_calib
except ImportError:
    sys.exit(
        "[error] qubee is not installed. This script must run INSIDE the "
        "Qubee Docker container:\n"
        "  pip install /NPU_Projects/qubee-0.12.0.0+aries2-py3-none-any.whl\n"
        "It cannot run in Colab. Use collect_calib_images.py there instead."
    )

from dataset import ImgDataset


def main():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Prepare Qubee calibration data from the TRAINING split"
    )
    parser.add_argument("--train-txt", default=os.path.join(ROOT, "dataset/imagenet_train20.txt"))
    parser.add_argument("--train-dir", default=os.path.join(ROOT, "dataset/imagenet_train20a"))
    parser.add_argument("--images-dir", default=os.path.join(ROOT, "calib_images"),
                        help="flat folder of raw calibration images")
    parser.add_argument("--calib-dir", default=os.path.join(ROOT, "calib_data"),
                        help="Qubee calibration dataset output folder")
    parser.add_argument("--num-calib", type=int, default=128,
                        help="total calibration images to use")
    parser.add_argument("--preproc-yaml", default=os.path.join(ROOT, "resnet20_preproc.yaml"))
    args = parser.parse_args()

    # 1) If the flat image folder isn't there yet, pull it from the training split.
    if not os.path.isdir(args.images_dir):
        ds = ImgDataset(args.train_txt, args.train_dir, transform=None)
        classes = {}
        for filename, label in ds.samples:
            classes.setdefault(label, []).append(filename)

        per_class = max(1, args.num_calib // len(classes))
        os.makedirs(args.images_dir, exist_ok=True)

        used = 0
        for label in sorted(classes):
            batch = classes[label][:per_class]
            for fn in batch:
                src = ds._resolve_path(fn)
                dst = os.path.join(args.images_dir, f"calib_{used:05d}.JPEG")
                shutil.copyfile(src, dst)
                used += 1
        print(f"Collected {used} calibration images into {args.images_dir}")
    else:
        used = len(os.listdir(args.images_dir))
        print(f"Using existing {used} calibration images in {args.images_dir}")

    # 2) Run Qubee's standard preprocessor (bakes resize/normalize/HWC).
    make_calib(
        args_pre=os.path.abspath(args.preproc_yaml),
        data_dir=os.path.abspath(args.images_dir),
        save_dir=os.path.abspath(args.calib_dir),
        save_name="resnet20_cali",
        max_size=used,
    )
    print(f"Calibration data ready in {args.calib_dir} (training-split based)")


if __name__ == "__main__":
    main()