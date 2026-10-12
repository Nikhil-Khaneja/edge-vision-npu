"""Copy a spread of TRAINING LR images into a flat folder for Qubee calibration.

Calibration images must be model INPUTS (LR), taken from the training split.
Runs anywhere (no qubee needed):
    python collect_calib_images.py --num 200 --out calib_images
"""

import argparse
import os
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr-dir", default="data/train/LR2")
    p.add_argument("--out", default="calib_images")
    p.add_argument("--num", type=int, default=200)
    args = p.parse_args()

    names = sorted(f for f in os.listdir(args.lr_dir) if not f.startswith("."))
    # Evenly spaced picks so the calibration set covers the whole training set.
    stride = max(1, len(names) // args.num)
    picks = names[::stride][:args.num]
    os.makedirs(args.out, exist_ok=True)
    for i, n in enumerate(picks):
        shutil.copyfile(os.path.join(args.lr_dir, n), os.path.join(args.out, f"calib_{i:05d}.jpg"))
    print(f"Copied {len(picks)} training LR images into {args.out}/")


if __name__ == "__main__":
    main()
