import argparse
import os
import shutil

from dataset import ImgDataset


def main():
    parser = argparse.ArgumentParser(
        description="Collect a flat folder of calibration images from the "
                    "TRAINING split (works anywhere, no Qubee required)"
    )
    ROOT = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--train-txt", default=os.path.join(ROOT, "dataset/imagenet_train20.txt"))
    parser.add_argument("--train-dir", default=os.path.join(ROOT, "dataset/imagenet_train20a"))
    parser.add_argument("--images-dir", default=os.path.join(ROOT, "calib_images"),
                        help="flat folder of raw calibration images")
    parser.add_argument("--num-calib", type=int, default=128,
                        help="total calibration images to use")
    args = parser.parse_args()

    ds = ImgDataset(args.train_txt, args.train_dir, transform=None)
    classes = {}
    for filename, label in ds.samples:
        classes.setdefault(label, []).append(filename)

    per_class = max(1, args.num_calib // len(classes))
    if os.path.isdir(args.images_dir):
        shutil.rmtree(args.images_dir)
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
    print("Zip this folder and run DOCKER-SIDE prepare_calib.py on the NPU "
          "machine to bake preprocessing into the MXQ.")


if __name__ == "__main__":
    main()