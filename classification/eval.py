import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImgDataset
from model import ResNet20

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Per-class accuracy")
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoint.pth"))
    parser.add_argument("--txt", default=os.path.join(ROOT, "dataset/imagenet_val20.txt"))
    parser.add_argument("--dir", default=os.path.join(ROOT, "dataset/imagenet_val20"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(240),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds = ImgDataset(args.txt, args.dir, transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = ResNet20(num_classes=20)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    class_correct = {c: 0 for c in range(20)}
    class_total = {c: 0 for c in range(20)}
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            for lb, pr in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                class_total[lb] += 1
                if lb == pr:
                    class_correct[lb] += 1

    print(f"{'class':>5} {'correct':>8} {'total':>6} {'acc%':>7}")
    overall_correct = 0
    overall_total = 0
    for c in range(20):
        acc = class_correct[c] / max(class_total[c], 1)
        overall_correct += class_correct[c]
        overall_total += class_total[c]
        print(f"{c:5d} {class_correct[c]:8d} {class_total[c]:6d} {acc*100:6.2f}%")
    print(f"Overall accuracy: {overall_correct/max(overall_total,1)*100:.2f}%")


if __name__ == "__main__":
    main()
