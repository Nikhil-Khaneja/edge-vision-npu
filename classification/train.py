import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImgDataset
from model import ResNet20, count_parameters

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(240, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Validation uses a fixed center crop to keep evaluation deterministic.
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(240),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, val_transform


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


def main():
    # Anchor default relative paths to this file's directory so the script
    # works regardless of the current working directory.
    ROOT = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Train ResNet20 on ImageNet-20")
    parser.add_argument("--train-txt", default=os.path.join(ROOT, "dataset/imagenet_train20.txt"))
    parser.add_argument("--train-dir", default=os.path.join(ROOT, "dataset/imagenet_train20a"))
    parser.add_argument("--val-txt", default=os.path.join(ROOT, "dataset/imagenet_val20.txt"))
    parser.add_argument("--val-dir", default=os.path.join(ROOT, "dataset/imagenet_val20"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="checkpoint.pth")
    parser.add_argument("--calib-out", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    train_transform, val_transform = get_transforms()

    train_ds = ImgDataset(args.train_txt, args.train_dir, train_transform)
    val_ds = ImgDataset(args.val_txt, args.val_dir, val_transform)
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = ResNet20(num_classes=20).to(device)
    print(f"Model parameter count: {count_parameters(model):,}")

    # Cosine annealing from a warm base LR gives smooth convergence for CNNs.
    base_lr = args.lr
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train acc {train_acc*100:6.2f}% | Loss {train_loss:.4f} | "
            f"Val acc {val_acc*100:6.2f}% | LR {lr:.5f} | {elapsed:.1f}s"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val,
                },
                args.out,
            )
            print(f"  -> saved checkpoint (val {val_acc*100:.2f}%)")

    print(f"Best validation accuracy: {best_val*100:.2f}%")


if __name__ == "__main__":
    main()
