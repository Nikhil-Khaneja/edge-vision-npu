import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two 3x3 conv residual block with optional downsampling."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    """Small residual CNN built from scratch for 240x240x3 input, 20 classes.

    Downsampling path (with/without shortcut):
        input 240x240x3
          -> conv3x3 s2            -> 120x120x32
          -> BasicBlock  s2        -> 60x60x64
          -> BasicBlock  s2        -> 30x30x128
          -> BasicBlock  s2        -> 15x15x256
          -> global avg pool       -> 256
          -> fc                   -> 20
    """

    def __init__(self, num_classes=20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        # Kaiming weight init (no pretrained weights used)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, planes, blocks, stride):
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
