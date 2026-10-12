"""Super-resolution / restoration models.

Contract: input [1, 3, 256, 256] float in 0..1  ->  output [1, 3, 256, 256].
All models predict a residual that is added to the input (global skip).
Only NPU-friendly ops are used: Conv2d, ReLU, Add, Concat, PixelShuffle.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """x + Conv3x3(ReLU(Conv3x3(x))). No BatchNorm."""

    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


def _scale_residual_init(model, scale=0.1):
    # Shrink the last conv of every block so the untrained net starts close to
    # identity. Equivalent to EDSR "residual scaling" without an extra Mul op.
    for m in model.modules():
        if isinstance(m, ResBlock):
            m.conv2.weight.data.mul_(scale)
            m.conv2.bias.data.zero_()


class PlainResNet(nn.Module):
    """Candidate A: every block at full 256x256 resolution."""

    def __init__(self, c=64, n_blocks=16):
        super().__init__()
        self.head = nn.Conv2d(3, c, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(c) for _ in range(n_blocks)])
        self.body_tail = nn.Conv2d(c, c, 3, padding=1)
        self.tail = nn.Conv2d(c, 3, 3, padding=1)
        _scale_residual_init(self)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        f = self.head(x)
        f = f + self.body_tail(self.body(f))
        return x + self.tail(f)


class DualScaleSR(nn.Module):
    """Candidate B: fine branch at 256x256, deep body at 128x128, fused back."""

    def __init__(self, c1=64, c2=96, n_fine=2, n_deep=12, n_refine=2):
        super().__init__()
        self.head = nn.Conv2d(3, c1, 3, padding=1)
        self.fine = nn.Sequential(*[ResBlock(c1) for _ in range(n_fine)])
        self.down = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.deep = nn.Sequential(*[ResBlock(c2) for _ in range(n_deep)])
        self.deep_tail = nn.Conv2d(c2, c2, 3, padding=1)
        self.up = nn.Sequential(nn.Conv2d(c2, c1 * 4, 3, padding=1), nn.PixelShuffle(2))
        self.fuse = nn.Conv2d(2 * c1, c1, 1)
        self.refine = nn.Sequential(*[ResBlock(c1) for _ in range(n_refine)])
        self.tail = nn.Conv2d(c1, 3, 3, padding=1)
        _scale_residual_init(self)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        f = self.fine(self.head(x))
        d = self.down(f)
        d = d + self.deep_tail(self.deep(d))  # long skip
        u = self.up(d)
        y = self.refine(self.fuse(torch.cat([f, u], 1)))
        return x + self.tail(y)  # global residual


class ResUNet(nn.Module):
    """Candidate C: 3 levels (256 -> 128 -> 64) with residual blocks and skip concats."""

    def __init__(self, c=(48, 96, 160), n=(2, 3, 6)):
        super().__init__()
        c1, c2, c3 = c
        self.head = nn.Conv2d(3, c1, 3, padding=1)
        self.enc1 = nn.Sequential(*[ResBlock(c1) for _ in range(n[0])])
        self.down1 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(*[ResBlock(c2) for _ in range(n[1])])
        self.down2 = nn.Sequential(nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.mid = nn.Sequential(*[ResBlock(c3) for _ in range(n[2])])
        self.up2 = nn.Sequential(nn.Conv2d(c3, c2 * 4, 3, padding=1), nn.PixelShuffle(2))
        self.fuse2 = nn.Conv2d(2 * c2, c2, 1)
        self.dec2 = nn.Sequential(*[ResBlock(c2) for _ in range(n[1])])
        self.up1 = nn.Sequential(nn.Conv2d(c2, c1 * 4, 3, padding=1), nn.PixelShuffle(2))
        self.fuse1 = nn.Conv2d(2 * c1, c1, 1)
        self.dec1 = nn.Sequential(*[ResBlock(c1) for _ in range(n[0])])
        self.tail = nn.Conv2d(c1, 3, 3, padding=1)
        _scale_residual_init(self)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        e1 = self.enc1(self.head(x))
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        d2 = self.dec2(self.fuse2(torch.cat([e2, self.up2(m)], 1)))
        d1 = self.dec1(self.fuse1(torch.cat([e1, self.up1(d2)], 1)))
        return x + self.tail(d1)


ARCHS = {
    "plain": PlainResNet,
    "dual": DualScaleSR,
    "unet": ResUNet,
}


def build_model(arch, **kwargs):
    return ARCHS[arch](**kwargs)


if __name__ == "__main__":
    for name in ARCHS:
        m = build_model(name).eval()
        x = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            y = m(x)
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{name:6s} params={n_params:.2f}M  out={tuple(y.shape)}  "
              f"identity_at_init={torch.allclose(x, y)}")
