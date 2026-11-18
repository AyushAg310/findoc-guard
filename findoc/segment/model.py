"""DocSegNet: a small U-Net that predicts "document" vs "background" for every pixel.

A **U-Net** has an encoder that halves the resolution step by step while increasing the number
of channels (learning *what* is in the image), and a decoder that upsamples back to full
resolution (recovering *where*). **Skip connections** concatenate each encoder level with the
decoder level of the same size, so fine detail such as the exact position of the paper edge is
not lost in the bottleneck.

This one is deliberately small (4 levels, 16-128 channels, about 1.2 M parameters, 256x256
input): documents are large, simple shapes, and it must run in a few milliseconds per video
frame on a CPU through OpenCV DNN. Upsampling is bilinear followed by a convolution (instead of
transposed convolutions) to avoid checkerboard artefacts and to keep the ONNX graph simple.
"""

from __future__ import annotations

import torch
from torch import nn

INPUT_SIZE = 256


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class DocSegNet(nn.Module):
    def __init__(self, widths: tuple[int, ...] = (16, 32, 64, 128)) -> None:
        super().__init__()
        self.enc = nn.ModuleList()
        cin = 3
        for w in widths:
            self.enc.append(block(cin, w))
            cin = w
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(widths[-1], widths[-1])
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec = nn.ModuleList()
        rev = list(reversed(widths))
        cin = widths[-1]
        for w in rev:
            self.dec.append(block(cin + w, w))
            cin = w
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for dec, skip in zip(self.dec, reversed(skips), strict=True):
            x = dec(torch.cat([self.up(x), skip], dim=1))
        return self.head(x)  # logits; sigmoid gives P(document)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
