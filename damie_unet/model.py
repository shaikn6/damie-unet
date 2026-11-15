"""U-Net encoder/decoder with skip connections (Ronneberger et al., 2015)."""

import torch
from torch import nn
from torch.nn import functional as F


def double_conv(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Four pooling levels, transposed convolutions, and two output logits.

    Padded convolutions preserve resolution; inputs are padded to multiples of
    16 and output padding is removed, including for full odd-sized images.
    """

    def __init__(self, width=64):
        super().__init__()
        widths = [width * 2**i for i in range(5)]
        self.encoders = nn.ModuleList([
            double_conv(3 if i == 0 else widths[i - 1], w)
            for i, w in enumerate(widths)
        ])
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2)
            for i in reversed(range(4))
        ])
        self.decoders = nn.ModuleList([
            double_conv(widths[i] * 2, widths[i]) for i in reversed(range(4))
        ])
        self.head = nn.Conv2d(width, 2, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 16, 0, (-h) % 16))
        skips = []
        for encoder in self.encoders[:-1]:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.encoders[-1](x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = decoder(torch.cat([up(x), skip], dim=1))
        return self.head(x)[..., :h, :w]
