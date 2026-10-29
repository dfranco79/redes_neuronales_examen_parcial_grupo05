"""U-Net implementada desde cero en PyTorch (sin librerías de segmentación).

Encoder: DoubleConv + `depth` bloques (MaxPool 2x2 -> DoubleConv), con base_ch * 2^i canales.
Decoder: upsampling bilineal x2, concatenación con la conexión de salto y DoubleConv.
Salida: conv 1x1 con un logit por píxel (agua); la sigmoide se aplica en la pérdida y en la inferencia.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_ch=32, depth=4):
        super().__init__()
        chs = [base_ch * 2 ** i for i in range(depth + 1)]
        self.inc = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList([Down(chs[i], chs[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList([Up(chs[i + 1] + chs[i], chs[i]) for i in range(depth - 1, -1, -1)])
        self.out_conv = nn.Conv2d(chs[0], out_channels, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips.pop()                      # el nivel más profundo es el cuello de botella
        for up in self.ups:
            x = up(x, skips.pop())
        return self.out_conv(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
