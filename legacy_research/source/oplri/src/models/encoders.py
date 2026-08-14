from __future__ import annotations

import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    def __init__(self, in_ch: int, branch_ch: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, branch_ch, kernel_size=1)
        self.k3 = nn.Conv1d(branch_ch, branch_ch, kernel_size=3, padding=1)
        self.k7 = nn.Conv1d(branch_ch, branch_ch, kernel_size=7, padding=3)
        self.k11 = nn.Conv1d(branch_ch, branch_ch, kernel_size=11, padding=5)
        self.pool = nn.Conv1d(in_ch, branch_ch, kernel_size=1)
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.bn = nn.BatchNorm1d(branch_ch * 4)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bottleneck(x)
        out = torch.cat([self.k3(y), self.k7(y), self.k11(y), self.pool(self.maxpool(x))], dim=1)
        return self.act(self.bn(out))


class InceptionTimeEncoder(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.b1 = InceptionBlock(in_channels, 32)
        self.b2 = InceptionBlock(128, 32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b1(x)
        x = self.b2(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class OSCNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Linear(64, 128)  # enforce [B,128]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.out_proj(x)


class XCMEncoder(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()
        # Branch 1: temporal patterns
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        # Branch 2: channel interactions
        self.spatial = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.temporal(x)
        s = self.spatial(x)
        fused = torch.cat([t, s], dim=1)  # explicit branch concat
        fused = self.pool(fused).flatten(1)
        return self.proj(fused)


class ResBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)


class ResNet1DEncoder(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, 64, 7, padding=3), nn.ReLU())
        self.b1 = ResBlock1D(64)
        self.b2 = ResBlock1D(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(64, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        # causal cut-back to align lengths
        y = y[..., : x.shape[-1]]
        return self.act(y + self.downsample(x))


class TCNEncoder(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.t1 = TemporalBlock(in_channels, 64, kernel_size=3, dilation=1)
        self.t2 = TemporalBlock(64, 64, kernel_size=3, dilation=2)
        self.t3 = TemporalBlock(64, 64, kernel_size=3, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(64, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


def get_dynamic_encoder(name: str, in_channels: int = 2) -> nn.Module:
    key = name.lower()
    if key in ["inceptiontime", "inception"]:
        return InceptionTimeEncoder(in_channels)
    if key in ["os-cnn", "oscnn"]:
        return OSCNNEncoder(in_channels)
    if key == "xcm":
        return XCMEncoder(in_channels)
    if key in ["1d-resnet", "resnet1d", "resnet"]:
        return ResNet1DEncoder(in_channels)
    if key == "tcn":
        return TCNEncoder(in_channels)
    raise ValueError(f"Unsupported encoder: {name}")
