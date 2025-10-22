"""A compact UNet backbone for diffusion training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    exponent = torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
    exponent = -math.log(10000.0) * exponent / max(half_dim - 1, 1)
    emb = torch.exp(exponent)[None, :] * timesteps[:, None]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden, hidden)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding(t, self.linear1.in_features)
        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        time_term = self.time_proj(time_emb)[:, :, None, None]
        h = h + time_term
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, channel_mults: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.time_embed = TimeEmbedding(dim=64, hidden=base_channels * 4)

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        time_dim = base_channels * 4
        self.downs = nn.ModuleList()
        in_ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            block1 = ResidualBlock(in_ch, out_ch, time_dim)
            block2 = ResidualBlock(out_ch, out_ch, time_dim)
            down = Downsample(out_ch) if i < len(channel_mults) - 1 else None
            self.downs.append(nn.ModuleDict({"block1": block1, "block2": block2, "down": down}))
            in_ch = out_ch

        self.mid1 = ResidualBlock(in_ch, in_ch, time_dim)
        self.mid2 = ResidualBlock(in_ch, in_ch, time_dim)

        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            block1 = ResidualBlock(in_ch + out_ch, out_ch, time_dim)
            block2 = ResidualBlock(out_ch, out_ch, time_dim)
            up = Upsample(out_ch) if i > 0 else None
            self.ups.append(nn.ModuleDict({"block1": block1, "block2": block2, "up": up}))
            in_ch = out_ch

        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        time_emb = self.time_embed(t)
        h = self.in_conv(x)

        skips: list[torch.Tensor] = []
        for module in self.downs:
            h = module["block1"](h, time_emb)
            h = module["block2"](h, time_emb)
            skips.append(h)
            if module["down"] is not None:
                h = module["down"](h)

        h = self.mid1(h, time_emb)
        h = self.mid2(h, time_emb)

        for module in self.ups:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = module["block1"](h, time_emb)
            h = module["block2"](h, time_emb)
            if module["up"] is not None:
                h = module["up"](h)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h

