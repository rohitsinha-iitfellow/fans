"""Cosine diffusion schedule and step coefficients."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)


@dataclass
class DiffusionSchedule:
    timesteps: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor

    @classmethod
    def create(cls, timesteps: int) -> "DiffusionSchedule":
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        posterior_variance = betas * torch.cat([torch.tensor([1.0]), 1 - alphas_cumprod[:-1]], dim=0) / (
            1 - alphas_cumprod
        )
        posterior_variance[0] = betas[0]
        return cls(
            timesteps=timesteps,
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            posterior_variance=posterior_variance,
        )

    def to(self, device: torch.device) -> "DiffusionSchedule":
        return DiffusionSchedule(
            timesteps=self.timesteps,
            betas=self.betas.to(device),
            alphas=self.alphas.to(device),
            alphas_cumprod=self.alphas_cumprod.to(device),
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod.to(device),
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod.to(device),
            posterior_variance=self.posterior_variance.to(device),
        )

