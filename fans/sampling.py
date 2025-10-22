"""Sampling helpers for DDPM style models."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn

from .fans_noise import FANSNoiseShaper
from .scheduler import DiffusionSchedule


@torch.no_grad()
def p_sample(
    model: nn.Module,
    x_t: torch.Tensor,
    t: int,
    schedule: DiffusionSchedule,
    fans: Optional[FANSNoiseShaper] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    betas = schedule.betas.to(x_t.device)
    sqrt_one_minus_alpha_bar = schedule.sqrt_one_minus_alphas_cumprod.to(x_t.device)
    sqrt_alpha_t_inv = torch.rsqrt(schedule.alphas.to(x_t.device))
    beta_t = betas[t]
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar[t]
    sqrt_alpha_t_inv_t = sqrt_alpha_t_inv[t]

    t_tensor = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
    eps_theta = model(x_t, t_tensor)
    x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps_theta) / schedule.sqrt_alphas_cumprod.to(x_t.device)[t]
    coef_eps = beta_t / sqrt_one_minus_alpha_bar_t
    mean = sqrt_alpha_t_inv_t * (x_t - coef_eps * eps_theta)

    if t == 0:
        return mean

    posterior_var = schedule.posterior_variance.to(x_t.device)[t]
    sigma = torch.sqrt(posterior_var)

    if fans is not None:
        noise = fans.fans_noise(x_t, sigma, torch.tensor((t) / (schedule.timesteps - 1), device=x_t.device))
    else:
        noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
    return mean + sigma * noise


@torch.no_grad()
def sample_loop(
    model: nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, int, int, int],
    *,
    fans: Optional[FANSNoiseShaper] = None,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    device = device or next(model.parameters()).device
    x = torch.randn(shape, device=device, generator=generator)
    for t in reversed(range(schedule.timesteps)):
        x = p_sample(model, x, t, schedule, fans=fans, generator=generator)
    return torch.clamp(x, -1.0, 1.0)

