"""Frequency-Aware Adaptive Noise Scheduling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass
class FANSConfig:
    """Configuration values required to instantiate :class:`FANSNoiseShaper`."""

    bands: Sequence[torch.BoolTensor]
    g_b: torch.Tensor
    beta: float = 1.0
    gamma: float = 1.0
    ramp: str = "linear"


class FANSNoiseShaper:
    """Generate FANS shaped Gaussian noise for diffusion models.

    Parameters
    ----------
    bands:
        Sequence of boolean tensors representing annular frequency bands in the
        ``rFFT`` layout. Each mask must be broadcastable to ``(N, C, H, W//2+1)``.
    g_b:
        1D tensor with dataset specific importance weights for each band. Larger
        values receive more power during the early steps.
    beta, gamma:
        Scalars that balance dataset importance against the temporal ramping
        factor.
    ramp:
        One of ``{"linear", "sigmoid"}`` controlling how :math:`\phi(t)` is
        computed.
    """

    def __init__(
        self,
        bands: Sequence[torch.BoolTensor],
        g_b: torch.Tensor,
        beta: float = 1.0,
        gamma: float = 1.0,
        ramp: str = "linear",
    ) -> None:
        if len(bands) == 0:
            raise ValueError("FANSNoiseShaper requires at least one band mask")
        self.bands = [mask.clone().bool() for mask in bands]
        self.g_b = torch.as_tensor(g_b, dtype=torch.float32)
        if self.g_b.ndim != 1 or self.g_b.numel() != len(self.bands):
            raise ValueError("g_b must be a 1D tensor with the same length as bands")
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.ramp = ramp

    # ------------------------------------------------------------------
    def _phi(self, t01: torch.Tensor) -> torch.Tensor:
        if self.ramp == "sigmoid":
            return torch.sigmoid(6.0 * (t01 - 0.5))
        return t01

    def weights_at(self, t01: torch.Tensor) -> torch.Tensor:
        """Return softmax-normalised band weights for time ``t01``."""

        t01 = torch.as_tensor(t01, dtype=torch.float32, device=self.g_b.device)
        if t01.ndim == 0:
            t01 = t01.unsqueeze(0)
        lamb = torch.linspace(0.0, 1.0, steps=len(self.bands), device=t01.device)
        logits = self.beta * self.g_b.to(t01.device)[None, :] - self.gamma * (
            lamb[None, :] * self._phi(t01)[:, None]
        )
        weights = torch.softmax(logits, dim=-1)
        return weights.squeeze(0) if weights.shape[0] == 1 else weights

    # ------------------------------------------------------------------
    @torch.no_grad()
    def fans_noise(self, x: torch.Tensor, sigma_t: torch.Tensor, t01: torch.Tensor) -> torch.Tensor:
        """Return FANS-shaped Gaussian noise.

        Parameters
        ----------
        x:
            Input tensor with shape ``(N, C, H, W)`` in either pixel space or
            latent space.
        sigma_t:
            Scalar variance parameter for the current step. ``sigma_t`` can be a
            float, a tensor broadcastable to ``(N,)`` or ``(N, 1, 1, 1)``.
        t01:
            Normalised time in :math:`[0, 1]`.
        """

        if x.ndim != 4:
            raise ValueError("Expected x to have shape (N, C, H, W)")

        device = x.device
        dtype = x.dtype
        eps = torch.randn_like(x)
        eps_f = torch.fft.rfftn(eps, dim=(-2, -1))

        sigma = torch.as_tensor(sigma_t, dtype=dtype, device=device)
        sigma = sigma.view(-1, *([1] * (x.ndim - 1)))

        t_tensor = torch.as_tensor(t01, dtype=dtype, device=device)
        t_tensor = t_tensor.view(-1, *([1] * (x.ndim - 1)))
        weights = self.weights_at(t_tensor.flatten())
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)

        mask_stack = torch.stack([mask.to(device=device) for mask in self.bands], dim=0)
        mask_stack = mask_stack.unsqueeze(0).unsqueeze(0)  # 1,1,B,H,W//2+1

        # Build the frequency dependent variance map
        variance = torch.sum(
            weights[:, None, :, None, None] * mask_stack.to(dtype), dim=2
        )
        variance = variance * sigma**2

        eps_f = eps_f * torch.sqrt(variance + 1e-12)
        shaped = torch.fft.irfftn(eps_f, s=x.shape[-2:], dim=(-2, -1))
        return shaped


def build_fans_from_state(
    *,
    bands: Iterable[torch.Tensor],
    g_b: torch.Tensor,
    beta: float,
    gamma: float,
    ramp: str = "linear",
) -> FANSNoiseShaper:
    """Factory helper used when loading saved band state from disk."""

    return FANSNoiseShaper(list(bands), torch.as_tensor(g_b), beta=beta, gamma=gamma, ramp=ramp)

