"""Helper functions for creating radial Fourier band masks and dataset weights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import torch


def _ensure_pil(img: Image.Image | np.ndarray | str | Path) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, (str, Path)):
        return Image.open(img)
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            return Image.fromarray(img)
        if img.ndim == 2:
            return Image.fromarray(img)
    raise TypeError(f"Unsupported image type: {type(img)!r}")


def _to_grayscale_array(img: Image.Image, image_size: int) -> np.ndarray:
    img = img.convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    arr = arr - np.mean(arr)
    return arr


def radial_frequency_masks(height: int, width: int, bands: int, device: torch.device | None = None) -> list[torch.BoolTensor]:
    """Return boolean masks for concentric frequency bands in rFFT layout."""

    if bands < 1:
        raise ValueError("bands must be >= 1")

    fy = torch.fft.fftfreq(height, d=1.0, device=device)
    fx = torch.fft.rfftfreq(width, d=1.0, device=device)
    grid_y = fy[:, None]
    grid_x = fx[None, :]
    radius = torch.sqrt(grid_x**2 + grid_y**2)
    max_r = radius.max()
    edges = torch.linspace(0.0, max_r + 1e-8, steps=bands + 1, device=device)

    masks: list[torch.BoolTensor] = []
    for b in range(bands):
        mask = (radius >= edges[b]) & (radius <= edges[b + 1])
        if b < bands - 1:
            mask &= radius < edges[b + 1]
        if not torch.any(mask):
            # Guarantee that each band has at least one frequency by expanding to
            # the closest available bin.
            flat_idx = torch.argmin(torch.abs(radius - 0.5 * (edges[b] + edges[b + 1])))
            mask.view(-1)[flat_idx] = True
        masks.append(mask)
    return masks


@dataclass
class DatasetSpectrum:
    band_energy: torch.Tensor
    pi_b: torch.Tensor
    g_b: torch.Tensor


def compute_dataset_band_profile(
    images: Sequence[Image.Image | np.ndarray | str | Path],
    *,
    num_bands: int,
    image_size: int,
    sample_size: int | None = None,
    alpha: float = 0.5,
) -> DatasetSpectrum:
    """Estimate the dataset energy allocation across Fourier bands."""

    if sample_size is not None:
        images = images[: sample_size]

    masks_np = [
        mask.cpu().numpy() for mask in radial_frequency_masks(image_size, image_size, num_bands, device=torch.device("cpu"))
    ]
    accum = np.zeros(num_bands, dtype=np.float64)

    for img_like in images:
        pil = _ensure_pil(img_like)
        arr = _to_grayscale_array(pil, image_size)
        fft = np.fft.rfftn(arr)
        power = np.abs(fft) ** 2
        for idx, mask in enumerate(masks_np):
            accum[idx] += power[mask].sum()

    total = accum.sum()
    if total <= 0:
        pi_b = np.ones_like(accum) / len(accum)
    else:
        pi_b = accum / total

    g_b = (pi_b + 1e-12) ** (-alpha)
    g_b = g_b / np.linalg.norm(g_b, ord=1)

    return DatasetSpectrum(
        band_energy=torch.from_numpy(accum.astype(np.float32)),
        pi_b=torch.from_numpy(pi_b.astype(np.float32)),
        g_b=torch.from_numpy(g_b.astype(np.float32)),
    )


def ensure_band_state(
    *,
    num_bands: int,
    image_size: int,
    existing_masks: Iterable[torch.Tensor] | None,
) -> list[torch.BoolTensor]:
    """Create masks if they are missing or mismatched."""

    if existing_masks is None:
        return radial_frequency_masks(image_size, image_size, num_bands)
    masks = [mask.bool() for mask in existing_masks]
    expected_shape = (image_size, image_size // 2 + 1)
    if any(mask.shape != expected_shape for mask in masks):
        return radial_frequency_masks(image_size, image_size, num_bands)
    if len(masks) != num_bands:
        return radial_frequency_masks(image_size, image_size, num_bands)
    return masks

