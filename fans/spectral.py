"""Spectral analysis utilities for Frequency-Aware Adaptive Noise Scheduling."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass
class SpectrumStats:
    """Summary statistics for a collection of spectral slopes."""

    slopes: list[float]
    mean: float
    std: float
    valid_count: int
    total_count: int


_RGB_TO_GRAY = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)


def _to_pil(image: Image.Image | np.ndarray | dict) -> Image.Image:
    """Convert a Hugging Face image representation to :class:`PIL.Image`."""

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        return Image.fromarray(arr)
    if isinstance(image, dict):
        if "bytes" in image:
            return Image.open(io.BytesIO(image["bytes"]))
        if "path" in image:
            return Image.open(image["path"])
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _prepare_array(image: Image.Image) -> np.ndarray:
    """Convert an image to a zero-mean float32 numpy array."""

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr @ _RGB_TO_GRAY
    if arr.max() > 1.0:
        arr /= 255.0
    arr -= np.mean(arr)
    return arr


def _radial_profile(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radial mean of the 2D power spectrum."""

    height, width = power.shape
    y, x = np.indices((height, width))
    center = np.array([(height - 1) / 2.0, (width - 1) / 2.0])
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r_int = np.floor(r).astype(np.int32)

    flat_power = power.ravel()
    r_flat = r_int.ravel()
    tbin = np.bincount(r_flat, weights=flat_power)
    nr = np.bincount(r_flat)
    radial_mean = tbin / np.maximum(nr, 1)

    radii = np.arange(len(radial_mean))
    max_radius = min(height, width) // 2
    valid = radii <= max_radius
    return radii[valid], radial_mean[valid]


def compute_radial_power_spectrum(
    image: Image.Image,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalized radial frequencies and power for an image."""

    arr = _prepare_array(image)
    fft = np.fft.fft2(arr)
    power = np.abs(np.fft.fftshift(fft)) ** 2
    radii, radial_mean = _radial_profile(power)
    if len(radii) == 0:
        return radii, radial_mean
    max_radius = radii[-1] if radii[-1] > 0 else 1
    freqs = radii / max_radius
    return freqs, radial_mean


def fit_power_law(
    freqs: np.ndarray,
    power: np.ndarray,
    min_freq: float = 0.05,
    max_freq: float = 0.5,
) -> Optional[tuple[float, float]]:
    """Fit a line to the log-log power spectrum and return slope and intercept."""

    if len(freqs) == 0:
        return None
    mask = (freqs >= min_freq) & (freqs <= max_freq) & (power > 0)
    if not np.any(mask):
        return None
    x = np.log(freqs[mask])
    y = np.log(power[mask])
    if x.size < 2:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def compute_image_slope(
    image: Image.Image | np.ndarray | dict,
    target_size: Optional[int] = 256,
    min_freq: float = 0.05,
    max_freq: float = 0.5,
) -> Optional[float]:
    """Compute the log-log slope of the radial power spectrum for a single image."""

    pil_image = _to_pil(image)
    if target_size is not None:
        pil_image = pil_image.resize((target_size, target_size), Image.BICUBIC)
    freqs, power = compute_radial_power_spectrum(pil_image)
    fit = fit_power_law(freqs, power, min_freq=min_freq, max_freq=max_freq)
    if fit is None:
        return None
    slope, _ = fit
    return slope


def compute_dataset_slopes(
    dataset: Sequence,
    *,
    column: str = "image",
    sample_size: Optional[int] = None,
    target_size: Optional[int] = 256,
    min_freq: float = 0.05,
    max_freq: float = 0.5,
    progress: bool = True,
) -> SpectrumStats:
    """Compute spectral slopes for each example in a dataset."""

    if sample_size is not None and sample_size < len(dataset):
        indices = list(range(sample_size))
        data_iter = (dataset[i] for i in indices)
        total = sample_size
    else:
        data_iter = iter(dataset)
        total = len(dataset)

    slopes: list[float] = []
    iterator: Iterable = tqdm(data_iter, total=total, disable=not progress)
    processed = 0
    for example in iterator:
        processed += 1
        if isinstance(example, dict):
            image = example[column]
        else:
            image = example
        slope = compute_image_slope(
            image,
            target_size=target_size,
            min_freq=min_freq,
            max_freq=max_freq,
        )
        if slope is not None:
            slopes.append(slope)
    iterator.close()

    if slopes:
        mean = float(np.mean(slopes))
        std = float(np.std(slopes))
    else:
        mean = float("nan")
        std = float("nan")

    return SpectrumStats(
        slopes=slopes,
        mean=mean,
        std=std,
        valid_count=len(slopes),
        total_count=processed,
    )
