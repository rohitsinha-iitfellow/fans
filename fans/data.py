"""Dataset utilities for training diffusion models."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None


def _resolve_image_paths(path: str) -> list[Path]:
    if any(ch in path for ch in "*?[]"):
        return [Path(p) for p in sorted(glob.glob(path))]
    p = Path(path)
    if p.is_dir():
        candidates = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            candidates.extend(sorted(p.glob(ext)))
        return sorted(candidates)
    if p.is_file():
        return [p]
    raise FileNotFoundError(path)


def _open_image(path: Path) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")


def _transform(image: Image.Image, image_size: int) -> torch.Tensor:
    if image_size is not None:
        image = image.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = tensor * 2.0 - 1.0
    return tensor


class ImageDataset(Dataset):
    """A lightweight dataset wrapper for local folders or Hugging Face datasets."""

    def __init__(
        self,
        *,
        data: str,
        image_size: int,
        split: str | None = None,
        column: str = "image",
    ) -> None:
        self.data_spec = data
        self.image_size = image_size
        self.split = split or "train"
        self.column = column

        if data.startswith("local:"):
            self.mode = "local"
            path = data[len("local:") :]
            self.paths = _resolve_image_paths(path)
            if not self.paths:
                raise RuntimeError(f"No images found for pattern: {path}")
        elif data.startswith("imagefolder:"):
            self.mode = "local"
            path = data[len("imagefolder:") :]
            self.paths = _resolve_image_paths(os.path.join(path, "**", "*"))
            if not self.paths:
                raise RuntimeError(f"ImageFolder empty: {path}")
        elif os.path.exists(data):
            self.mode = "local"
            self.paths = _resolve_image_paths(os.path.join(data, "**", "*"))
            if not self.paths:
                raise RuntimeError(f"No images in directory: {data}")
        else:
            if load_dataset is None:
                raise RuntimeError("datasets library not available to load remote datasets")
            self.mode = "hf"
            parts = data.split(":")
            name = parts[0]
            subset = parts[1] if len(parts) > 1 else None
            self.dataset = load_dataset(name, subset, split=self.split)

    def __len__(self) -> int:
        if getattr(self, "mode", "local") == "local":
            return len(self.paths)
        return len(self.dataset)  # type: ignore[attr-defined]

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.mode == "local":
            image = _open_image(self.paths[index])
        else:
            record = self.dataset[index]  # type: ignore[attr-defined]
            image = record[self.column]
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                raise TypeError(f"Unsupported HF dataset type: {type(image)!r}")
        return _transform(image, self.image_size)

    # ------------------------------------------------------------------
    @cached_property
    def pil_images(self) -> list[Image.Image]:
        """Return a list of PIL images for spectral analysis."""

        if self.mode == "local":
            return [_open_image(path) for path in self.paths]
        images = []
        for record in self.dataset:  # type: ignore[attr-defined]
            img = record[self.column]
            if isinstance(img, Image.Image):
                images.append(img.convert("RGB"))
            elif isinstance(img, np.ndarray):
                images.append(Image.fromarray(img))
        return images

