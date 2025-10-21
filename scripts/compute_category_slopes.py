"""Batch computation of radial power spectrum slopes for curated categories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from datasets import Dataset, Image, load_dataset

from fans.spectral import SpectrumStats, compute_dataset_slopes


FilterFn = Callable[[dict], bool]


def _ensure_image_column(ds: Dataset, column: str) -> Dataset:
    """Cast the requested column to :class:`datasets.Image` if needed."""

    if column in ds.column_names:
        return ds.cast_column(column, Image())
    raise ValueError(f"Column '{column}' not found in dataset columns: {ds.column_names}")


@dataclass
class CategoryConfig:
    """Configuration for a dataset category to analyse."""

    name: str
    dataset_id: str
    split: str = "train"
    column: str = "image"
    sample_size: Optional[int] = 200
    filter_fn: Optional[FilterFn] = None
    target_size: Optional[int] = 256
    min_freq: float = 0.05
    max_freq: float = 0.5
    load_kwargs: dict[str, object] = field(default_factory=dict)

    def load(self) -> Dataset:
        ds = load_dataset(self.dataset_id, split=self.split, **self.load_kwargs)
        ds = _ensure_image_column(ds, self.column)
        if self.filter_fn is not None:
            ds = ds.filter(self.filter_fn, batched=False, num_proc=1)
        return ds


def _caption_contains(*needles: str) -> FilterFn:
    lowered = tuple(needle.lower() for needle in needles)

    def predicate(example: dict) -> bool:
        caption = example.get("caption")
        if not isinstance(caption, str):
            return False
        text = caption.lower()
        return any(needle in text for needle in lowered)

    return predicate


def _caption_contains_all(*needles: str) -> FilterFn:
    lowered = tuple(needle.lower() for needle in needles)

    def predicate(example: dict) -> bool:
        caption = example.get("caption")
        if not isinstance(caption, str):
            return False
        text = caption.lower()
        return all(needle in text for needle in lowered)

    return predicate


CATEGORIES: list[CategoryConfig] = [
    CategoryConfig(
        name="sem_micrographs",
        dataset_id="kvriza8/microscopy_images",
        filter_fn=_caption_contains("sem", "scanning electron"),
    ),
    CategoryConfig(
        name="crystal_lattice_tems",
        dataset_id="kvriza8/microscopy_images",
        filter_fn=_caption_contains("tem", "transmission electron"),
    ),
    CategoryConfig(
        name="materials_project_micrographs",
        dataset_id="kvriza8/microscopy_images",
        filter_fn=_caption_contains("materials project", "mp-"),
        sample_size=None,
    ),
    CategoryConfig(
        name="pcb_inspection",
        dataset_id="Francesco/printed-circuit-board",
        sample_size=200,
    ),
    CategoryConfig(
        name="resisc45",
        dataset_id="timm/resisc45",
        sample_size=200,
    ),
    CategoryConfig(
        name="aid",
        dataset_id="jonathan-roberts1/AID_MultiLabel",
        sample_size=200,
    ),
    CategoryConfig(
        name="spacenet",
        dataset_id="imagefolder",
        split="train",
        sample_size=200,
        load_kwargs={"data_dir": "/tmp/spacenet9/train"},
    ),
    CategoryConfig(
        name="kth_tips2",
        dataset_id="imagefolder",
        sample_size=200,
        load_kwargs={"data_dir": "/tmp/kthtips2b/KTH-TIPS2-b"},
    ),
    CategoryConfig(
        name="adobe_textures",
        dataset_id="imagefolder",
        sample_size=None,
        load_kwargs={"data_dir": "/tmp/adobe_textures/train"},
    ),
]


def summarise(category: CategoryConfig, stats: SpectrumStats) -> dict:
    return {
        "category": category.name,
        "dataset": category.dataset_id,
        "split": category.split,
        "mean_slope": stats.mean,
        "std_slope": stats.std,
        "valid_images": stats.valid_count,
        "total_images": stats.total_count,
        "sample_size": category.sample_size,
        "target_size": category.target_size,
        "min_freq": category.min_freq,
        "max_freq": category.max_freq,
    }


def main() -> None:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for config in CATEGORIES:
        dataset = config.load()
        stats = compute_dataset_slopes(
            dataset,
            column=config.column,
            sample_size=config.sample_size,
            target_size=config.target_size,
            min_freq=config.min_freq,
            max_freq=config.max_freq,
            progress=True,
        )
        summary = summarise(config, stats)
        summaries.append(summary)

        out_path = results_dir / f"{config.name}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(json.dumps(summary, indent=2))

    aggregate_path = results_dir / "categories_summary.json"
    with aggregate_path.open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2)


if __name__ == "__main__":
    main()
