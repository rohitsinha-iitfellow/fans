"""CLI for computing radial power spectrum slopes of Hugging Face datasets."""

from __future__ import annotations

import argparse
import json
from typing import Optional

from datasets import Image, load_dataset

from fans.spectral import SpectrumStats, compute_dataset_slopes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset name on the Hugging Face Hub")
    parser.add_argument("--subset", help="Dataset subset/configuration", default=None)
    parser.add_argument("--split", help="Dataset split", default="train")
    parser.add_argument("--column", help="Image column name", default="image")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument(
        "--target-size",
        type=int,
        default=256,
        help="Resize images to this square size before FFT (omit to keep original)",
    )
    parser.add_argument("--min-freq", type=float, default=0.05, help="Minimum normalized frequency for fitting")
    parser.add_argument("--max-freq", type=float, default=0.5, help="Maximum normalized frequency for fitting")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when shuffling")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset before sampling")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading dataset scripts that execute remote code",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the JSON summary",
    )
    return parser.parse_args()


def load_split(args: argparse.Namespace):
    if args.subset is not None:
        dataset = load_dataset(
            args.dataset,
            args.subset,
            split=args.split,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        dataset = load_dataset(
            args.dataset,
            split=args.split,
            trust_remote_code=args.trust_remote_code,
        )
    if args.column not in dataset.column_names:
        raise ValueError(f"Column '{args.column}' not found in dataset. Available: {dataset.column_names}")
    dataset = dataset.cast_column(args.column, Image())
    if args.shuffle and args.sample_size is not None:
        dataset = dataset.shuffle(seed=args.seed)
    return dataset


def summarise(dataset_id: str, subset: Optional[str], split: str, stats: SpectrumStats) -> dict:
    return {
        "dataset": dataset_id,
        "subset": subset,
        "split": split,
        "mean_slope": stats.mean,
        "std_slope": stats.std,
        "valid_images": stats.valid_count,
        "total_images": stats.total_count,
    }


def main() -> None:
    args = parse_args()
    dataset = load_split(args)
    stats = compute_dataset_slopes(
        dataset,
        column=args.column,
        sample_size=args.sample_size,
        target_size=args.target_size,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        progress=not args.quiet,
    )
    summary = summarise(args.dataset, args.subset, args.split, stats)
    print(json.dumps(summary, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
