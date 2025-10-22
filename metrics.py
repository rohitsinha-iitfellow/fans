from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from torch_fidelity import calculate_metrics
import lpips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID and LPIPS between folders")
    parser.add_argument("--real", type=str, required=True, help="Path to folder containing real images")
    parser.add_argument("--gen", type=str, required=True, help="Path to folder with generated images")
    parser.add_argument("--out", type=str, default="metrics.json")
    parser.add_argument("--lpips-samples", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path) -> None:
        self.paths = sorted(
            [p for p in path.glob("*.png")] + [p for p in path.glob("*.jpg")] + [p for p in path.glob("*.jpeg")]
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {path}")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        from PIL import Image

        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def compute_lpips(real_dir: Path, gen_dir: Path, device: torch.device, max_samples: int) -> float:
    real_ds = ImageFolderDataset(real_dir)
    gen_ds = ImageFolderDataset(gen_dir)
    count = min(len(real_ds), len(gen_ds), max_samples)
    indices = torch.linspace(0, count - 1, steps=count).long()

    lpips_model = lpips.LPIPS(net="alex").to(device)
    scores = []
    for idx in tqdm(indices.tolist(), desc="LPIPS"):
        x = real_ds[idx].unsqueeze(0).to(device)
        y = gen_ds[idx].unsqueeze(0).to(device)
        score = lpips_model(x, y)
        scores.append(float(score.item()))
    return sum(scores) / len(scores)


def main() -> None:
    args = parse_args()
    real_dir = Path(args.real)
    gen_dir = Path(args.gen)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    metrics = calculate_metrics(input1=str(real_dir), input2=str(gen_dir), fid=True, verbose=False)
    lpips_score = compute_lpips(real_dir, gen_dir, device, args.lpips_samples)

    result = {"fid": float(metrics["frechet_inception_distance"]), "lpips": float(lpips_score)}
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)


if __name__ == "__main__":
    main()

