from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from fans.data import ImageDataset
from fans.fans_noise import FANSNoiseShaper
from fans.scheduler import DiffusionSchedule
from fans.unet import UNet
from fans.utils_fft import compute_dataset_band_profile, ensure_band_state, radial_frequency_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a diffusion model with optional FANS noise")
    parser.add_argument("--data", type=str, required=True, help="Dataset specifier or path")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to store checkpoints")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--use-fans", action="store_true", help="Enable FANS noise shaping")
    parser.add_argument("--bands", type=int, default=5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--spectral-samples", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-mults", type=str, default="1,2,4")
    return parser.parse_args()


def prepare_fans(
    dataset: ImageDataset,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FANSNoiseShaper, list[torch.Tensor], torch.Tensor]:
    stats = compute_dataset_band_profile(
        dataset.pil_images,
        num_bands=args.bands,
        image_size=args.image_size,
        sample_size=min(args.spectral_samples, len(dataset.pil_images)) if args.spectral_samples else None,
    )
    masks = radial_frequency_masks(args.image_size, args.image_size, args.bands, device=torch.device("cpu"))
    fans = FANSNoiseShaper(masks, stats.g_b, beta=args.beta, gamma=args.gamma)
    return fans, masks, stats.g_b


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = ImageDataset(data=args.data, image_size=args.image_size)
    num_workers = max(0, min(2, (os.cpu_count() or 2) - 1))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    channel_mults = tuple(int(part) for part in args.channel_mults.split(","))
    model = UNet(base_channels=args.base_channels, channel_mults=channel_mults).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    schedule = DiffusionSchedule.create(args.steps)
    schedule = schedule.to(device)

    fans: Optional[FANSNoiseShaper] = None
    mask_list: Optional[list[torch.Tensor]] = None
    g_b: Optional[torch.Tensor] = None
    if args.use_fans:
        fans, mask_list, g_b = prepare_fans(dataset, args=args, device=device)

    global_step = 0
    for epoch in range(args.epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress:
            batch = batch.to(device)
            t = torch.randint(0, args.steps, (batch.size(0),), device=device)
            alpha = schedule.sqrt_alphas_cumprod[t]
            sigma = schedule.sqrt_one_minus_alphas_cumprod[t]
            t01 = t.float() / max(args.steps - 1, 1)

            if fans is not None:
                eps = fans.fans_noise(batch, sigma.view(-1, 1, 1, 1), t01.view(-1, 1, 1, 1))
            else:
                eps = torch.randn_like(batch)

            x_t = alpha.view(-1, 1, 1, 1) * batch + sigma.view(-1, 1, 1, 1) * eps
            pred = model(x_t, t)
            loss = nn.functional.mse_loss(pred, eps)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            progress.set_postfix(loss=float(loss.item()))

    torch.save(model.state_dict(), outdir / "model.pt")
    torch.save(schedule, outdir / "schedule.pt")

    if args.use_fans and mask_list is not None and g_b is not None:
        torch.save({"bands": torch.stack(mask_list), "g_b": g_b, "beta": args.beta, "gamma": args.gamma}, outdir / "fans.pt")

    with open(outdir / "training_config.json", "w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, indent=2)


if __name__ == "__main__":
    main()

