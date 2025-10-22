from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.serialization
from PIL import Image
from tqdm import tqdm

from fans.fans_noise import build_fans_from_state
from fans.scheduler import DiffusionSchedule
from fans.sampling import sample_loop
from fans.unet import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a trained diffusion model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save generated images")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sampler-steps", type=int, default=None, help="Override number of diffusion steps")
    parser.add_argument("--use-fans", action="store_true")
    parser.add_argument("--fans", type=str, default=None, help="Path to FANS state")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_tensor_images(tensor: torch.Tensor, outdir: Path, offset: int) -> None:
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.cpu()
    for idx, img in enumerate(tensor):
        arr = (img.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        Image.fromarray(arr).save(outdir / f"{offset + idx:06d}.png")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    config_path = Path(args.ckpt).parent / "training_config.json"
    base_channels = 64
    channel_mults = (1, 2, 4)
    if config_path.exists():
        config = json.loads(config_path.read_text())
        base_channels = int(config.get("base_channels", base_channels))
        channel_mults = tuple(int(x) for x in config.get("channel_mults", "1,2,4").split(","))

    model = UNet(base_channels=base_channels, channel_mults=channel_mults)
    state_dict = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    schedule_path = Path(args.ckpt).parent / "schedule.pt"
    if schedule_path.exists():
        torch.serialization.add_safe_globals([DiffusionSchedule])
        schedule: DiffusionSchedule = torch.load(schedule_path, map_location=device, weights_only=False)
        schedule = schedule.to(device)
    else:
        steps = args.sampler_steps or 1000
        schedule = DiffusionSchedule.create(steps).to(device)

    fans = None
    if args.use_fans:
        fans_path = args.fans or (Path(args.ckpt).parent / "fans.pt")
        state = torch.load(fans_path, map_location="cpu")
        fans = build_fans_from_state(
            bands=[band for band in state["bands"]],
            g_b=state["g_b"],
            beta=state.get("beta", 1.0),
            gamma=state.get("gamma", 1.0),
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    total = args.num
    batch_size = args.batch
    batches = math.ceil(total / batch_size)
    counter = 0
    for _ in tqdm(range(batches), desc="Sampling"):
        current = min(batch_size, total - counter)
        samples = sample_loop(
            model,
            schedule,
            (current, 3, args.image_size, args.image_size),
            fans=fans,
            device=device,
            generator=generator,
        )
        save_tensor_images(samples, outdir, counter)
        counter += current


if __name__ == "__main__":
    main()

