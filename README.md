# Frequency-Aware Adaptive Noise Scheduling Tools

This repository collects utilities that support Frequency-Aware Adaptive Noise Scheduling (FANS) experiments. The focus is on measuring how different image datasets deviate from the natural image power-law spectrum.

## Installation

```bash
pip install -r requirements.txt
```

## Radial power spectrum CLI

`scripts/compute_radial_power_slope.py` downloads an image dataset from the Hugging Face Hub and estimates the slope of its radial power spectrum in log–log space.

```bash
python -m scripts.compute_radial_power_slope \
  <dataset_id> \
  --split train \
  --sample-size 100 \
  --output results/<name>.json
```

Key options:

- `--subset`: configuration name for multi-configuration datasets.
- `--sample-size`: number of images to evaluate (defaults to the whole split).
- `--target-size`: resize images before the FFT (default `256`).
- `--min-freq` / `--max-freq`: normalised frequency range used for fitting the slope.
- `--shuffle`: randomly shuffle before sampling.

The tool saves a JSON summary with the mean slope, standard deviation, and sample counts. Cached datasets are reused across runs, so repeated invocations are fast once the images are downloaded.

## Example dataset slopes

| Domain | Dataset (split) | Samples | Mean slope | Std. dev. | Result file |
| --- | --- | --- | --- | --- | --- |
| Texture | `dream-textures/textures-color-1k` (train) | 100 | −1.56 | 0.75 | `results/textures_color_1k.json` |
| Medical X-ray | `mmenendezg/pneumonia_x_ray` (train) | 100 | −2.87 | 0.13 | `results/pneumonia_x_ray.json` |
| Remote sensing | `vicm0r/eurosat` (train) | 100 | −4.17 | 0.30 | `results/eurosat.json` |
| Astronomy | `matthieulel/galaxy10_decals` (train) | 100 | −1.94 | 0.46 | `results/galaxy10_decals.json` |

These measurements quantify how strongly each domain deviates from the natural-image prior (slope ≈ −2). Domains with flatter spectra (e.g., medical and astronomical imaging) are prime candidates for larger FANS gains, while steep spectra such as EuroSAT indicate a closer match to the natural-image assumption.
