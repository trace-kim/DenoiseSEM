"""Render the figures embedded in edge_denoise/docs/edge_denoise_report.md.

Reproduces, from the pilot checkpoints, the three report figures:

1. ``pilot_outputs.png`` -- real denoised outputs per arm next to the noisy
   input, the 16-frame average, and clean, PSNR-captioned.
2. ``pilot_sigma_maps.png`` -- per-pixel repeatability sigma across the same
   10 seed frames the pilot used, per arm; tiles are normalized per-tile to
   their own p99 so the SPATIAL PATTERN (edge concentration) is readable, with
   absolute values in the captions.
3. ``pilot_gradient_domain.png`` -- what the pure-gradient arm sees and
   predicts, and the reconstructed image next to the sobloss output.

Pure PIL composition (repo convention: no matplotlib). Usage:

    python tools/edge_denoise_report_figures.py `
      --config edge_denoise/configs/miic_p10_dedup_hybrid.yml `
      --n2n-checkpoint <burst n2n ckpt> `
      --hybrid-checkpoint <...> --grad-checkpoint <...> --sobloss-checkpoint <...> `
      --out edge_denoise/docs/images
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from burst_diffusion.data import BurstCache
from burst_diffusion.metrics import psnr
from burst_diffusion.repeatability import _c4
from burst_diffusion.sample import Sampler

from edge_denoise.config import load_config
from edge_denoise.gradient import sobel
from edge_denoise.infer import Denoiser

CAPTION_HEIGHT = 14
PAD = 4
UPSCALE = 3
NUM_SEEDS = 10


def _to_model(frames01: np.ndarray) -> torch.Tensor:
    """[K, H, W] float01 -> [K, 1, H, W] float32 in [-1, 1]."""
    return torch.from_numpy(frames01.astype(np.float32) * 2.0 - 1.0).unsqueeze(1)


def _to01(tensor: torch.Tensor) -> np.ndarray:
    """[K, 1, H, W] in [-1, 1] -> [K, H, W] float64 in [0, 1]."""
    return ((tensor.clamp(-1.0, 1.0) + 1.0) / 2.0).squeeze(1).cpu().numpy().astype(np.float64)


def _tile(array01: np.ndarray) -> Image.Image:
    scaled = np.rint(np.clip(array01, 0.0, 1.0) * 255.0).astype(np.uint8)
    scaled = scaled.repeat(UPSCALE, axis=0).repeat(UPSCALE, axis=1)
    return Image.fromarray(scaled).convert("RGB")


def _grid(rows: list[list[tuple[str, np.ndarray]]], out_path: Path) -> None:
    font = ImageFont.load_default()
    tile_side = rows[0][0][1].shape[0] * UPSCALE
    columns = max(len(row) for row in rows)
    cell_w, cell_h = tile_side + PAD, tile_side + CAPTION_HEIGHT + PAD
    canvas = Image.new("RGB", (columns * cell_w + PAD, len(rows) * cell_h + PAD), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        for column_index, (caption, array01) in enumerate(row):
            x = PAD + column_index * cell_w
            y = PAD + row_index * cell_h
            canvas.paste(_tile(array01), (x, y))
            draw.text((x, y + tile_side + 1), caption, fill=(230, 230, 230), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=True)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n2n-checkpoint", required=True)
    parser.add_argument("--hybrid-checkpoint", required=True)
    parser.add_argument("--grad-checkpoint", required=True)
    parser.add_argument("--sobloss-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sources", default="5,21,87,90", help="val source indices, comma-separated")
    parser.add_argument("--gradient-source", type=int, default=90)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    size = config.data.image_size
    cache = BurstCache(
        config.data.dataset_dir,
        channels=config.data.channels,
        min_replicas=NUM_SEEDS,
        min_size=size,
        val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction,
        split_seed=config.data.split_seed,
    )
    by_index = {source.source_index: source for source in cache.val_sources}
    wanted = [int(part) for part in args.sources.split(",")]
    missing = [index for index in wanted if index not in by_index]
    if missing:
        raise SystemExit(f"not in the val split: {missing} (val = {sorted(by_index)})")

    n2n = Sampler.from_checkpoint(args.n2n_checkpoint, device=args.device)
    arms = {
        "sobloss": Denoiser.from_checkpoint(args.sobloss_checkpoint, device=args.device),
        "hybrid": Denoiser.from_checkpoint(args.hybrid_checkpoint, device=args.device),
        "grad": Denoiser.from_checkpoint(args.grad_checkpoint, device=args.device),
    }
    out_dir = Path(args.out)

    def center01(source, count: int) -> tuple[np.ndarray, np.ndarray]:
        height, width = source.clean.shape[:2]
        top, left = (height - size) // 2, (width - size) // 2
        window = np.s_[top : top + size, left : left + size]
        clean01 = source.clean[window].astype(np.float64) / 255.0
        frames01 = np.stack([f[window].astype(np.float64) / 255.0 for f in source.frames[:count]])
        return clean01, frames01

    def run_arms(frames01: np.ndarray) -> dict[str, np.ndarray]:
        """[K, H, W] noisy -> per-arm [K, H, W] denoised outputs."""
        batch = _to_model(frames01)
        outputs = {"n2n": _to01(n2n.run(batch, schedule=[n2n.num_steps]).prediction)}
        for name, denoiser in arms.items():
            outputs[name] = _to01(denoiser.denoise(batch))
        return outputs

    # ---- Figure 1: outputs -------------------------------------------------
    rows = []
    for index in wanted:
        clean01, frames01 = center01(by_index[index], 16)
        outputs = run_arms(frames01[:1])
        avg16 = frames01.mean(axis=0)

        def cap(label: str, image01: np.ndarray) -> tuple[str, np.ndarray]:
            return f"{label} {psnr(clean01, image01):.1f}dB", image01

        rows.append(
            [
                (f"src {index}: noisy {psnr(clean01, frames01[0]):.1f}dB", frames01[0]),
                cap("avg16", avg16),
                cap("n2n", outputs["n2n"][0]),
                cap("sobloss", outputs["sobloss"][0]),
                cap("hybrid", outputs["hybrid"][0]),
                cap("grad", outputs["grad"][0]),
                ("clean", clean01),
            ]
        )
    _grid(rows, out_dir / "pilot_outputs.png")

    # ---- Figure 2: repeatability sigma maps --------------------------------
    rows = []
    for index in wanted:
        clean01, frames01 = center01(by_index[index], NUM_SEEDS)
        outputs = run_arms(frames01)
        outputs = {"single frame": frames01, **outputs}
        row = [(f"src {index}: clean", clean01)]
        for name, stack in outputs.items():
            sigma = stack.std(axis=0, ddof=1) / _c4(NUM_SEEDS)
            display = sigma / max(np.percentile(sigma, 99.0), 1e-9)
            row.append(
                (f"{name} m{sigma.mean() * 1e3:.1f} p95 {np.percentile(sigma, 95) * 1e3:.1f}", display)
            )
        rows.append(row)
    _grid(rows, out_dir / "pilot_sigma_maps.png")

    # ---- Figure 3: the gradient domain -------------------------------------
    source = by_index[args.gradient_source]
    clean01, frames01 = center01(source, 1)
    noisy = _to_model(frames01[:1])
    grad_denoiser = arms["grad"]
    with torch.no_grad():
        predicted_field = grad_denoiser.model(noisy.to(grad_denoiser.device)).cpu()
    reconstructed = _to01(grad_denoiser.denoise(noisy))[0]
    sobloss_out = _to01(arms["sobloss"].denoise(noisy))[0]

    def magnitude(image01: np.ndarray) -> np.ndarray:
        field = sobel(_to_model(image01[None]))
        return field.norm(dim=1).squeeze(0).numpy().astype(np.float64)

    mag_noisy = magnitude(frames01[0])
    mag_pred = predicted_field.norm(dim=1).squeeze(0).numpy().astype(np.float64)
    mag_clean = magnitude(clean01)
    scale = max(np.percentile(mag_clean, 99.0), 1e-9)

    def gcap(label: str, mag: np.ndarray) -> tuple[str, np.ndarray]:
        return f"{label} rms {np.sqrt((mag ** 2).mean()):.3f}", np.clip(mag / scale, 0.0, 1.0)

    rows = [
        [
            (f"src {args.gradient_source}: noisy y", frames01[0]),
            (f"grad arm x_hat {psnr(clean01, reconstructed):.1f}dB", reconstructed),
            (f"sobloss x_hat {psnr(clean01, sobloss_out):.1f}dB", sobloss_out),
            ("clean x", clean01),
        ],
        [
            gcap("|S y| (input)", mag_noisy),
            gcap("|S x| predicted", mag_pred),
            gcap("|S x| clean", mag_clean),
            gcap("|S x_hat| sobloss", magnitude(sobloss_out)),
        ],
    ]
    _grid(rows, out_dir / "pilot_gradient_domain.png")


if __name__ == "__main__":
    main()
