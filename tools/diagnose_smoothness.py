"""Quantify the smoothness caveat in edge_denoise/docs/edge_denoise_report.md.

Every learned arm outputs images visibly smoother than the clean reference it
is scored against. This tool measures why, per source:

- the clean flats' fine grain (sigma_m) vs the per-frame Poisson noise
  (sigma_n) and the resulting per-pixel Wiener recovery gain;
- how much high-frequency texture each arm actually emits in flat regions;
- the correlation between each arm's flat-region residual and the clean
  image's own grain (near -1 means the "error" is the grain the arm did not
  reproduce);
- where the squared error lives (flat vs edge masks);
- the grain's spatial structure on the full frame: row/column streak share,
  lag-1 autocorrelation, and intensity scaling (recoverability depends on it).

Train sources are included to show the signature is estimator character, not
a generalization gap. Pure PIL composition (repo convention: no matplotlib).
Usage:

    python tools/diagnose_smoothness.py `
      --config edge_denoise/configs/miic_p10_dedup_sobloss.yml `
      --n2n-checkpoint <burst n2n ckpt> `
      --sobloss-checkpoint <...> --hybrid-checkpoint <...> --grad-checkpoint <...> `
      --figure runs/edge_denoise/smoothness_diagnosis.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from burst_diffusion.data import BurstCache
from burst_diffusion.metrics import _gaussian_window
from burst_diffusion.sample import Sampler

from edge_denoise.config import load_config
from edge_denoise.infer import Denoiser

MARGIN = 8          # interior kept per tile for statistics (kills conv border effects)
BLUR_SIGMA = 1.5    # grain = clean - gauss(clean, BLUR_SIGMA)
FLAT_T = 0.006      # |grad| of smoothed clean below this -> flat
EDGE_T = 0.030      # above this -> edge
POISSON_PEAK = 10.0

_KERNEL = _gaussian_window(11, BLUR_SIGMA).to(torch.float32)


def _gauss(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(x.astype(np.float32))[None, None]
    return F.conv2d(t, _KERNEL, padding=5)[0, 0].numpy().astype(np.float64)


def _erode(mask: np.ndarray, k: int = 5) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return (-F.max_pool2d(-t, k, stride=1, padding=k // 2))[0, 0].numpy() > 0.5


def _dilate(mask: np.ndarray, k: int = 5) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.max_pool2d(t, k, stride=1, padding=k // 2)[0, 0].numpy() > 0.5


def _to_model(frames01: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames01.astype(np.float32) * 2.0 - 1.0).unsqueeze(1)


def _to01(t: torch.Tensor) -> np.ndarray:
    return ((t.clamp(-1, 1) + 1) / 2).squeeze(1).cpu().numpy().astype(np.float64)


def _tiles_of(img: np.ndarray, tile: int) -> np.ndarray:
    rows, cols = img.shape[0] // tile, img.shape[1] // tile
    return np.stack([img[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
                     for r in range(rows) for c in range(cols)])


def _run_batched(fn, tiles01: np.ndarray, chunk: int = 16) -> np.ndarray:
    return np.concatenate([_to01(fn(_to_model(tiles01[s:s + chunk])))
                           for s in range(0, len(tiles01), chunk)])


def _grain_structure(clean: np.ndarray) -> str:
    """Streak share, lag-1 autocorrelation, and intensity scaling of the grain."""
    margin = 16
    smooth = _gauss(clean)
    grain = (clean - smooth)[margin:-margin, margin:-margin]
    inten = smooth[margin:-margin, margin:-margin]
    gy, gx = np.gradient(inten)
    flat = _erode(np.hypot(gx, gy) < FLAT_T)
    g = np.where(flat, grain, np.nan)
    var_total = float(np.nanvar(g))

    def axis_share(a: np.ndarray) -> float:
        means, counts = [], []
        for line in a:
            vals = line[~np.isnan(line)]
            if len(vals) >= 80:
                means.append(vals.mean())
                counts.append(len(vals))
        if not means:
            return float("nan")
        raw = float(np.mean(np.square(means)))
        return max(raw - var_total * float(np.mean(1.0 / np.array(counts))), 0.0) / var_total

    def lag1(a: np.ndarray, axis: int) -> float:
        s1 = a[:, :-1] if axis == 1 else a[:-1, :]
        s2 = a[:, 1:] if axis == 1 else a[1:, :]
        ok = ~np.isnan(s1) & ~np.isnan(s2)
        return float(np.corrcoef(s1[ok], s2[ok])[0, 1])

    dark = flat & (inten < 0.32)
    bright = flat & (inten > 0.42)
    scaling = ""
    if dark.sum() > 500 and bright.sum() > 500:
        sd = float(np.sqrt(np.mean(grain[dark] ** 2)))
        sb = float(np.sqrt(np.mean(grain[bright] ** 2)))
        predicted = float(np.sqrt(inten[bright].mean() / inten[dark].mean()))
        scaling = f", bright/dark sigma ratio {sb / sd:.2f} (sqrt(I) predicts {predicted:.2f})"
    return (f"grain structure: row-streak {axis_share(g):.1%}, col {axis_share(g.T):.1%}, "
            f"lag1 h {lag1(g, 1):+.2f} v {lag1(g, 0):+.2f}{scaling}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n2n-checkpoint", required=True)
    parser.add_argument("--sobloss-checkpoint", required=True)
    parser.add_argument("--hybrid-checkpoint", required=True)
    parser.add_argument("--grad-checkpoint", required=True)
    parser.add_argument("--sources", default="5,21,87,90", help="val source indices, comma-separated")
    parser.add_argument("--train-sources", type=int, default=2,
                        help="how many train sources to include (memorization check)")
    parser.add_argument("--figure", default=None, help="optional path for the residual-vs-grain figure")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    tile = config.data.image_size
    cache = BurstCache(
        config.data.dataset_dir, channels=config.data.channels, min_replicas=10,
        min_size=tile, val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction, split_seed=config.data.split_seed,
    )
    by_index = {s.source_index: s for s in cache.val_sources}
    wanted = [int(part) for part in args.sources.split(",")]
    missing = [index for index in wanted if index not in by_index]
    if missing:
        raise SystemExit(f"not in the val split: {missing} (val = {sorted(by_index)})")

    n2n = Sampler.from_checkpoint(args.n2n_checkpoint, device=args.device)
    denoisers = {
        "sobloss": Denoiser.from_checkpoint(args.sobloss_checkpoint, device=args.device),
        "hybrid": Denoiser.from_checkpoint(args.hybrid_checkpoint, device=args.device),
        "grad": Denoiser.from_checkpoint(args.grad_checkpoint, device=args.device),
    }
    arm_fns = {
        "single": lambda b: b,
        "n2n": lambda b: n2n.run(b, schedule=[n2n.num_steps]).prediction,
        **{name: (lambda d: (lambda b: d.denoise(b)))(d) for name, d in denoisers.items()},
    }

    interior = np.s_[:, MARGIN:tile - MARGIN, MARGIN:tile - MARGIN]
    header = (f"{'arm':<8}{'PSNR':>7}{'RMS flat':>10}{'RMS edge':>10}"
              f"{'corr(res,grain)':>17}{'tex RMS':>9}{'SE% flat':>10}{'SE% edge':>10}")

    def analyze(tag: str, source) -> None:
        clean = source.clean.astype(np.float64) / 255.0
        frames = np.stack([f.astype(np.float64) / 255.0 for f in source.frames[:16]])
        clean_tiles = _tiles_of(clean, tile)
        outputs = {"avg16": _tiles_of(frames.mean(axis=0), tile)}
        frame0_tiles = _tiles_of(frames[0], tile)
        for name, fn in arm_fns.items():
            outputs[name] = _run_batched(fn, frame0_tiles)

        smooth = np.stack([_gauss(t) for t in clean_tiles])
        grain = clean_tiles - smooth
        gy, gx = np.gradient(smooth, axis=(1, 2))
        magnitude = np.hypot(gx, gy)
        flat = np.stack([_erode(m < FLAT_T) for m in magnitude])
        edge = np.stack([_dilate(m > EDGE_T) for m in magnitude])

        ci, gi, fi, ei = clean_tiles[interior], grain[interior], flat[interior], edge[interior]
        sigma_m = float(np.sqrt(np.mean(gi[fi] ** 2)))
        mean_i = float(ci[fi].mean())
        sigma_n = float(np.sqrt(mean_i / POISSON_PEAK))
        grain_only_db = -10 * np.log10(np.sum(gi[fi] ** 2) / gi.size)
        print(f"\n=== {tag} src {source.source_index} | flat px {fi.mean():.0%}, edge px {ei.mean():.0%} | "
              f"grain sigma_m={sigma_m:.4f} | poisson sigma_n={sigma_n:.3f} @ I={mean_i:.2f} | "
              f"wiener gain={sigma_m**2 / (sigma_m**2 + sigma_n**2):.4f} | "
              f"drop-grain-only ceiling={grain_only_db:.1f} dB")
        print("    " + _grain_structure(clean))
        print(header)
        for name, out in outputs.items():
            oi = out[interior]
            resid = oi - ci
            texture = (out - np.stack([_gauss(t) for t in out]))[interior]
            se_all = float(np.sum(resid ** 2))
            rf = resid[fi]
            print(f"{name:<8}{-10 * np.log10(se_all / resid.size):>7.2f}"
                  f"{np.sqrt(np.mean(rf ** 2)):>10.4f}{np.sqrt(np.mean(resid[ei] ** 2)):>10.4f}"
                  f"{float(np.corrcoef(rf, gi[fi])[0, 1]):>17.3f}"
                  f"{np.sqrt(np.mean(texture[fi] ** 2)):>9.4f}"
                  f"{np.sum(rf ** 2) / se_all:>10.1%}{np.sum(resid[ei] ** 2) / se_all:>10.1%}")

    for index in wanted:
        analyze("VAL", by_index[index])
    for source in list(cache.train_sources)[:args.train_sources]:
        analyze("TRAIN", source)

    if args.figure is None:
        return
    rows = []
    for index in wanted[:2]:
        source = by_index[index]
        clean = source.clean.astype(np.float64) / 255.0
        frame0 = source.frames[0].astype(np.float64) / 255.0
        top, left = (clean.shape[0] - tile) // 2, (clean.shape[1] - tile) // 2
        window = np.s_[top:top + tile, left:left + tile]
        c, y = clean[window], frame0[window]
        out = _to01(denoisers["sobloss"].denoise(_to_model(y[None])))[0]
        rows.append([
            (f"src {index}: clean", c), ("noisy", y), ("sobloss", out),
            ("resid x6 +.5", np.clip((out - c) * 6 + 0.5, 0, 1)),
            ("grain x6 +.5", np.clip((c - _gauss(c)) * 6 + 0.5, 0, 1)),
        ])
    font = ImageFont.load_default()
    up, pad, cap = 4, 4, 14
    side = tile * up
    canvas = Image.new("RGB", (5 * (side + pad) + pad, len(rows) * (side + cap + pad) + pad), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(rows):
        for cix, (label, arr) in enumerate(row):
            image = Image.fromarray(np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8)
                                    .repeat(up, 0).repeat(up, 1)).convert("RGB")
            x, y0 = pad + cix * (side + pad), pad + r * (side + cap + pad)
            canvas.paste(image, (x, y0))
            draw.text((x, y0 + side + 1), label, fill=(230, 230, 230), font=font)
    out_path = Path(args.figure)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=True)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
