"""Model-free single-frame detection bound for the clean features of a burst dataset.

    python tools/detection_bound.py [--dataset data/MIIC-burst-p10-drift] [--sources 5,21,...]
                                    [--peak 10] [--draws 400] [--doses 1,2,4,8,16] [--out bound.json]

For every structured feature the fine-feature diagnostic finds in the clean
images (``edge_denoise.finefeat.find_features``: connected components of the
2-12 px band above 4x the grain's robust sigma), this computes the
Neyman-Pearson test -- the Poisson log-likelihood ratio between "the feature
is there" and "it is not" -- with the feature's exact shape and position
GIVEN to the detector, on a single frame of the stated dose.  No network, no
prior, no estimator can detect the feature more reliably than this test, so
its detection rate at a fixed false-alarm rate is the ceiling any
single-frame method is bounded by.  Rows are grouped by the diagnostic's
single-frame SNR (contrast x sqrt(area) / local Poisson sigma), and the same
test is repeated with the rate multiplied by each entry of ``--doses`` (the
frames-equivalent dose).

The absent hypothesis replaces the feature's pixels by the clean image with
its 2-12 px band removed; both hypotheses are drawn ``--draws`` times as
Poisson counts on the feature's bounding box (plus a 4 px margin), and the
threshold is set on the absent draws.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from burst_diffusion.data import resolve_burst_dir
from edge_denoise.finefeat import _dilate, feature_band, find_features, region_labels

DEFAULT_SOURCES = "5,21,28,48,50,71,80,83,87,90"
BINS = ((0.0, 0.4), (0.4, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, float("inf")))


def feature_rows(dataset: Path, sources: list[int], *, peak: float, draws: int, doses: list[float], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    burst_dir = resolve_burst_dir(dataset)
    rows: list[dict] = []
    for src in sources:
        clean = np.asarray(Image.open(burst_dir / "clean" / f"{src:05d}.png")).astype(np.float64) / 255.0
        regions = region_labels(clean)
        features, labels, _, _ = find_features(clean, regions, peak=peak)
        band = feature_band(clean, regions)
        height, width = clean.shape
        for feature in features:
            member = labels == feature.label
            ys, xs = np.nonzero(member)
            y0, y1 = max(ys.min() - 4, 0), min(ys.max() + 5, height)
            x0, x1 = max(xs.min() - 4, 0), min(xs.max() + 5, width)
            win = np.s_[y0:y1, x0:x1]
            present = clean[win] * peak
            absent = present.copy()
            support = _dilate(member, 1)[win]
            absent[support] = (clean[win] - band[win])[support] * peak
            present = np.clip(present, 1e-3, None)
            absent = np.clip(absent, 1e-3, None)
            row = {"source": src, "label": feature.label, "area": feature.area, "contrast": abs(feature.contrast), "snr_single": feature.snr_single, "detection": {}}
            for dose in doses:
                lam1, lam0 = present * dose, absent * dose
                weight = np.log(lam1 / lam0)
                offset = float((lam1 - lam0).sum())
                k1 = rng.poisson(lam1, size=(draws,) + lam1.shape)
                k0 = rng.poisson(lam0, size=(draws,) + lam0.shape)
                t1 = (k1 * weight).sum(axis=(1, 2)) - offset
                t0 = (k0 * weight).sum(axis=(1, 2)) - offset
                row["detection"][str(dose)] = {
                    "at_5pct_false_alarm": float((t1 > np.percentile(t0, 95.0)).mean()),
                    "at_1pct_false_alarm": float((t1 > np.percentile(t0, 99.0)).mean()),
                    "at_50pct_false_alarm": float((t1 > np.percentile(t0, 50.0)).mean()),
                }
            rows.append(row)
    return rows


def summarize(rows: list[dict], doses: list[float]) -> str:
    lines = [f"{len(rows)} features; ideal single-frame detector with the feature's shape and position known (Neyman-Pearson).", ""]
    lines.append("| single-frame SNR bin | features | median area | median contrast | " + " | ".join(f"dose x{d:g}: detected at 5 % / 1 % false alarm" for d in doses) + " |")
    lines.append("|---|---|---|---|" + "---|" * len(doses))
    for lo, hi in BINS:
        sel = [r for r in rows if lo <= r["snr_single"] < hi]
        if not sel:
            continue
        cells = []
        for d in doses:
            a = np.median([r["detection"][str(d)]["at_5pct_false_alarm"] for r in sel])
            b = np.median([r["detection"][str(d)]["at_1pct_false_alarm"] for r in sel])
            cells.append(f"{100 * a:.0f} % / {100 * b:.0f} %")
        label = f"{lo:.1f} – {hi:.1f}" if hi != float("inf") else f"≥ {lo:.1f}"
        lines.append(f"| {label} | {len(sel)} | {np.median([r['area'] for r in sel]):.0f} px | {np.median([r['contrast'] for r in sel]):.3f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="data/MIIC-burst-p10-drift")
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--peak", type=float, default=10.0)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--doses", default="1,2,4,8,16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    doses = [float(v) for v in args.doses.split(",")]
    rows = feature_rows(Path(args.dataset), [int(v) for v in args.sources.split(",")], peak=args.peak, draws=args.draws, doses=doses, seed=args.seed)
    print(summarize(rows, doses))
    if args.out:
        Path(args.out).write_text(json.dumps({"dataset": args.dataset, "peak": args.peak, "draws": args.draws, "doses": doses, "features": rows}, indent=1), encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
