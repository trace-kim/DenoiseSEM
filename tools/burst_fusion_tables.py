"""Markdown tables for edge_denoise/docs/burst_fusion_report.md from the eval JSONs.

    python tools/burst_fusion_tables.py <repeatability.json> <fine_features.json> [--arm fuse] [--controls ft_noisy_b16,ft_avgdebias_b16]

Prints (1) the dose ladder -- one row per frame count K for the drifting
average, the registered average and the fusion network -- with retention,
CD 3-sigma, bias, pixel sigma and PSNR, and (2) the single-frame reference
rows, all read from the two JSON files so the report never hand-copies a
number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COUNTS = (1, 2, 4, 8, 16)


def _fmt(value, spec: str, scale: float = 1.0) -> str:
    return "–" if value is None else format(value * scale, spec)


def _row(name: str, label: str, rep: dict, ff: dict, ff_name: str | None = None) -> str:
    m = rep["methods"].get(name)
    f = ff["summary"]["methods"].get(ff_name or name)
    if m is None:
        return f"| {label} | – | – | – | – | – | – | – |"
    pixel = m["pixel_repeatability"]
    return "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
        label,
        _fmt(None if f is None else f.get("feature_retention_median"), ".2f"),
        _fmt(None if f is None else f.get("feature_retention_snr_ge_1"), ".2f"),
        _fmt(m["cd"]["scene_median_3sigma_px"], ".3f"),
        _fmt(m["cd"]["bias_mean_px"], "+.3f"),
        _fmt(m["cd"]["bias_abs_mean_px"], ".3f"),
        _fmt(None if pixel is None else pixel["sigma_mean"], ".2f", 1e3),
        _fmt(m["accuracy"]["psnr_mean"], ".2f"),
    )


HEADER = (
    "| method | retention median | retention SNR ≥ 1 | CD 3σ scene px | CD bias px | \\|bias\\| px | pixel σ ×10⁻³ | PSNR dB |\n"
    "|---|---|---|---|---|---|---|---|"
)


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the tables carry Greek letters and superscripts
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repeatability")
    parser.add_argument("fine_features")
    parser.add_argument("--arm", default="fuse")
    parser.add_argument("--controls", default="ft_noisy_b16,ft_avgdebias_b16")
    args = parser.parse_args()
    rep = json.loads(Path(args.repeatability).read_text(encoding="utf-8"))
    ff = json.loads(Path(args.fine_features).read_text(encoding="utf-8"))
    # Per-SNR-bin retention is a list aligned with summary["snr_bins"]; the
    # last bin is [1, inf) -- the features one frame already detects.
    bins = ff["summary"].get("snr_bins") or []
    for record in ff["summary"]["methods"].values():
        by_snr = record.get("retention_by_snr") or []
        for (low, _), item in zip(bins, by_snr):
            if low >= 1.0 and item.get("count"):
                record["feature_retention_snr_ge_1"] = item.get("retention_median")
    print(f"Dose ladder ({rep['count']} scenes, {rep['num_seeds']} retakes, seed stride {rep.get('seed_stride', 1)})\n")
    print(HEADER)
    for count in COUNTS:
        avg = "single_frame" if count == 1 else f"avg_of_{count}"
        print(_row(avg, f"drifting average, K = {count}", rep, ff))
    for count in COUNTS:
        print(_row(f"regavg{count}@{args.arm}", f"registered average, K = {count}", rep, ff))
    for count in COUNTS:
        print(_row(f"fuse{count}@{args.arm}", f"**burst fusion, K = {count}**", rep, ff))
    print("\nSingle-frame reference arms (first frame of each retake)\n")
    print(HEADER)
    for control in args.controls.split(","):
        print(_row(f"one_shot@{control}", control, rep, ff, ff_name=control))
    print("\nFine-feature detail (band gains 4-8 / 8-16 / 16-32 px, retention median / mean, retake std, false features per 1000 px)\n")
    print("| method | gain 4–8 | gain 8–16 | gain 16–32 | retention median / mean | retake std | false /1000 px |\n|---|---|---|---|---|---|---|")
    order = [("single_frame", "drifting average, K = 1")] + [(f"avg_of_{c}", f"drifting average, K = {c}") for c in (4, 16)]
    order += [(f"regavg{c}@{args.arm}", f"registered average, K = {c}") for c in COUNTS]
    order += [(f"fuse{c}@{args.arm}", f"**burst fusion, K = {c}**") for c in COUNTS]
    order += [(c, c) for c in args.controls.split(",")]
    labels = ff["summary"]["bands"]
    for name, label in order:
        f = ff["summary"]["methods"].get(name)
        if f is None:
            continue
        gains = dict(zip(labels, f["band_gain"]))
        print(
            "| {} | {} | {} | {} | {} / {} | {} | {} |".format(
                label,
                _fmt(gains.get("4-8px"), ".3f"),
                _fmt(gains.get("8-16px"), ".3f"),
                _fmt(gains.get("16-32px"), ".3f"),
                _fmt(f.get("feature_retention_median"), ".3f"),
                _fmt(f.get("feature_retention_mean"), ".3f"),
                _fmt(f.get("feature_retention_std_mean"), ".3f"),
                _fmt(f.get("false_features_per_1000_flat_px"), ".2f"),
            )
        )


if __name__ == "__main__":
    main()
