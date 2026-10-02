"""Slide-ready drift and brightness trend figures from stability reports.

Each CASE is one tools/real_sem_stability.py output folder (one N2N training
correction). Its stability-raw/ and stability-png/ reports hold frames.csv for
the raw input and the model output; the input must be the same for every case.

    python tools/stability_trend_figures.py --out output/stability_figures \
        --case "Affine + brightness=output/stability_affine_percentile" \
        --case "No correction=output/stability_none_none"

Writes PNG (300 dpi), SVG and PDF for: drift_x, drift_y, brightness (one panel
per case) and case_<n> (x drift, y drift and brightness of one case), plus the
plotted numbers in trends.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_COLOR, OUTPUT_COLOR = "#8c8b87", "#2a78d6"
INK, MUTED = "#1a1a19", "#52514e"
QUANTITIES = {"drift_x": ("dx_px", "x drift (px)"), "drift_y": ("dy_px", "y drift (px)"),
              "brightness": ("raw_mean_dn", "Mean intensity (DN)")}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
})


def read_frames(report: Path) -> list[dict]:
    files = sorted(report.glob("**/frames.csv"))
    if len(files) != 1:
        raise ValueError(f"Expected one site frames.csv under {report}, found {len(files)}")
    with files[0].open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["included"] == "True"]

    def number(value: str) -> float:
        return float(value) if value not in ("", None) else math.nan

    return [{"frame": int(row["frame_index"]), "pixel_sha256": row["pixel_sha256"],
             **{key: number(row.get(key, "")) for key, _ in QUANTITIES.values()}} for row in rows]


def rms_gap(a: list[float], b: list[float]) -> float:
    pairs = [(x - y) ** 2 for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    return math.sqrt(sum(pairs) / len(pairs)) if pairs else math.nan


def draw(ax, frames: list[int], raw: list[float], output: list[float], unit: str) -> None:
    ax.plot(frames, raw, color=INPUT_COLOR, lw=2.2, label="Raw input", solid_capstyle="round")
    ax.plot(frames, output, color=OUTPUT_COLOR, lw=1.2, label="N2N output", solid_capstyle="round")
    ax.text(0, 1.02, f"RMS gap {rms_gap(raw, output):.2f} {unit}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.5, color=MUTED)
    ax.margins(x=0.01)


def save(fig, out: Path, name: str) -> None:
    for suffix, extra in ((".png", {"dpi": 300}), (".svg", {}), (".pdf", {})):
        fig.savefig(out / f"{name}{suffix}", bbox_inches="tight", facecolor="white", **extra)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", action="append", required=True, metavar="LABEL=DIR",
                        help="Legend label and real_sem_stability.py output folder; repeat in display order")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width-in", type=float, default=7.0, help="Width of the small-multiple figures (inches)")
    parser.add_argument("--case-width-in", type=float, default=3.6, help="Width of the per-case figures (inches)")
    args = parser.parse_args()

    cases = []
    for value in args.case:
        label, _, folder = value.partition("=")
        if not label or not folder:
            raise SystemExit("Use --case LABEL=DIR")
        folder = Path(folder)
        raw, output = read_frames(folder / "stability-raw"), read_frames(folder / "stability-png")
        if [r["frame"] for r in raw] != [r["frame"] for r in output]:
            raise SystemExit(f"{label}: raw and output reports include different frames")
        cases.append((label, raw, output))
    reference = [r["pixel_sha256"] for r in cases[0][1]]
    for label, raw, _ in cases[1:]:
        if [r["pixel_sha256"] for r in raw] != reference:
            raise SystemExit(f"{label}: raw input differs from {cases[0][0]}; all cases must use the same site")
    args.out.mkdir(parents=True, exist_ok=True)

    columns = min(3, len(cases))
    rows = math.ceil(len(cases) / columns)
    for name, (key, ylabel) in QUANTITIES.items():
        fig, axes = plt.subplots(rows, columns, figsize=(args.width_in, 1.75 * rows + 0.6),
                                 sharex=True, sharey=True, squeeze=False, constrained_layout=True)
        unit = "DN" if name == "brightness" else "px"
        for ax, (label, raw, output) in zip(axes.flat, cases):
            frames = [r["frame"] for r in raw]
            draw(ax, frames, [r[key] for r in raw], [r[key] for r in output], unit)
            ax.set_title(label, color=INK, loc="left", pad=13)
        for ax in axes.flat[len(cases):]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        for ax in axes[-1, :]:
            ax.set_xlabel("Acquisition")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper right", ncols=2, fontsize=8)
        save(fig, args.out, name)

    for number, (label, raw, output) in enumerate(cases, 1):
        fig, axes = plt.subplots(3, 1, figsize=(args.case_width_in, 5.2), sharex=True, constrained_layout=True)
        frames = [r["frame"] for r in raw]
        for ax, (name, (key, ylabel)) in zip(axes, QUANTITIES.items()):
            draw(ax, frames, [r[key] for r in raw], [r[key] for r in output], "DN" if name == "brightness" else "px")
            ax.set_ylabel(ylabel)
        axes[0].set_title(label, color=INK, loc="left", pad=13)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper right", ncols=2, fontsize=7.5)
        axes[-1].set_xlabel("Acquisition")
        save(fig, args.out, f"case_{number}")

    with (args.out / "trends.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_number", "case", "series", "frame", *QUANTITIES])
        for number, (label, raw, output) in enumerate(cases, 1):
            for series, rows_ in (("raw_input", raw), ("n2n_output", output)):
                for r in rows_:
                    writer.writerow([number, label, series, r["frame"], *(r[k] for k, _ in QUANTITIES.values())])
    print(f"Figures: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
