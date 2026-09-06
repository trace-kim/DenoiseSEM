"""Figures for edge_denoise/docs/burst_fusion_report.md (PIL only, TrueType text).

    python tools/burst_fusion_figures.py ladder <repeatability.json> <fine_features.json> <out.png> [--arm fuse] [--control ft_noisy_b16]
    python tools/burst_fusion_figures.py plate  <out.png> --domain image|band --fusion-checkpoint <pt> [--control-checkpoint <pt>] [--predenoise <pt>] [--crops ...]
    python tools/burst_fusion_figures.py registration <out.png> --table LABEL=<accuracy.json> [--table ...]
    python tools/burst_fusion_figures.py summary <repeatability.json> <fine_features.json> <out.png>

`ladder`: the dose ladder -- blemish retention, CD 3-sigma, PSNR and pixel
sigma against the number of frames K for the drifting average, the
registered average and the fusion network, with the single-frame control as a
reference line.  `plate`: the stain gallery on drifting bursts (clean,
drifting 16-frame average, registered average, single-frame control, fusion
at two frame counts), as pictures or 2-12 px band views, each tile captioned
with PSNR and the fraction of the stain's contrast kept.  `registration`:
per-source registration error of each method against the recorded drift.
`summary`: the two headline panels (retention and CD 3-sigma vs K).
Every text element uses a TrueType font at >= 22 px.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fine_feature_figures import BG, GRID, INK, INK2, PALETTE, _stain_mask, font, text_width  # noqa: E402

FRAME_COUNTS = (1, 2, 4, 8, 16)
SERIES = (
    ("drifting average", "avg", (127, 127, 127)),
    ("registered average", "regavg", (255, 127, 14)),
    ("burst fusion (this work)", "fuse", (31, 119, 180)),
)


# ---------------------------------------------------------------------------
# generic line-chart panel


def _panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    ylabel: str,
    series: list[tuple[str, tuple[int, int, int], list[tuple[float, float]], str]],
    reference: tuple[str, float] | None,
    *,
    log_x: bool = True,
    log_y: bool = False,
    y_range: tuple[float, float] | None = None,
    fmt: str = "{:.2f}",
) -> None:
    f_tick, f_label, f_title = font(22), font(24), font(28, bold=True)
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 44), title, fill=INK, font=f_title)
    values = [v for _, _, pts, _ in series for _, v in pts if v is not None and not math.isnan(v)]
    if reference is not None:
        values.append(reference[1])
    if not values:
        return
    tf = (lambda v: math.log10(max(v, 1e-9))) if log_y else (lambda v: v)
    lo, hi = (min(values), max(values)) if y_range is None else y_range
    lo, hi = tf(lo), tf(hi)
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.08 * (hi - lo)
    lo, hi = (lo - pad, hi + pad) if y_range is None else (lo, hi)
    xs_all = [x for _, _, pts, _ in series for x, _ in pts]
    xmin, xmax = min(xs_all), max(xs_all)

    def px(x: float) -> float:
        if log_x:
            return x0 + (math.log2(x) - math.log2(xmin)) / (math.log2(xmax) - math.log2(xmin)) * (x1 - x0)
        return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)

    def py(v: float) -> float:
        return y1 - (tf(v) - lo) / (hi - lo) * (y1 - y0)

    # grid + ticks
    if log_y:
        ticks = [10.0**e for e in range(int(math.floor(lo)), int(math.ceil(hi)) + 1) if lo <= e <= hi]
    else:
        ticks = [lo + (hi - lo) * k / 4 for k in range(5)]
    for v in ticks:
        y = py(v)
        draw.line([(x0, y), (x1, y)], fill=GRID, width=1)
        label = fmt.format(v)
        draw.text((x0 - 12 - text_width(draw, label, f_tick), y - 12), label, fill=INK2, font=f_tick)
    for x in sorted(set(xs_all)):
        xp = px(x)
        draw.line([(xp, y0), (xp, y1)], fill=GRID, width=1)
        label = str(int(x))
        draw.text((xp - text_width(draw, label, f_tick) / 2, y1 + 8), label, fill=INK2, font=f_tick)
    draw.rectangle([x0, y0, x1, y1], outline=INK2, width=2)
    draw.text((x0 + (x1 - x0) / 2 - text_width(draw, "frames fused (K)", f_label) / 2, y1 + 38), "frames fused (K)", fill=INK2, font=f_label)
    if reference is not None:
        y = py(reference[1])
        for xx in range(int(x0), int(x1), 18):
            draw.line([(xx, y), (min(xx + 9, x1), y)], fill=(214, 39, 40), width=3)
        draw.text((x0 + 8, y - 30), reference[0], fill=(214, 39, 40), font=f_tick)
    for _, color, pts, style in series:
        pts = [(x, v) for x, v in pts if v is not None and not math.isnan(v)]
        for (xa, va), (xb, vb) in zip(pts, pts[1:]):
            draw.line([(px(xa), py(va)), (px(xb), py(vb))], fill=color, width=5)
        for x, v in pts:
            r = 9
            draw.ellipse([px(x) - r, py(v) - r, px(x) + r, py(v) + r], fill=color if style == "solid" else BG, outline=color, width=3)


def _legend(
    draw: ImageDraw.ImageDraw, x: int, y: int, entries: list[tuple[str, tuple[int, int, int]]], *, max_x: int | None = None
) -> None:
    """Horizontal legend; wraps to a new row when an entry would pass ``max_x``."""
    f = font(24)
    x_start = x
    for label, color in entries:
        entry_width = 58 + text_width(draw, label, f) + 48
        if max_x is not None and x > x_start and x + entry_width > max_x:
            x = x_start
            y += 40
        draw.line([(x, y + 14), (x + 46, y + 14)], fill=color, width=6)
        draw.ellipse([x + 14, y + 5, x + 32, y + 23], fill=color)
        draw.text((x + 58, y), label, fill=INK, font=f)
        x += entry_width


# ---------------------------------------------------------------------------
# data access


def _rep_method(rep: dict, name: str) -> dict | None:
    return rep["methods"].get(name)


def _series_from_results(rep: dict, ff: dict | None, arm: str, key: str, counts=FRAME_COUNTS):
    """``{series_key: [(K, value), ...]}`` for a repeatability metric ``key``
    (dotted path into a method record) or a fine-feature metric when ``ff``
    is given and ``key`` starts with ``ff:``."""
    out: dict[str, list[tuple[float, float]]] = {}
    for _, skey, _ in SERIES:
        pts = []
        for count in counts:
            if skey == "avg":
                name = "single_frame" if count == 1 else f"avg_of_{count}"
            else:
                name = f"{skey}{count}@{arm}"
            if key.startswith("ff:"):
                methods = ff["summary"]["methods"] if ff is not None else {}
                record = methods.get(name)
                value = None if record is None else record.get(key[3:])
            else:
                record = _rep_method(rep, name)
                value = record
                if record is not None:
                    for part in key.split("."):
                        value = value.get(part) if isinstance(value, dict) else None
                        if value is None:
                            break
            pts.append((float(count), None if value is None else float(value)))
        out[skey] = pts
    return out


def _control_value(rep: dict, ff: dict | None, control: str, key: str) -> float | None:
    name = f"one_shot@{control}"
    if key.startswith("ff:"):
        # The fine-feature diagnostic names single-frame arms by their bare arm name.
        record = ff["summary"]["methods"].get(control) if ff is not None else None
        return None if record is None else record.get(key[3:])
    record = _rep_method(rep, name)
    if record is None:
        return None
    value = record
    for part in key.split("."):
        value = value.get(part) if isinstance(value, dict) else None
        if value is None:
            return None
    return float(value)


PANELS = (
    ("Blemish retention (median, clean = 1)", "retention", "ff:feature_retention_median", "{:.2f}", 1.0, False),
    ("CD repeatability, 3-sigma per scene (px, lower is better)", "CD 3-sigma px", "cd.scene_median_3sigma_px", "{:.2f}", None, False),
    ("PSNR vs clean (dB)", "dB", "accuracy.psnr_mean", "{:.1f}", None, False),
    ("Pixel repeatability sigma (x1e-3, log scale, lower is better)", "sigma x1e-3", "pixel_repeatability.sigma_mean", "{:g}", None, True),
)


def ladder(args: argparse.Namespace) -> None:
    rep = json.loads(Path(args.repeatability).read_text(encoding="utf-8"))
    ff = json.loads(Path(args.fine_features).read_text(encoding="utf-8"))
    panels = PANELS if not args.two else PANELS[:2]
    cols = 2
    rows = math.ceil(len(panels) / cols)
    pw, ph, margin_l, margin_t, gap_x, gap_y = 900, 520, 150, 120, 120, 150
    width = margin_l + cols * pw + (cols - 1) * gap_x + 60
    height = margin_t + rows * ph + (rows - 1) * gap_y + 140
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    f_title = font(34, bold=True)
    draw.text((margin_l, 24), args.title, fill=INK, font=f_title)
    for index, (title, ylabel, key, fmt, cap, log_y) in enumerate(panels):
        r, c = divmod(index, cols)
        x0 = margin_l + c * (pw + gap_x)
        y0 = margin_t + r * (ph + gap_y)
        series_data = _series_from_results(rep, ff, args.arm, key)
        scale = 1e3 if "pixel" in key else 1.0
        series = []
        for label, skey, color in SERIES:
            pts = [(x, None if v is None else v * scale) for x, v in series_data[skey]]
            series.append((label, color, pts, "solid"))
        reference = None
        control_value = _control_value(rep, ff, args.control, key)
        if control_value is not None:
            reference = (f"single-frame control ({args.control})", control_value * scale)
        y_range = None
        if cap is not None:
            values = [v for _, _, pts, _ in series for _, v in pts if v is not None] + ([control_value] if control_value else [])
            y_range = (max(0.0, min(values) - 0.05), max(cap + 0.02, max(values) + 0.02))
        _panel(draw, (x0, y0, x0 + pw, y0 + ph), title, ylabel, series, reference, fmt=fmt, y_range=y_range, log_y=log_y)
    _legend(draw, margin_l, height - 70, [(label, color) for label, _, color in SERIES] + [("single-frame control", (214, 39, 40))])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print("wrote", args.out, canvas.size)


def summary(args: argparse.Namespace) -> None:
    args.two = True
    args.title = args.title or "Fine features come back with the burst's frames; the drifting average never gets them"
    ladder(args)


# ---------------------------------------------------------------------------
# registration accuracy


def registration(args: argparse.Namespace) -> None:
    tables = []
    for spec in args.table:
        label, path = spec.split("=", 1)
        tables.append((label, json.loads(Path(path).read_text(encoding="utf-8"))))
    split = args.split
    sources = sorted({item["source_index"] for _, t in tables for item in t[split]["per_burst"]})
    if args.sources:
        wanted = {int(v) for v in args.sources.split(",")}
        sources = [s for s in sources if s in wanted]
    f_tick, f_label, f_title = font(22), font(24), font(30, bold=True)
    pw, ph, margin_l, margin_t = 90 * len(sources) + 200, 520, 140, 110
    width, height = max(margin_l + pw + 80, 1500), margin_t + ph + 190 + 40 * (len(tables) // 3)
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_l, 24), "Registration error per source: RMS position error of the worse axis (px, log scale)", fill=INK, font=f_title)
    x0, y0, x1, y1 = margin_l, margin_t, margin_l + pw, margin_t + ph
    lo, hi = -2.0, 1.2  # log10 px
    def py(v: float) -> float:
        return y1 - (math.log10(max(v, 10 ** lo)) - lo) / (hi - lo) * (y1 - y0)
    for e in range(-2, 2):
        y = py(10.0**e)
        draw.line([(x0, y), (x1, y)], fill=GRID, width=1)
        label = f"{10.0**e:g}"
        draw.text((x0 - 14 - text_width(draw, label, f_tick), y - 12), label, fill=INK2, font=f_tick)
    for v in (0.25,):
        y = py(v)
        for xx in range(int(x0), int(x1), 18):
            draw.line([(xx, y), (min(xx + 9, x1), y)], fill=(214, 39, 40), width=2)
        draw.text((x1 - 260, y - 30), "0.25 px (a quarter pixel)", fill=(214, 39, 40), font=f_tick)
    draw.rectangle([x0, y0, x1, y1], outline=INK2, width=2)
    group = 90
    bar = max(8, int((group - 16) / len(tables)))
    for si, source in enumerate(sources):
        gx = x0 + 20 + si * group
        for ti, (label, table) in enumerate(tables):
            errors = np.concatenate([np.asarray(i["position_errors"]) for i in table[split]["per_burst"] if i["source_index"] == source])
            rms = float(np.sqrt((errors**2).mean(axis=0)).max())
            color = PALETTE[ti % len(PALETTE)]
            bx = gx + ti * bar
            draw.rectangle([bx, py(rms), bx + bar - 3, y1], fill=color)
        lab = str(source)
        draw.text((gx + group / 2 - 6 - text_width(draw, lab, f_tick) / 2, y1 + 8), lab, fill=INK2, font=f_tick)
    draw.text((x0 + pw / 2 - 80, y1 + 40), "source index", fill=INK2, font=f_label)
    _legend(
        draw, margin_l, y1 + 90, [(label, PALETTE[i % len(PALETTE)]) for i, (label, _) in enumerate(tables)], max_x=width - 40
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print("wrote", args.out, canvas.size)


# ---------------------------------------------------------------------------
# stain gallery on drifting bursts


DEFAULT_CROPS = "48:235:260,48:405:126,50:196:274,87:309:281,28:366:363,21:134:116"


def plate(args: argparse.Namespace) -> None:
    from burst_diffusion.data import BurstCache
    from burst_diffusion.metrics import psnr
    from edge_denoise.distill import build_teacher, denoise_full_frame
    from edge_denoise.finefeat import feature_band, region_labels
    from edge_denoise.fusion import BurstFusionArms, FusionDenoiser, frames_per_retake_of

    per_retake = frames_per_retake_of(args.dataset, default=16)
    cache = BurstCache(args.dataset, channels=1, min_replicas=per_retake, min_size=64, val_fraction=0.1, test_fraction=0.1, split_seed=2019)
    by = {s.source_index: s for s in cache.val_sources}
    denoiser = FusionDenoiser.from_checkpoint(args.fusion_checkpoint)
    predenoise = None
    if args.predenoise:
        fn, _ = build_teacher(args.predenoise)
        predenoise = lambda frame01: denoise_full_frame(fn, frame01, tile=64, stride=48)  # noqa: E731
    counts = [int(v) for v in args.frames.split(",")]
    from edge_denoise.drift import load_drift_truth

    truth = load_drift_truth(args.dataset) if args.align == "truth" else None
    fusion = BurstFusionArms(denoiser, frame_counts=counts, frames_per_retake=per_retake, align=args.align, regavg=True, predenoise=predenoise, truth=truth)
    control = None
    if args.control_checkpoint:
        control, _ = build_teacher(args.control_checkpoint)
    columns = ["clean reference", f"average of {per_retake} drifting frames", f"registered average of {per_retake}"]
    if control is not None:
        columns.append(f"{args.control_label} (1 frame)")
    columns += [f"burst fusion, K = {count}" for count in counts]
    crop, up = args.crop, args.up
    f_cap, f_sub, f_head = font(24), font(20), font(26, bold=True)
    side = crop * up
    pad, cap_h, head_h = 10, 66, 50
    blocks: list[Image.Image] = []
    for spec in args.crops.split(","):
        idx, top, left = (int(v) for v in spec.split(":"))
        s = by[idx]
        clean = s.clean.astype(np.float64) / 255.0
        regions = region_labels(clean)
        clean_band = feature_band(clean, regions)
        top = int(np.clip(top, 0, clean.shape[0] - crop))
        left = int(np.clip(left, 0, clean.shape[1] - crop))
        win = np.s_[top:top + crop, left:left + crop]
        stain = _stain_mask(clean_band, regions, win)
        outputs = fusion.outputs(s, args.retake)
        burst = fusion.retake_frames(s, args.retake)
        images = {
            "clean reference": clean,
            columns[1]: np.mean(burst[:per_retake], axis=0),
            columns[2]: outputs[f"regavg{max(counts)}"] if max(counts) == per_retake else fusion.denoiser.registered_mean(burst, per_retake, fusion.trajectory(s, args.retake, burst)),
        }
        if control is not None:
            images[columns[3]] = denoise_full_frame(control, burst[0], tile=64, stride=48)
        for count in counts:
            images[f"burst fusion, K = {count}"] = outputs[f"fuse{count}"]
        clean_contrast = float(clean_band[win][stain].mean()) if stain.any() else float("nan")
        outline = stain & ~(np.roll(stain, 1, 0) & np.roll(stain, -1, 0) & np.roll(stain, 1, 1) & np.roll(stain, -1, 1))
        tiles = []
        for name in columns:
            img = images[name]
            band = feature_band(img, regions)
            kept = float(band[win][stain].mean()) / clean_contrast if stain.any() else float("nan")
            if name == "clean reference":
                sub = f"stain contrast {clean_contrast:+.3f}"
            elif args.domain == "image":
                sub = f"PSNR {psnr(clean[win][..., None], img[win][..., None]):.1f} dB - stain kept {kept:.0%}"
            else:
                sub = f"stain kept {kept:.0%}"
            if args.domain == "image":
                tiles.append(((name, sub), img[win], outline if name == "clean reference" and args.outline else None))
            else:
                tiles.append(((name, sub), np.clip(band[win] * 8 + 0.5, 0, 1), outline))
        width = len(columns) * (side + pad) + pad
        block = Image.new("RGB", (width, head_h + side + cap_h + pad), (24, 24, 24))
        d = ImageDraw.Draw(block)
        d.text((pad, 10), f"scene {idx}, retake {args.retake}, {crop}x{crop} px crop at rows {top}-{top + crop}, cols {left}-{left + crop}", fill=(235, 235, 235), font=f_head)
        for c, ((name, sub), arr, outl) in enumerate(tiles):
            rgb = np.stack([np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8)] * 3, axis=-1)
            if outl is not None:
                rgb[outl] = (255, 80, 40)
            tile = Image.fromarray(rgb.repeat(up, 0).repeat(up, 1))
            x, y = pad + c * (side + pad), head_h
            block.paste(tile, (x, y))
            d.text((x, y + side + 6), name, fill=(235, 235, 235), font=f_cap)
            d.text((x, y + side + 38), sub, fill=(170, 170, 170), font=f_sub)
        blocks.append(block)
        print("scene", idx, "done", flush=True)
    canvas = Image.new("RGB", (max(b.width for b in blocks), sum(b.height for b in blocks)), (24, 24, 24))
    y = 0
    for b in blocks:
        canvas.paste(b, (0, y))
        y += b.height
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print("wrote", args.out, canvas.size)


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ladder", "summary"):
        p = sub.add_parser(name)
        p.add_argument("repeatability")
        p.add_argument("fine_features")
        p.add_argument("out")
        p.add_argument("--arm", default="fuse")
        p.add_argument("--control", default="ft_noisy_b16")
        p.add_argument("--title", default="" if name == "summary" else "Dose ladder on drifting bursts: what K frames buy, by estimator")
        p.add_argument("--two", action="store_true", help="only the two headline panels")
        p.set_defaults(func=ladder if name == "ladder" else summary)
    p = sub.add_parser("registration")
    p.add_argument("out")
    p.add_argument("--table", action="append", required=True, help="LABEL=<accuracy.json> (repeatable)")
    p.add_argument("--split", default="holdout")
    p.add_argument("--sources", default=None, help="comma-separated source indices to show (default: all)")
    p.set_defaults(func=registration)
    p = sub.add_parser("plate")
    p.add_argument("out")
    p.add_argument("--dataset", default="data/MIIC-burst-p10-drift")
    p.add_argument("--fusion-checkpoint", required=True)
    p.add_argument("--control-checkpoint", default=None)
    p.add_argument("--control-label", default="single-frame control")
    p.add_argument("--predenoise", default=None)
    p.add_argument("--align", default="registered")
    p.add_argument("--frames", default="4,16")
    p.add_argument("--retake", type=int, default=0)
    p.add_argument("--domain", default="image", choices=("image", "band"))
    p.add_argument("--crops", default=DEFAULT_CROPS)
    p.add_argument("--crop", type=int, default=96)
    p.add_argument("--up", type=int, default=4)
    p.add_argument("--outline", action="store_true")
    p.set_defaults(func=plate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
