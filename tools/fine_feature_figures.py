"""Figures for edge_denoise/docs/fine_feature_report.md (PIL only, TrueType text).

    python tools/fine_feature_figures.py charts  <fine_features.json> <out.png> --arms a,b,c
    python tools/fine_feature_figures.py plate   <out.png> --domain image|band [--crops src:top:left,...]
    python tools/fine_feature_figures.py summary <out.png>

`charts`: per-band transfer gain against the Wiener bounds + blemish retention
vs single-frame SNR.  `plate`: a stain gallery -- one row per crop with a clear
stain, as pictures (`--domain image`) or 2-12 px band views (`--domain band`),
each tile captioned with PSNR and the fraction of the stain's contrast kept.  `summary`: the
headline metrics of the key arms as labelled horizontal bars (the report's
one-look conclusion).  Every text element uses a TrueType font at >= 22 px so
the figures stay legible when scaled to a page column.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]
FONT_CANDIDATES = (
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)
BOLD_CANDIDATES = (
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)
INK, INK2, GRID, BG = (28, 37, 48), (71, 86, 100), (222, 226, 230), (255, 255, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (BOLD_CANDIDATES if bold else FONT_CANDIDATES):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=fnt)
    return right - left


# ---------------------------------------------------------------------------
# charts


def band_gain_chart(summary: dict, arms: list[str], *, width: int = 2000, height: int = 900) -> Image.Image:
    f_tick, f_label, f_title = font(24), font(26), font(30, bold=True)
    bands = summary["bands"]
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    left, right, top, bottom = 110, width - 470, 70, height - 120
    plot_w, plot_h = right - left, bottom - top
    ymin, ymax = -0.05, 1.08

    def y_of(v: float) -> float:
        return bottom - (max(ymin, min(ymax, v)) - ymin) / (ymax - ymin) * plot_h

    d.text((left, 16), "Transfer gain per band in flat regions  (1 = structure transmitted, 0 = erased)", fill=INK, font=f_title)
    d.rectangle([left, top, right, bottom], outline=INK, width=2)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_of(tick)
        d.line([left, y, right, y], fill=GRID, width=2)
        d.text((left - 80, y - 14), f"{tick:.2f}", fill=INK, font=f_tick)
    group_w = plot_w / len(bands)
    bar_w = group_w * 0.82 / (len(arms) + 1)
    for b, label in enumerate(bands):
        x0 = left + b * group_w
        d.text((x0 + group_w / 2 - text_width(d, label, f_label) / 2, bottom + 14), label, fill=INK, font=f_label)
        w1, w16 = summary["wiener_gain_single"][b], summary["wiener_gain_avg16"][b]
        if w1 is not None:
            d.line([x0 + 6, y_of(w1), x0 + group_w - 6, y_of(w1)], fill=INK, width=4)
        if w16 is not None:
            d.line([x0 + 6, y_of(w16), x0 + group_w - 6, y_of(w16)], fill=(130, 130, 130), width=2)
        for a, arm in enumerate(arms):
            gain = summary["methods"][arm]["band_gain"][b]
            if gain is None:
                continue
            x = x0 + group_w * 0.09 + a * bar_w
            d.rectangle([x, y_of(max(gain, 0.0)), x + bar_w - 4, y_of(min(gain, 0.0))], fill=PALETTE[a % len(PALETTE)])
    lx, ly = right + 24, top
    for a, arm in enumerate(arms):
        d.rectangle([lx, ly + 6, lx + 26, ly + 30], fill=PALETTE[a % len(PALETTE)])
        d.text((lx + 38, ly), arm, fill=INK, font=f_label)
        ly += 40
    d.line([lx, ly + 18, lx + 26, ly + 18], fill=INK, width=4)
    d.text((lx + 38, ly), "Wiener bound, 1 frame", fill=INK, font=f_label)
    ly += 40
    d.line([lx, ly + 18, lx + 26, ly + 18], fill=(130, 130, 130), width=2)
    d.text((lx + 38, ly), "Wiener bound, 16 frames", fill=INK, font=f_label)
    d.text((left, bottom + 56), "band (difference of Gaussians, sigma in px)", fill=INK2, font=f_label)
    return img


def retention_chart(results: dict, arms: list[str], *, panel: int = 500) -> Image.Image:
    f_tick, f_title, f_axis, f_head = font(22), font(26, bold=True), font(22), font(28, bold=True)
    cols = min(4, len(arms))
    rows = (len(arms) + cols - 1) // cols
    header = 70
    img = Image.new("RGB", (cols * panel + 40, header + rows * (panel + 30) + 40), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 16), "Blemish contrast retained (output / clean) vs single-frame SNR = contrast x sqrt(area) / Poisson sigma; one dot per clean feature",
           fill=INK, font=f_head)
    snr_max = 2.5
    for index, arm in enumerate(arms):
        px = 20 + (index % cols) * panel
        py = header + 20 + (index // cols) * (panel + 30)
        left, right, top, bottom = px + 80, px + panel - 24, py + 44, py + panel - 60
        d.rectangle([left, top, right, bottom], outline=INK, width=2)
        d.text((px + 80, py + 8), arm, fill=INK, font=f_title)

        def x_of(s: float) -> float:
            return left + min(s, snr_max) / snr_max * (right - left)

        def y_of(r: float) -> float:
            return bottom - (max(-0.25, min(1.25, r)) + 0.25) / 1.5 * (bottom - top)

        for tick in (0.0, 0.5, 1.0):
            d.line([left, y_of(tick), right, y_of(tick)], fill=GRID, width=2)
            d.text((px + 20, y_of(tick) - 14), f"{tick:.1f}", fill=INK, font=f_tick)
        for tick in (0.5, 1.0, 1.5, 2.0):
            d.line([x_of(tick), top, x_of(tick), bottom], fill=GRID, width=2)
            d.text((x_of(tick) - 14, bottom + 8), f"{tick:g}", fill=INK, font=f_tick)
        d.text((left, bottom + 34), "single-frame SNR", fill=INK2, font=f_axis)
        color = PALETTE[index % len(PALETTE)]
        for record in results["per_source"]:
            entry = record["methods"].get(arm)
            if entry is None:
                continue
            for feature, ret in zip(record["features"], entry["feature_retention"]):
                x, y = x_of(feature["snr_single"]), y_of(ret["retention_mean"])
                d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=color)
    return img


def charts(args: argparse.Namespace) -> None:
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    arms = args.arms.split(",") if args.arms else list(results["summary"]["methods"])
    top = band_gain_chart(results["summary"], arms)
    bottom = retention_chart(results, arms)
    canvas = Image.new("RGB", (max(top.width, bottom.width), top.height + bottom.height), BG)
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, top.height))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print("wrote", args.out, canvas.size)


# ---------------------------------------------------------------------------
# plates


def _arm_callables(which: str) -> dict:
    """label -> denoise callable ([B,1,S,S] in [-1,1] -> same, CPU)."""
    from burst_diffusion.sample import Sampler
    from edge_denoise.infer import Denoiser
    from edge_denoise.prior import PosteriorSampler

    R = Path("runs/edge_denoise")
    edge = lambda name: Denoiser.from_checkpoint(R / name / "ckpt_latest.pt").denoise  # noqa: E731
    if which == "baseline":
        n2n = Sampler.from_checkpoint(R / "ft_ladder/teacher_n2n.pt")
        return {
            "N2N teacher": lambda frames: n2n.run(frames, schedule=[1]).prediction,
            "control ft_noisy_b16": edge("miic_p10_dedup_ft_noisy_b16"),
            "ft_consist": edge("miic_p10_dedup_ft_consist"),
        }
    arms = {
        "control ft_noisy_b16": edge("miic_p10_dedup_ft_noisy_b16"),
        "A burst-mean target": edge("miic_p10_dedup_ft_avgfull_b16"),
        "A-debias (recommended)": edge("miic_p10_dedup_ft_avgdebias_b16"),
        "C A + consistency": edge("miic_p10_dedup_ft_avgfull_consist"),
        "B clean-target oracle": edge("miic_p10_dedup_ft_cleanfull_b16"),
        "D sdedit16 (diffusion)": PosteriorSampler.from_checkpoint(
            R / "miic_p10_dedup_prior/ckpt_latest.pt", peak=10.0, mode="sdedit", num_steps=25, eta=1.0, noise_variance=0.16
        ).denoise,
        "E defect-augmented": edge("miic_p10_dedup_ft_avgfull_aug_b16"),
    }
    return arms


DEFAULT_CROPS = "48:235:260,48:405:126,50:196:274,87:309:281,28:366:363,21:134:116"
IMAGE_ARMS = ("clean reference", "average of 16 frames", "control ft_noisy_b16", "A-debias (recommended)", "C A + consistency", "D sdedit16 (diffusion)")
BAND_ARMS = ("clean reference", "average of 16 frames", "control ft_noisy_b16", "A-debias (recommended)", "B clean-target oracle", "D sdedit16 (diffusion)")


def _stain_mask(band: np.ndarray, regions: np.ndarray, win) -> np.ndarray:
    """Largest-evidence connected stain (dark or bright) of the clean 2-12 px band inside the crop."""
    from edge_denoise.finefeat import label_components

    from edge_denoise.finefeat import _erode

    sub, inside = band[win], _erode(regions[win] > 0, 3)  # keep off the region borders
    best, best_score = None, 0.0
    for sign in (-1.0, 1.0):
        labels, count = label_components(inside & (sign * sub > 0.02))
        for k in range(1, count + 1):
            member = labels == k
            area = int(member.sum())
            if area < 20:
                continue
            ys, xs = np.nonzero(member)
            extent = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
            compact = area / float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
            if extent > 40 or compact < 0.3:  # a stain is a compact blob, not a strip along an edge
                continue
            score = abs(float(sub[member].mean())) * np.sqrt(area)
            if score > best_score:
                best, best_score = member, score
    return best if best is not None else np.zeros_like(sub, dtype=bool)


def plate(args: argparse.Namespace) -> None:
    """Stain gallery: one row per crop, ``--domain image`` (the pictures) or
    ``--domain band`` (2-12 px band, contrast x8, stain outlined); each tile's
    second caption line reports PSNR and the fraction of the stain's clean
    contrast the estimator keeps."""
    from burst_diffusion.data import BurstCache
    from burst_diffusion.metrics import psnr
    from edge_denoise.distill import denoise_full_frame
    from edge_denoise.finefeat import feature_band, region_labels

    cache = BurstCache("data/MIIC-burst-p10-dedup", channels=1, min_replicas=16, min_size=64,
                       val_fraction=0.1, test_fraction=0.1, split_seed=2019)
    by = {s.source_index: s for s in cache.val_sources}
    arms = _arm_callables("final")
    wanted = IMAGE_ARMS if args.domain == "image" else BAND_ARMS
    crop, up = args.crop, args.up
    f_cap, f_sub, f_head = font(24), font(20), font(26, bold=True)
    side = crop * up
    pad, cap_h, head_h = 10, 66, 50
    cols = len(wanted)
    blocks: list[Image.Image] = []
    cache_out: dict[tuple[int, str], np.ndarray] = {}
    for spec in args.crops.split(","):
        idx, top, left = (int(v) for v in spec.split(":"))
        s = by[idx]
        clean = s.clean.astype(np.float64) / 255.0
        frames = [f.astype(np.float64) / 255.0 for f in s.frames]
        regions = region_labels(clean)
        clean_band = feature_band(clean, regions)
        top = int(np.clip(top, 0, clean.shape[0] - crop))
        left = int(np.clip(left, 0, clean.shape[1] - crop))
        win = np.s_[top:top + crop, left:left + crop]
        stain = _stain_mask(clean_band, regions, win)
        outputs = {"clean reference": clean, "one noisy frame": frames[0], "average of 16 frames": np.mean(frames[:16], axis=0)}
        for name, fn in arms.items():
            key = (idx, name)
            if key not in cache_out:
                cache_out[key] = denoise_full_frame(fn, frames[0], tile=64, stride=48, tile_batch=32)
            outputs[name] = cache_out[key]
        clean_contrast = float(clean_band[win][stain].mean()) if stain.any() else float("nan")
        outline = stain & ~(np.roll(stain, 1, 0) & np.roll(stain, -1, 0) & np.roll(stain, 1, 1) & np.roll(stain, -1, 1))
        tiles = []
        for name in wanted:
            img = outputs[name]
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
        width = cols * (side + pad) + pad
        block = Image.new("RGB", (width, head_h + side + cap_h + pad), (24, 24, 24))
        d = ImageDraw.Draw(block)
        d.text((pad, 10), f"scene {idx}, {crop}x{crop} px crop at rows {top}-{top + crop}, cols {left}-{left + crop}", fill=(235, 235, 235), font=f_head)
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
    canvas = Image.new("RGB", (max(b.width for b in blocks), sum(b.height for b in blocks)), (24, 24, 24))
    y = 0
    for b in blocks:
        canvas.paste(b, (0, y))
        y += b.height
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print("wrote", args.out, canvas.size)


# ---------------------------------------------------------------------------
# summary (the one-look conclusion)


def _method(results: dict, name: str) -> dict:
    return results["methods"][name]


def summary(args: argparse.Namespace) -> None:
    R = Path("runs/edge_denoise")
    final = json.loads((R / "repeatability_val_ff_final/repeatability.json").read_text(encoding="utf-8"))
    post = json.loads((R / "repeatability_val_ff_posterior/repeatability.json").read_text(encoding="utf-8"))
    ff_phase1 = json.loads((R / "fine_features_phase1/fine_features.json").read_text(encoding="utf-8"))["summary"]["methods"]
    ff_final = json.loads((R / "fine_features_final/fine_features.json").read_text(encoding="utf-8"))["summary"]["methods"]
    ff_post = json.loads((R / "fine_features_posterior/fine_features.json").read_text(encoding="utf-8"))["summary"]["methods"]
    ff_base = json.loads((R / "fine_features_baseline/fine_features.json").read_text(encoding="utf-8"))["summary"]["methods"]
    arms = [
        ("control  ft_noisy_b16", _method(final, "one_shot@ft_noisy_b16"), ff_phase1["ft_noisy_b16"], (127, 127, 127)),
        ("A  burst-mean target", _method(final, "one_shot@ft_avgfull_b16"), ff_phase1["ft_avgfull_b16"], (31, 119, 180)),
        ("A-debias  (recommended)", _method(final, "one_shot@ft_avgdebias_b16"), ff_final["ft_avgdebias_b16"], (20, 104, 143)),
        ("C  A + consistency", _method(final, "one_shot@ft_avgfull_consist"), ff_phase1["ft_avgfull_consist"], (44, 160, 44)),
        ("B  clean-target oracle", _method(final, "one_shot@ft_cleanfull_b16"), ff_phase1["ft_cleanfull_b16"], (162, 113, 28)),
        ("D  diffusion sdedit16", _method(post, "one_shot@sdedit16"), ff_post["sdedit16"], (148, 103, 189)),
        ("ft_consist  (previous best)", _method(final, "one_shot@ft_consist"), ff_base["ft_consist"], (180, 180, 180)),
    ]
    control_psnr = arms[0][1]["accuracy"]["psnr_mean"]
    panels = [
        ("PSNR gain vs control (dB)  - higher is better", lambda m, f: m["accuracy"]["psnr_mean"] - control_psnr, "{:+.2f}", (-0.6, 0.8)),
        ("CD 3-sigma per scene, median (px)  - lower is better", lambda m, f: m["cd"]["scene_median_3sigma_px"], "{:.3f}", (0.0, 0.7)),
        ("signed CD bias (px)  - closer to 0 is better", lambda m, f: m["cd"]["bias_mean_px"], "{:+.3f}", (-0.02, 0.16)),
        ("blemish contrast retained, median  - higher is better", lambda m, f: f["feature_retention_median"], "{:.2f}", (0.0, 0.8)),
    ]
    f_title, f_label, f_val, f_head = font(28, bold=True), font(24), font(24, bold=True), font(34, bold=True)
    panel_w, row_h, label_w = 1000, 46, 420
    n = len(arms)
    panel_h = 70 + n * row_h + 30
    width, height = 2 * panel_w + 60, 110 + 2 * panel_h + 40
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.text((30, 24), "Headline metrics on the dev split (10 scenes x 10 retakes): recommended recipe vs control, oracle and alternatives",
           fill=INK, font=f_head)
    for p, (title, getter, fmt, (lo, hi)) in enumerate(panels):
        px = 30 + (p % 2) * (panel_w + 30)
        py = 110 + (p // 2) * (panel_h + 20)
        d.text((px, py), title, fill=INK, font=f_title)
        bar_left = px + label_w
        bar_right = px + panel_w - 120
        scale = (bar_right - bar_left) / (hi - lo)
        zero_x = bar_left + (0.0 - lo) * scale if lo < 0 < hi else bar_left
        d.line([zero_x, py + 60, zero_x, py + 60 + n * row_h], fill=INK, width=2)
        for i, (name, m, f, color) in enumerate(arms):
            y = py + 66 + i * row_h
            value = getter(m, f)
            d.text((px, y + 8), name, fill=INK, font=f_label)
            x_val = bar_left + (max(lo, min(hi, value)) - lo) * scale
            x0, x1 = (zero_x, x_val) if x_val >= zero_x else (x_val, zero_x)
            d.rectangle([x0, y + 6, x1, y + row_h - 10], fill=color)
            d.text((max(x0, x1) + 10, y + 8), fmt.format(value), fill=INK, font=f_val)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out, optimize=True)
    print("wrote", args.out, img.size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("charts")
    c.add_argument("results")
    c.add_argument("out")
    c.add_argument("--arms", default=None)
    c.set_defaults(func=charts)
    p = sub.add_parser("plate")
    p.add_argument("out")
    p.add_argument("--domain", default="image", choices=("image", "band"))
    p.add_argument("--crops", default=DEFAULT_CROPS, help="src:top:left, comma-separated")
    p.add_argument("--crop", type=int, default=96)
    p.add_argument("--up", type=int, default=4)
    p.add_argument("--outline", action="store_true", help="outline the stain on the clean tile of the image plate")
    p.set_defaults(func=plate)
    s = sub.add_parser("summary")
    s.add_argument("out")
    s.set_defaults(func=summary)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
