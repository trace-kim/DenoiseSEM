"""CD / registration 3-sigma vs frame count: two side-by-side panels, one highlighted model.

Edit the settings block below and run:  python tools/plot_3sigma_scatter.py
Input: TSV with columns  Model, Frame #, CD 3sig, Regi 3sig.
Output: PNG (300 dpi) + SVG sized for a 12.7 x 12.7 cm slide box.

At each frame count POR is a short black bar and the models are markers next to it,
ordered by value (highest on the left), so "below the bar" reads as "better than POR".
HIGHLIGHT is drawn in the deck blue; the other models keep their own light colour.
Each model is named once, right beside one of its own markers, at the first spot
(latest frame count first) where the name overlaps no marker, bar or other name.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------- settings
TSV = Path("results.tsv")
OUT = Path("three_sigma_scatter")          # writes OUT.png and OUT.svg
SIZE_CM = (12.7, 12.7)                     # (width, height)
TRANSPARENT = False                        # True: no white box behind the plot on a coloured slide

PANELS = [                                 # (column, panel title)
    ("CD 3sig", "CD 3σ (nm)"),
    ("Regi 3sig", "Registration 3σ (nm)"),
]
X_LABEL = "Number of frames"

HIGHLIGHT = "Fusion"                       # the model the slide is about
HIGHLIGHT_COLOR = "#1428A0"                # deck blue
OTHER_COLORS = ["#F2A77F", "#7FCFB1", "#ADA3DD", "#EFA9C3", "#E3C062",   # one per model, in order;
                "#9DCB86", "#8EC3EA", "#C9A58A", "#EE9A9A", "#B5B4AE",   # 12 models before repeating
                "#D6A6E0"]
MARKERS = ["s", "^", "D", "v", "P", "X", "<", ">", "p", "h", "*"]
MODEL_ORDER = None                         # e.g. ["Raw avg", "N2N", "Burst diffusion", "Fusion"]

POR_MODEL = "POR"                          # rows with this model name are drawn as the black bars
POR_COLOR = "#1A1A1A"
POR_BAR = 0.76                             # bar length, fraction of one frame slot

BANDS = False                              # light block behind each frame count
SPREAD = 0.7                               # width the markers of one frame count spread over (fraction of slot)
HIGHLIGHT_SIZE, OTHER_SIZE = 6.5, 4.5      # marker sizes in points
LABEL_SIZE = 6.5                           # model names beside their markers
FONT = ["Arial", "DejaVu Sans"]            # matplotlib renders the variable Noto Sans KR as Thin
FONT_SIZE = 9
INK, SECONDARY, BASELINE, GRID, BAND = "#1A1A1A", "#4A4A4A", "#BDBCB6", "#E9E8E3", "#F4F3EF"
# --------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": FONT_SIZE,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8, "axes.labelcolor": SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.facecolor": "none", "axes.axisbelow": True,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 5,
    "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
    "svg.fonttype": "none", "axes.unicode_minus": False,
})

data = pd.read_csv(TSV, sep="\t")
data.columns = data.columns.str.strip()
models = [m for m in (MODEL_ORDER or dict.fromkeys(data["Model"])) if m != POR_MODEL]
frames = sorted(data["Frame #"].unique())
others = [m for m in models if m != HIGHLIGHT]
has_por = POR_MODEL in set(data["Model"])
last = len(frames) - 1


def style(model):
    if model == HIGHLIGHT:
        return dict(color=HIGHLIGHT_COLOR, marker="o", ms=HIGHLIGHT_SIZE, mew=1.0, zorder=5)
    k = others.index(model)
    return dict(color=OTHER_COLORS[k % len(OTHER_COLORS)], marker=MARKERS[k % len(MARKERS)],
                ms=OTHER_SIZE, mew=0.7, zorder=4)


def offset(model, column, frame):
    """Position inside the frame slot: models ordered by value, highest on the left."""
    present = [m for m in models if not math.isnan(value(m, column, frame))]
    ranked = sorted(present, key=lambda m: -value(m, column, frame))
    step = SPREAD / max(len(ranked) - 1, 1)
    return (ranked.index(model) - (len(ranked) - 1) / 2) * step if model in ranked else 0.0


def value(model, column, frame):
    hit = data[(data["Model"] == model) & (data["Frame #"] == frame)][column]
    return hit.iloc[0] if len(hit) else math.nan


def draw(ax, column, title):
    slots = range(len(frames))
    if BANDS:
        for i in slots:
            ax.axvspan(i - 0.46, i + 0.46, color=BAND, lw=0, zorder=0)
    if has_por:
        for i, f in enumerate(frames):
            y = value(POR_MODEL, column, f)
            ax.plot([i - POR_BAR / 2, i + POR_BAR / 2], [y, y], color=POR_COLOR, lw=2.0,
                    solid_capstyle="round", zorder=3)
    for model in models:
        ys = [value(model, column, f) for f in frames]
        xs = [i + offset(model, column, f) for i, f in enumerate(frames)]
        ax.plot(xs, ys, ls="none", mec="white", **style(model))
    ax.set_xticks(list(slots), [f"{f:g}" for f in frames])
    ax.set_xlim(-0.5, last + 0.5)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=1)
    ax.set_title(title, loc="left", color=INK, fontsize=FONT_SIZE + 1, fontweight="bold", pad=6)


def anchors(model, column):
    """Every drawn mark of a model as (slot, centre x, centre y, half width, half height) in pixels."""
    pt = plt.gcf().dpi / 72
    out = []
    for i, f in enumerate(frames):
        v = value(model, column, f)
        if math.isnan(v):
            continue
        if model == POR_MODEL:
            (x0, y), (x1, _) = AX.transData.transform([(i - POR_BAR / 2, v), (i + POR_BAR / 2, v)])
            out.append((i, (x0 + x1) / 2, y, (x1 - x0) / 2, 1.2 * pt))
        else:
            x, y = AX.transData.transform((i + offset(model, column, f), v))
            half = style(model)["ms"] / 2 * pt
            out.append((i, x, y, half, half))
    return out


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def names(ax, column):
    """Name each model once, beside one of its own markers, where the name hits nothing else."""
    global AX
    AX = ax
    renderer = ax.figure.canvas.get_renderer()
    pad = 1.2 * ax.figure.dpi / 72
    frame = ax.get_window_extent()
    marks = {m: anchors(m, column) for m in ([POR_MODEL] if has_por else []) + models}
    clear = 3.5 * pad                                              # empty space between a name and other models' marks
    taken = [(x - w - clear, y - h - clear, x + w + clear, y + h + clear)
             for a in marks.values() for _, x, y, w, h in a]
    for model in [m for m in (POR_MODEL, HIGHLIGHT) if m in marks] + [m for m in models if m != HIGHLIGHT]:
        bold = model in (HIGHLIGHT, POR_MODEL)
        text = ax.text(0, 0, model, fontsize=LABEL_SIZE, ha="left", va="baseline", zorder=6,
                       color=HIGHLIGHT_COLOR if model == HIGHLIGHT else INK if bold else SECONDARY,
                       fontweight="bold" if bold else "normal")
        box = text.get_window_extent(renderer)
        origin = AX.transData.transform((0, 0))
        dx0, dy0, tw, th = box.x0 - origin[0], box.y0 - origin[1], box.width, box.height
        best = None
        def covered(a):                                            # a marker partly hidden by another model's
            box = (a[1] - a[3], a[2] - a[4], a[1] + a[3], a[2] + a[4])
            return sum(overlap(box, (x - w, y - h, x + w, y + h))
                       for m, other in marks.items() if m != model for _, x, y, w, h in other) > 0
        for _, x, y, w, h in sorted(marks[model], key=lambda a: (covered(a), -a[0])):
            spots = [(x + w + pad, y - th / 2), (x - w - pad - tw, y - th / 2),   # right, left
                     (x - tw / 2, y + h + pad), (x - tw / 2, y - h - pad - th)]   # above, below
            for x0, y0 in spots:
                rect = (x0, y0, x0 + tw, y0 + th)
                if x0 < frame.x0 or rect[2] > frame.x1 or y0 < frame.y0 or rect[3] > frame.y1:
                    continue
                own = [(ox - ow - clear, oy - oh - clear, ox + ow + clear, oy + oh + clear)
                       for _, ox, oy, ow, oh in marks[model]]
                cost = sum(overlap(rect, t) for t in taken if t not in own)
                if best is None or cost < best[0]:
                    best = (cost, rect)
                if cost == 0:
                    break
            if best and best[0] == 0:
                break
        rect = best[1] if best else (frame.x1 - tw, frame.y1 - th, frame.x1, frame.y1)
        text.set_position(AX.transData.inverted().transform((rect[0] - dx0, rect[1] - dy0)))
        taken.append((rect[0] - pad / 2, rect[1] - pad / 2, rect[2] + pad / 2, rect[3] + pad / 2))


fig, axes = plt.subplots(1, len(PANELS), figsize=(SIZE_CM[0] / 2.54, SIZE_CM[1] / 2.54),
                         layout="constrained")
for ax, (column, title) in zip(axes, PANELS):
    draw(ax, column, title)
fig.supxlabel(X_LABEL, color=SECONDARY, fontsize=FONT_SIZE)
fig.canvas.draw()                                                  # lays out the panels, so names are placed in real points
fig.set_layout_engine("none")
for ax, (column, title) in zip(axes, PANELS):
    names(ax, column)
for suffix, extra in ((".png", {"dpi": 300}), (".svg", {})):
    fig.savefig(OUT.with_suffix(suffix), transparent=TRANSPARENT,
                facecolor="none" if TRANSPARENT else "white", **extra)
print("wrote", OUT.with_suffix(".png"), OUT.with_suffix(".svg"))
