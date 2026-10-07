"""CD / registration 3-sigma vs frame count: two side-by-side panels, one highlighted model.

Edit the settings block below and run:  python tools/plot_3sigma_scatter.py
Input: TSV with columns  Model, Frame #, CD 3sig, Regi 3sig.
Output: PNG (300 dpi) + SVG sized for a 12.7 x 12.7 cm slide box.

At each frame count POR is a short black bar and the models are markers next to it,
ordered by value (highest on the left), so "below the bar" reads as "better than POR".
HIGHLIGHT is drawn in the deck blue; the other models keep their own light colour.
Each model is named beside its marker at the last frame count (no legend).
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
LABEL_SIZE = 6.5                           # model names next to the last frame count
FONT = ["Arial", "DejaVu Sans"]            # matplotlib renders the variable Noto Sans KR as Thin
FONT_SIZE = 9
INK, SECONDARY, BASELINE, GRID, BAND = "#1A1A1A", "#4A4A4A", "#BDBCB6", "#E9E8E3", "#F4F3EF"
LEADER = "#C8C7C2"                         # thin line from a name to its marker
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


def last_point(model, column):
    """(value, slot) at the model's own last frame count; the file's last one may not include it."""
    for i in range(last, -1, -1):
        y = value(model, column, frames[i])
        if not math.isnan(y):
            return y, i
    return math.nan, last


def names(ax, column):
    """Name every model (and POR) beside its last marker, pushed apart to not overlap."""
    entries = []
    for m in models:
        y, i = last_point(m, column)
        entries.append((m, y, i + offset(m, column, frames[i])))
    if has_por:
        y, i = last_point(POR_MODEL, column)
        entries.append((POR_MODEL, y, i + POR_BAR / 2))
    entries = sorted((e for e in entries if not math.isnan(e[1])), key=lambda e: e[1])
    bottom, top = ax.get_ylim()
    height_pt = ax.get_window_extent().height * 72 / ax.figure.dpi
    gap = LABEL_SIZE * 1.25 / height_pt * (top - bottom)
    ys = [e[1] for e in entries]
    for i in range(1, len(ys)):                                    # push up past the one below
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for i in range(len(ys) - 1, -1, -1):                           # then back down under the top
        ys[i] = min(ys[i], (ys[i + 1] if i + 1 < len(ys) else top) - gap)
    x_text = last + 0.5 + 0.3
    for (model, y, x), y_text in zip(entries, ys):
        bold = model in (HIGHLIGHT, POR_MODEL)
        ax.plot([x + 0.06, x_text - 0.05], [y, y_text], color=LEADER, lw=0.5, zorder=2, clip_on=False)
        ax.text(x_text, y_text, model, va="center", ha="left", fontsize=LABEL_SIZE, clip_on=False,
                color=HIGHLIGHT_COLOR if model == HIGHLIGHT else INK if bold else SECONDARY,
                fontweight="bold" if bold else "normal")


fig, axes = plt.subplots(1, len(PANELS), figsize=(SIZE_CM[0] / 2.54, SIZE_CM[1] / 2.54),
                         layout="constrained")
for ax, (column, title) in zip(axes, PANELS):
    draw(ax, column, title)
fig.supxlabel(X_LABEL, color=SECONDARY, fontsize=FONT_SIZE)
fig.canvas.draw()                                                  # lays out the panels, so name spacing is in real points
for ax, (column, title) in zip(axes, PANELS):
    names(ax, column)
for suffix, extra in ((".png", {"dpi": 300}), (".svg", {})):
    fig.savefig(OUT.with_suffix(suffix), transparent=TRANSPARENT,
                facecolor="none" if TRANSPARENT else "white", **extra)
print("wrote", OUT.with_suffix(".png"), OUT.with_suffix(".svg"))
