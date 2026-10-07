"""CD / registration 3-sigma vs frame count: two side-by-side panels, one highlighted model.

Edit the settings block below and run:  python tools/plot_3sigma_scatter.py
Input: TSV with columns  Model, Frame #, CD 3sig, Regi 3sig.
Output: PNG (300 dpi) + SVG sized for a 12.7 x 12.7 cm slide box.

At each frame count POR is a short black bar and every model is a marker next to it,
so "below the bar" reads as "better than POR". HIGHLIGHT is drawn in the deck blue;
the other models keep their own light colour.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------- settings
TSV = Path("results.tsv")
OUT = Path("three_sigma_scatter")          # writes OUT.png and OUT.svg
SIZE_CM = (12.7, 12.7)                     # (width, height)
TRANSPARENT = True                         # no white box behind the plot on a coloured slide

PANELS = [                                 # (column, panel title)
    ("CD 3sig", "CD 3σ (nm)"),
    ("Regi 3sig", "Registration 3σ (nm)"),
]
X_LABEL = "Number of frames"

HIGHLIGHT = "Fusion"                       # the model the slide is about
HIGHLIGHT_COLOR = "#1428A0"                # deck blue
OTHER_COLORS = ["#F2A77F", "#7FCFB1", "#ADA3DD", "#EFA9C3"]   # light orange, aqua, violet, pink
MARKERS = ["s", "^", "D", "v"]             # other models, in file order (or MODEL_ORDER)
MODEL_ORDER = None                         # e.g. ["Raw avg", "N2N", "Burst diffusion", "Fusion"]

POR_MODEL = "POR"                          # rows with this model name are drawn as the black bars
POR_COLOR = "#1A1A1A"
POR_BAR = 0.76                             # bar length, fraction of one frame slot

BANDS = True                               # light block behind each frame count
DODGE = 0.17                               # spacing between models within a slot (fraction of slot)
HIGHLIGHT_SIZE, OTHER_SIZE = 7.5, 5.5      # marker sizes in points
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
    "legend.frameon": False, "svg.fonttype": "none", "axes.unicode_minus": False,
})

data = pd.read_csv(TSV, sep="\t")
data.columns = data.columns.str.strip()
models = [m for m in (MODEL_ORDER or dict.fromkeys(data["Model"])) if m != POR_MODEL]
frames = sorted(data["Frame #"].unique())
others = [m for m in models if m != HIGHLIGHT]


def style(model):
    if model == HIGHLIGHT:
        return dict(color=HIGHLIGHT_COLOR, marker="o", ms=HIGHLIGHT_SIZE, mew=1.2, zorder=5)
    k = others.index(model)
    return dict(color=OTHER_COLORS[k % len(OTHER_COLORS)], marker=MARKERS[k % len(MARKERS)],
                ms=OTHER_SIZE, mew=0.9, zorder=4)


def value(model, column, frame):
    hit = data[(data["Model"] == model) & (data["Frame #"] == frame)][column]
    return hit.iloc[0] if len(hit) else math.nan


def draw(ax, column, title):
    slots = range(len(frames))
    if BANDS:
        for i in slots:
            ax.axvspan(i - 0.46, i + 0.46, color=BAND, lw=0, zorder=0)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=1)
    if POR_MODEL in set(data["Model"]):
        for i, f in enumerate(frames):
            y = value(POR_MODEL, column, f)
            ax.plot([i - POR_BAR / 2, i + POR_BAR / 2], [y, y], color=POR_COLOR, lw=2.0,
                    solid_capstyle="round", zorder=3)
    for k, model in enumerate(models):
        offset = (k - (len(models) - 1) / 2) * DODGE
        ys = [value(model, column, f) for f in frames]
        ax.plot([i + offset for i in slots], ys, ls="none", mec="white", **style(model))
    ax.set_xticks(list(slots), [f"{f:g}" for f in frames])
    ax.set_xlim(-0.5, len(frames) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_title(title, loc="left", color=INK, fontsize=FONT_SIZE + 1, fontweight="bold", pad=6)


def legend(fig):
    handles = [plt.Line2D([], [], color=POR_COLOR, lw=2.0, solid_capstyle="round")]
    labels = [POR_MODEL]
    for model in models:
        handles.append(plt.Line2D([], [], ls="none", mec="white", **style(model)))
        labels.append(model)
    ncols = 3
    rows = math.ceil(len(labels) / ncols)              # matplotlib fills columns first;
    order = [r * ncols + c for c in range(ncols) for r in range(rows) if r * ncols + c < len(labels)]
    leg = fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="outside upper left",
                     ncols=ncols, handlelength=1.4, handletextpad=0.5, columnspacing=1.4,
                     labelcolor=SECONDARY)
    for text in leg.get_texts():
        if text.get_text() == HIGHLIGHT:
            text.set_color(INK)
            text.set_fontweight("bold")


fig, axes = plt.subplots(1, len(PANELS), figsize=(SIZE_CM[0] / 2.54, SIZE_CM[1] / 2.54),
                         layout="constrained")
for ax, (column, title) in zip(axes, PANELS):
    draw(ax, column, title)
fig.supxlabel(X_LABEL, color=SECONDARY, fontsize=FONT_SIZE)
legend(fig)
for suffix, extra in ((".png", {"dpi": 300}), (".svg", {})):
    fig.savefig(OUT.with_suffix(suffix), transparent=TRANSPARENT,
                facecolor="none" if TRANSPARENT else "white", **extra)
print("wrote", OUT.with_suffix(".png"), OUT.with_suffix(".svg"))
