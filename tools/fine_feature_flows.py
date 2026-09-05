"""Training-flow illustrations for edge_denoise/docs/fine_feature_report.md.

    python tools/fine_feature_flows.py edge_denoise/docs/images

Writes one SVG per flow (real 64-px crops of dev scene 48 embedded as PNG data
URIs, so the reader sees actual frames, targets and outputs, not icons):

  ff_flow_data_n2n.svg     one clean capture -> 16 Poisson replicas -> N2N teacher
  ff_flow_burst_mean.svg   arm A / A-debias: leave-one-out burst-mean target (+ g^-1)
  ff_flow_consistency.svg  arm C: two frames through one network, agreement term
  ff_flow_diffusion.svg    arm D: DDPM prior on clean crops, SDEdit posterior chain
  ff_flow_augment.svg      arm E: one defect field added to every frame of a window
  ff_flow_evaluation.svg   metrology harness and fine-feature diagnostic

Pure text SVG (no matplotlib); fonts fall back through the system stack.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

FONT = "Segoe UI, Arial, Helvetica, sans-serif"
MONO = "Consolas, 'IBM Plex Mono', monospace"
INK, INK2, LINE = "#1C2530", "#475664", "#8A96A3"
BEAM, BEAM_SOFT, GRAIN, GRAIN_SOFT, GREEN, GREEN_SOFT, RED_SOFT = "#14688F", "#E3EEF4", "#A2711C", "#F4ECDD", "#2E7D4F", "#E4F1E8", "#F6E3E0"


# ---------------------------------------------------------------------------
# SVG primitives


class Svg:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.parts: list[str] = []

    def rect(self, x, y, w, h, fill="#FFFFFF", stroke=LINE, rx=10, dash: str | None = None, sw=2):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>')

    def text(self, x, y, s, size=22, fill=INK, weight="normal", anchor="start", mono=False, italic=False):
        family = MONO if mono else FONT
        style = ' font-style="italic"' if italic else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}"{style}>{s}</text>')

    def lines(self, x, y, rows, size=20, fill=INK2, gap=1.35, anchor="start", mono=False):
        for i, row in enumerate(rows):
            self.text(x, y + i * size * gap, row, size=size, fill=fill, anchor=anchor, mono=mono)

    def arrow(self, x1, y1, x2, y2, label: str | None = None, color=INK2, size=17, dash: str | None = None, side="above"):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.5" marker-end="url(#arrow)"{dash_attr}/>')
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dy = -10 if side == "above" else size + 8
            self.text(mx, my + dy, label, size=size, fill=color, anchor="middle")

    def image(self, x, y, size, array01: np.ndarray, label: str | None = None, sub: str | None = None, scale: int = 4):
        arr = np.rint(np.clip(array01, 0, 1) * 255).astype(np.uint8)
        arr = arr.repeat(scale, 0).repeat(scale, 1)
        buffer = io.BytesIO()
        Image.fromarray(arr).save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        self.parts.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="data:image/png;base64,{data}" style="image-rendering:pixelated"/>')
        self.parts.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="none" stroke="{LINE}" stroke-width="1.5"/>')
        if label:
            self.text(x + size / 2, y + size + 24, label, size=19, anchor="middle", fill=INK)
        if sub:
            self.text(x + size / 2, y + size + 46, sub, size=16, anchor="middle", fill=INK2, mono=True)

    def box(self, x, y, w, h, title, rows=(), fill="#FFFFFF", stroke=LINE, title_fill=INK, dash=None, size=18):
        self.rect(x, y, w, h, fill=fill, stroke=stroke, dash=dash)
        self.text(x + 16, y + 30, title, size=21, weight="bold", fill=title_fill)
        self.lines(x + 16, y + 58, rows, size=size)

    def save(self, path: Path, title: str) -> None:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" font-family="{FONT}">'
            f'<title>{title}</title>'
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK2}"/></marker></defs>'
            '<rect width="100%" height="100%" fill="#FFFFFF"/>'
        )
        path.write_text(head + "".join(self.parts) + "</svg>", encoding="utf-8")
        print("wrote", path, f"{path.stat().st_size / 1024:.0f} KB")


# ---------------------------------------------------------------------------
# real crops


def load_crops():
    from burst_diffusion.data import BurstCache
    from edge_denoise.data import ClipDebiaser
    from edge_denoise.gradient import sobel
    from edge_denoise.infer import Denoiser

    cache = BurstCache("data/MIIC-burst-p10-dedup", channels=1, min_replicas=16, min_size=64,
                       val_fraction=0.1, test_fraction=0.1, split_seed=2019)
    source = {s.source_index: s for s in cache.val_sources}[48]
    top, left = 256 - 32, 256 - 32
    win = np.s_[top:top + 64, left:left + 64]
    clean = source.clean[win].astype(np.float64) / 255.0
    frames = np.stack([f[win].astype(np.float64) / 255.0 for f in source.frames])
    loo = frames[1:].mean(axis=0)
    debiased = ClipDebiaser(10.0)(loo)
    crops = {
        "clean": clean, "y0": frames[0], "y1": frames[1], "y2": frames[2], "loo": loo, "debiased": debiased,
        "avg16": frames.mean(axis=0), "noise_level": np.clip(0.5 + (frames[0] - clean) * 1.5, 0, 1),
    }
    x = torch.from_numpy((frames[:1] * 2 - 1).astype(np.float32))[:, None]
    for arm, run in (("control", "miic_p10_dedup_ft_noisy_b16"), ("debias", "miic_p10_dedup_ft_avgdebias_b16")):
        path = Path("runs/edge_denoise") / run / "ckpt_latest.pt"
        if path.is_file():
            crops[arm] = (Denoiser.from_checkpoint(path).denoise(x)[0, 0].numpy() + 1) / 2
    grad = sobel(torch.from_numpy((clean[None, None] * 2 - 1).astype(np.float32)))
    crops["sobel"] = np.clip(grad.norm(dim=1)[0].numpy() * 3, 0, 1)
    # Diffusion: a mid-chain noisy state and the SDEdit output, if the prior exists.
    prior = Path("runs/edge_denoise/miic_p10_dedup_prior/ckpt_latest.pt")
    if prior.is_file():
        from edge_denoise.prior import PosteriorSampler

        sampler = PosteriorSampler.from_checkpoint(prior, peak=10.0, mode="sdedit", num_steps=25, eta=1.0, noise_variance=0.16)
        crops["sdedit"] = (sampler.denoise(x)[0, 0].numpy() + 1) / 2
        rng = np.random.default_rng(0)
        crops["xt_mid"] = np.clip(0.5 + 0.55 * (clean - 0.5) + rng.normal(0, 0.25, clean.shape), 0, 1)
        crops["xt_hi"] = np.clip(0.5 + 0.15 * (clean - 0.5) + rng.normal(0, 0.45, clean.shape), 0, 1)
    return crops


# ---------------------------------------------------------------------------
# flows


def flow_data_n2n(c, out: Path) -> None:
    s = Svg(1600, 620)
    s.text(30, 44, "Data and phase 1 — the Noise2Noise teacher (shared by every arm)", size=28, weight="bold")
    s.text(30, 76, "Dev scene 48, the 64-px crop every table is measured on.  Nothing here ever sees a clean image inside the loss.", size=19, fill=INK2)
    # data column
    s.image(40, 120, 150, c["clean"], "clean capture x", "long-dwell reference")
    s.arrow(200, 195, 310, 195, "Poisson, peak 10", side="below")
    for k, key in enumerate(("y0", "y1", "y2")):
        s.image(320 + k * 50, 120 + k * 26, 150, c[key])
    s.text(445, 340, "16 replicas y_1 … y_16", size=19, fill=INK, anchor="middle", mono=True)
    s.text(445, 364, "same scene, independent noise", size=16, fill=INK2, anchor="middle")
    # training block
    s.rect(600, 110, 960, 380, fill="#F7F9FB")
    s.text(620, 142, "one training sample (aligned 64-px window)", size=20, weight="bold")
    s.image(630, 170, 130, c["y0"], "input y_i", "one replica")
    s.arrow(770, 235, 850, 235)
    s.box(860, 175, 220, 120, "U-Net f (8.95 M)", ["ε-UNet backbone,", "t held constant", "1 frame in, 1 image out"], fill=BEAM_SOFT, stroke=BEAM, title_fill=BEAM)
    s.arrow(1090, 235, 1160, 235, "f(y_i)")
    s.image(1170, 170, 130, c["control"] if "control" in c else c["avg16"], "prediction", "the denoised image")
    s.image(1400, 170, 130, c["y1"], "target y_j, j ≠ i", "a DIFFERENT replica")
    s.arrow(1235, 372, 1465, 372, color=GRAIN)
    s.text(1350, 405, "loss  ‖ f(y_i) − y_j ‖²", size=22, weight="bold", fill=GRAIN, anchor="middle", mono=True)
    s.lines(620, 440, ["The target's noise is independent of the input's, so the L2 optimum is E[x | y_i]: a denoiser learned without a clean image.",
                       "The loss plateaus at the target-noise variance (0.154 in model units) — that plateau is correct, not divergence."], size=17)
    s.lines(30, 530, ["30 000 steps, batch 8, Adam 2·10⁻⁴, EMA 0.999.  The EMA weights of this teacher are the starting point of every fine-tuned arm below;",
                      "the compute-matched control `ft_noisy_b16` continues this exact objective (plus the 4× Sobel term) for 10 000 more steps at batch 16."], size=17)
    s.save(out / "ff_flow_data_n2n.svg", "Data and Noise2Noise teacher")


def flow_burst_mean(c, out: Path) -> None:
    s = Svg(1600, 640)
    s.text(30, 44, "Arm A and A-debias — the burst as a cleaner, unbiased target (recommended recipe)", size=28, weight="bold")
    s.text(30, 76, "Fine-tune the teacher for 10 000 steps (batch 16).  Inference is unchanged: one frame in, one image out.", size=19, fill=INK2)
    s.image(40, 130, 140, c["y0"], "input y_i", "one replica")
    s.arrow(190, 200, 260, 200)
    s.box(270, 140, 220, 120, "U-Net f", ["warm start = teacher", "EMA weights"], fill=BEAM_SOFT, stroke=BEAM, title_fill=BEAM)
    s.arrow(500, 200, 570, 200, "f(y_i)")
    s.image(580, 130, 140, c["debias"] if "debias" in c else c["avg16"], "prediction", "")
    # target construction
    s.rect(800, 110, 770, 250, fill="#F7F9FB")
    s.text(820, 142, "target construction (training time only)", size=20, weight="bold")
    for k, key in enumerate(("y1", "y2")):
        s.image(820 + k * 30, 160 + k * 16, 100, c[key])
    s.text(880, 310, "the other 15 replicas", size=16, fill=INK2, anchor="middle")
    s.arrow(960, 225, 1030, 225, "mean", side="below")
    s.image(1040, 165, 120, c["loo"], "ȳ₋ᵢ", "1/15 of the noise")
    s.arrow(1170, 225, 1240, 225, "g⁻¹ (A-debias)", color=GRAIN, size=15, side="below")
    s.image(1250, 165, 120, c["debiased"], "g⁻¹(ȳ₋ᵢ)", "clip bias removed")
    s.lines(1385, 190, ["g(x) = E[min(Pois(10x),10)]/10", "is the stored frames' mean;", "it sits below x by 0.06 at the", "bright end.  Monotone, so a", "1-D lookup inverts it."], size=14, mono=False)
    # losses
    s.arrow(650, 306, 650, 400, color=GRAIN)
    s.arrow(1310, 342, 1310, 400, color=GRAIN)
    s.rect(540, 410, 1020, 110, fill=GRAIN_SOFT, stroke=GRAIN)
    s.text(1050, 448, "loss  ‖ f(y_i) − T ‖²  +  4 · ‖ S f(y_i) − S T ‖²      T = ȳ₋ᵢ (A)   or   g⁻¹(ȳ₋ᵢ) (A-debias)", size=20, weight="bold", fill=GRAIN, anchor="middle", mono=True)
    s.text(1050, 490, "S = Sobel operator: the edge band where CD is read counts 4× more.  The input replica is never inside its own target.", size=17, fill=INK2, anchor="middle")
    s.lines(30, 560, ["Why it helps: the fresh-frame target of N2N carries the full noise variance; the leave-one-out mean carries 1/15 of it, so the per-step gradient for weak, low-contrast structure is 15× cleaner",
                      "at the same compute.  Why debias: every noisy-target arm otherwise regresses onto the clipped mean g(x) < x — the origin of the +0.1 px systematic CD bias (report §5, §8)."], size=17)
    s.save(out / "ff_flow_burst_mean.svg", "Arm A / A-debias flow")


def flow_consistency(c, out: Path) -> None:
    s = Svg(1600, 640)
    s.text(30, 44, "Arm C — burst-mean target + consistency term (the precision head)", size=28, weight="bold")
    s.text(30, 76, "Two replicas of the same window go through the same network in one step; the outputs are asked to agree.", size=19, fill=INK2)
    s.image(40, 130, 130, c["y0"], "y_i", "replica i")
    s.image(40, 330, 130, c["y2"], "y_k, k ≠ i, j", "replica k")
    s.arrow(180, 195, 300, 215)
    s.arrow(180, 395, 300, 375)
    s.box(310, 230, 230, 130, "U-Net f (shared)", ["one set of weights,", "gradients through", "BOTH passes"], fill=BEAM_SOFT, stroke=BEAM, title_fill=BEAM)
    s.arrow(550, 260, 640, 195)
    s.text(588, 208, "f(y_i)", size=17, fill=INK2, anchor="end")
    s.arrow(550, 330, 640, 395)
    s.text(592, 392, "f(y_k)", size=17, fill=INK2, anchor="end")
    s.image(650, 130, 130, c["debias"] if "debias" in c else c["avg16"], "f(y_i)")
    s.image(650, 330, 130, c["control"] if "control" in c else c["avg16"], "f(y_k)")
    s.rect(830, 150, 730, 100, fill=GRAIN_SOFT, stroke=GRAIN)
    s.text(1195, 190, "fidelity   ‖ f(y_i) − ȳ₋ᵢ ‖²  +  4 · ‖ S f(y_i) − S ȳ₋ᵢ ‖²", size=21, weight="bold", fill=GRAIN, anchor="middle", mono=True)
    s.text(1195, 225, "keeps the estimate anchored to the burst mean (arm A)", size=17, fill=INK2, anchor="middle")
    s.rect(830, 300, 730, 100, fill=GREEN_SOFT, stroke=GREEN)
    s.text(1195, 340, "agreement   λ_c · ‖ f(y_i) − f(y_k) ‖²        λ_c = 1", size=21, weight="bold", fill=GREEN, anchor="middle", mono=True)
    s.text(1195, 375, "= 2·Var(f(y) | x): repeatability itself, penalised directly", size=17, fill=INK2, anchor="middle")
    s.arrow(790, 195, 825, 195, color=GRAIN)
    s.arrow(790, 395, 825, 350, color=GREEN)
    s.lines(30, 570, ["Batch 8 with two forward passes = 16 network inputs per step, the same per-step cost as the batch-16 controls.  Alone the agreement term is minimised by any constant image,",
                      "so the fidelity term is mandatory.  Result: CD 3σ 0.368 px (control 0.477), pixel σ −28 %, at −0.35 dB and a small, not significant bias cost.  On real bursts the pair must be registered first."], size=17)
    s.save(out / "ff_flow_consistency.svg", "Arm C flow")


def flow_diffusion(c, out: Path) -> None:
    s = Svg(1600, 720)
    s.text(30, 44, "Arm D — a diffusion prior on clean crops, conditioned on one frame", size=28, weight="bold")
    s.text(30, 76, "Top: how the prior is trained (never a noisy frame).  Bottom: how one measured frame is turned into a sample.", size=19, fill=INK2)
    # prior training
    s.rect(30, 100, 1540, 250, fill="#F7F9FB")
    s.text(50, 132, "prior training — 40 000 steps on 64-px crops of the 76 TRAIN clean images (flips on)", size=20, weight="bold")
    s.image(60, 160, 120, c["clean"], "clean crop x", "")
    s.arrow(190, 220, 300, 220, "+ noise at level t", size=15, side="below")
    s.image(310, 160, 120, c.get("xt_mid", c["y0"]), "x_t", "t ≈ 500")
    s.image(450, 160, 120, c.get("xt_hi", c["y0"]), "x_t", "t ≈ 900")
    s.arrow(580, 220, 640, 220)
    s.box(650, 165, 240, 120, "U-Net ε_θ(x_t, t)", ["same backbone,", "timestep embedding live"], fill=BEAM_SOFT, stroke=BEAM, title_fill=BEAM)
    s.arrow(900, 220, 1000, 220, "predicted noise", size=15, side="below")
    s.rect(1010, 175, 540, 90, fill=GRAIN_SOFT, stroke=GRAIN)
    s.text(1280, 212, "loss  ‖ ε_θ(√ᾱ_t x + √(1−ᾱ_t) ε, t) − ε ‖²", size=18, weight="bold", fill=GRAIN, anchor="middle", mono=True)
    s.text(1280, 246, "standard DDPM; unconditional samples carry grain, vias, blemishes", size=15, fill=INK2, anchor="middle")
    # posterior sampling
    s.rect(30, 380, 1540, 240, fill="#F7F9FB")
    s.text(50, 412, "inference — SDEdit-style chain: the frame IS the state of the chain at its own noise level", size=20, weight="bold")
    s.image(60, 440, 120, c["y0"], "frame y", "one replica")
    s.arrow(190, 500, 275, 500, "√ᾱ_t* · y", size=15)
    s.text(430, 560, "t* ≈ 115 is the level whose noise matches the frame's variance (0.16)", size=15, fill=INK2, anchor="middle")
    x = 285
    for k in range(6):
        s.rect(x, 470, 74, 60, fill=BEAM_SOFT, stroke=BEAM, rx=6)
        s.text(x + 37, 495, "ε_θ", size=17, fill=BEAM, anchor="middle", mono=True)
        s.text(x + 37, 517, f"t={115 - k * 23 if k < 5 else 0}", size=13, fill=INK2, anchor="middle", mono=True)
        if k < 5:
            s.arrow(x + 76, 500, x + 90, 500)
        x += 92
    s.text(560, 460, "25 DDIM steps, η = 1 (fresh noise each step); each step: predict x̂₀, re-noise to the next level", size=15, fill=INK2, anchor="middle")
    s.arrow(x - 10, 500, x + 40, 500)
    s.image(x + 50, 440, 120, c.get("sdedit", c["avg16"]), "sample", "deterministic given seed")
    s.lines(x + 200, 470, ["No likelihood term at all — the frame enters only as the start state.",
                           "The exact-likelihood chain (`dps`, proximal Poisson step per iteration)",
                           "is documented in §6: stable, but 25–28 dB with this prior."], size=16)
    s.lines(30, 650, ["What it buys: PSNR ≈ N2N, bias at the oracle's level (the prior never saw clipped frames), synthetic grain texture.  What it costs: +41 % pixel σ and +0.2 px CD 3σ as a single sample,",
                      "and LESS blemish contrast than arm A (0.60 vs 0.68): the chain re-samples what the frame does not pin down instead of measuring it.  A display channel, not a metrology estimator."], size=17)
    s.save(out / "ff_flow_diffusion.svg", "Arm D flow")


def flow_augment(c, out: Path) -> None:
    s = Svg(1600, 480)
    s.text(30, 44, "Arm E — defect augmentation: raise the prior odds of blobs and scratches without breaking Noise2Noise", size=28, weight="bold")
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:64, 0:64].astype(float)
    field = -0.06 * np.exp(-((yy - 22) ** 2 / (2 * 4.0**2) + (xx - 40) ** 2 / (2 * 6.0**2)))
    theta = 0.6
    along = np.clip((xx - 20) * np.cos(theta) + (yy - 44) * np.sin(theta), -14, 14)
    px, py = 20 + along * np.cos(theta), 44 + along * np.sin(theta)
    field += 0.05 * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 1.0**2))
    s.image(40, 100, 130, np.clip(0.5 + field * 6, 0, 1), "defect field δ (×6)", "1–3 blobs / scratches")
    s.text(105, 300, "drawn once per", size=16, fill=INK2, anchor="middle")
    s.text(105, 320, "training window", size=16, fill=INK2, anchor="middle")
    s.arrow(185, 165, 320, 165, "+ δ to EVERY tensor", size=16, side="below")
    s.image(330, 100, 130, np.clip(c["y0"] + field, 0, 1), "input y_i + δ")
    s.image(480, 100, 130, np.clip(c["loo"] + field, 0, 1), "target ȳ₋ᵢ + δ")
    s.image(630, 100, 130, np.clip(c["y2"] + field, 0, 1), "agreement y_k + δ", "(if λ_c > 0)")
    s.arrow(770, 165, 830, 165)
    s.box(840, 105, 300, 120, "arm A objective, unchanged", ["‖ f(y_i+δ) − (ȳ₋ᵢ+δ) ‖² + 4·Sobel term"], fill=GRAIN_SOFT, stroke=GRAIN, title_fill=GRAIN, size=16)
    s.lines(1165, 130, ["The same field on input and target keeps", "the noises independent, so the L2 optimum", "is still E[x + δ | y + δ]: the network learns", "that soft stains exist in flats.", "Half of the windows get defects (p = 0.5)."], size=17)
    s.lines(30, 380, ["Outcome (§7): retention of the REAL dev blemishes did not move (0.67 vs 0.68 median), no false features appeared.  The shrinkage of weak features is set by the evidence in one frame",
                      "(median stain SNR 0.45), not by a prior that considers stains rare — a closed hypothesis, and a harmless option for corpora where defects really are rare in training."], size=17)
    s.save(out / "ff_flow_augment.svg", "Arm E flow")


def flow_evaluation(c, out: Path) -> None:
    s = Svg(1600, 560)
    s.text(30, 44, "Evaluation — how every arm is scored (dev split only; the test split stays locked)", size=28, weight="bold")
    s.rect(30, 90, 770, 400, fill="#F7F9FB")
    s.text(50, 122, "metrology harness  (python -m edge_denoise repeatability)", size=20, weight="bold")
    for k, key in enumerate(("y0", "y1", "y2")):
        s.image(60 + k * 24, 150 + k * 12, 90, c[key])
    s.text(125, 290, "10 retakes of the scene", size=16, fill=INK2, anchor="middle")
    s.arrow(215, 205, 300, 205, "denoise each", size=15, side="below")
    for k, key in enumerate(("debias", "debias", "debias")):
        s.image(310 + k * 24, 150 + k * 12, 90, c.get(key, c["avg16"]))
    s.arrow(460, 205, 520, 205)
    s.box(530, 145, 250, 130, "CD-SEM recipe per site", ["16-row band profile,", "50 % threshold between", "robust extremes, sub-pixel", "crossings → CD, centre"], fill="#FFFFFF", size=15)
    s.lines(50, 330, ["Per scene: pooled c4-debiased σ of CD over retakes → 3σ (the headline, median over 10 scenes);",
                      "bias vs the clean-image CD; feature-centre σ; global shift σ; pixel σ; PSNR.",
                      "Paired per scene against the compute-matched control (t and sign tests): sites inside a scene",
                      "share frames and outputs, so the scene is the unit — 37 sites are not 37 observations."], size=15)
    s.rect(820, 90, 750, 400, fill="#F7F9FB")
    s.text(840, 122, "fine-feature diagnostic  (python -m edge_denoise fine-features)", size=20, weight="bold")
    s.image(850, 150, 110, c["clean"], "clean", "")
    s.image(980, 150, 110, c["sobel"], "regions between", "strong edges")
    s.image(1110, 150, 110, c.get("debias", c["avg16"]), "output", "")
    s.lines(1240, 175, ["1. difference-of-Gaussian bands", "   (1–2 … 16–32 px) inside the regions;", "   regress output band on clean band", "   → transfer gain, correlation,", "   Wiener bound from measured noise."], size=15)
    s.lines(840, 330, ["2. structured features = clean 2–12 px components above 4× the grain σ (area ≥ 8 px),",
                       "   each with a single-frame SNR; retention = output contrast / clean contrast on the",
                       "   feature's pixels, per SNR bin and across retakes; false features = output-only",
                       "   components (hallucination rate).  Full frames, blended 64-px tiles."], size=15)
    s.lines(30, 520, ["Both tools run on identical sources, seeds, crops and sites, so a single table compares classical averaging, burst, edge and diffusion arms like for like."], size=17)
    s.save(out / "ff_flow_evaluation.svg", "Evaluation flow")


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "edge_denoise/docs/images")
    out.mkdir(parents=True, exist_ok=True)
    crops = load_crops()
    for fn in (flow_data_n2n, flow_burst_mean, flow_consistency, flow_diffusion, flow_augment, flow_evaluation):
        fn(crops, out)


if __name__ == "__main__":
    main()
