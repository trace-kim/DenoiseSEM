# Gradient-Target Ladder & Consistency — Experiment Report

*2026-09-02 · code: [`edge_denoise/`](../) · method & derivations:
[`edge_denoise_method.md`](edge_denoise_method.md) · precedes: pilot report
[`edge_denoise_report.md`](edge_denoise_report.md) · results:
`runs/edge_denoise/repeatability_val_ft_ladder/`*

**TL;DR**

- **The consistency term is the first statistically significant precision
  result in this project.** `ft_consist` (sobloss objective + λ_c = 1, the
  term implemented-but-OFF since the pilot) improves CD 3σ on **10/10 val
  scenes against both references**: −0.094 px mean vs the N2N teacher
  (t = −3.04) and −0.070 px vs the from-scratch sobloss champion
  (t = −3.91). Headline: **CD 3σ scene 0.406 px** (N2N 0.539, sobloss
  0.533), site-pooled 0.723 (−30%), center σ 0.159, shift σ 0.122, pixel σ
  4.52·10⁻³ (−30%). The gains are largest exactly on the hardest scenes
  (src 87 −0.337, src 21 −0.181).
- **The price is real and measured**: per-site |CD bias| grows +0.052 px
  (worse on 25/37 sites, t = +2.31 — also significant), signed bias +0.144
  px, PSNR −0.28 dB. This is the bias–variance dial of method doc §6 doing
  exactly what it said; λ_c = 1 was an uncalibrated first setting, so the
  λ_c sweep is now the decisive next study.
- **The two-phase distillation proposal behaves exactly as its decomposition
  predicts.** With frozen per-scene targets, the loss splits as
  E‖S f(y) − T‖² = Var(S f(y) | x) + ‖E S f(y) − T‖² — a consistency term
  plus an anchor to the teacher's mean. Measured: `ft_distill` takes
  precision gains from the hidden consistency half (scene median 0.484 px,
  better on 9/10 scenes) and **inherits the teacher's accuracy exactly**
  (per-site Δ|bias| +0.002 px, t = +0.28; signed +0.135 vs the teacher's
  +0.138) — while every unbiased-target arm reduced bias. It also
  destabilized the hardest scene (src 87: +0.144 px). The useful content of
  the proposal is the consistency component, which `ft_consist` delivers
  directly, stronger, and without freezing the teacher's bias in.
- **Better gradient targets help, but modestly — the target was not the
  binding constraint.** The supervised oracle (`gradient_target: clean`,
  runnable because the corpus is synthetic) and the leave-one-out noisy
  average (`noisy_mean`, clean-free) give small consistent precision gains
  (9/10 scenes each, t = −2.34 / −2.70 vs N2N, scene 0.514 / 0.511) plus the
  best bias (oracle −0.021 px/site) and the best PSNR (35.68 / 35.60 dB).
  Worth keeping — but the objective, not the target, is where precision
  lives.
- **The fine-tune protocol is validated as a cheap sweep vehicle**: 10k
  steps warm-started from the N2N teacher reproduce (or slightly beat) the
  30k-from-scratch sobloss champion under the same objective (`ft_noisy`:
  scene 0.539, site 0.917 vs 0.959, PSNR +0.15 dB) at ~11 min/run
  (~20 min with the consistency term's second forward pass).

## 1. Protocol

Five arms, all **warm-started from the same phase-1 teacher** — the existing
`miic_p10_dedup_n2n` burst checkpoint (30k steps, EMA weights, bridged onto
the edge backbone by `training.init_checkpoint`) — and fine-tuned **10,000
steps** with the pilot recipe otherwise unchanged (batch 8, 64px crops, Adam
2·10⁻⁴, EMA 0.999, seed 0, dataset `data/MIIC-burst-p10-dedup`, dev split
only). Each arm differs from the control in **one line**:

| arm | config | gradient term chases | λ_c |
|---|---|---|---|
| `ft_noisy` (control) | `miic_p10_dedup_ft_noisy.yml` | fresh noisy frame (= sobloss objective) | 0 |
| `ft_oracle` | `miic_p10_dedup_ft_oracle.yml` | `S`(clean) — supervised ceiling | 0 |
| `ft_avg` | `miic_p10_dedup_ft_avg.yml` | `S`(leave-one-out mean of the other 15 noisy replicas) | 0 |
| `ft_distill` | `miic_p10_dedup_ft_distill.yml` | `S`(frozen per-scene average of the teacher's 16 denoised replicas) | 0 |
| `ft_consist` | `miic_p10_dedup_ft_consist.yml` | fresh noisy frame | 1.0 |

The image term is `λ_i = 1` against a fresh noisy frame in every arm (the
unbiased anchor; also the guard against per-scene target memorization and the
Sobel null space). `λ_g = 4` throughout. Target statistics, for reading the
ladder: the fresh frame is unbiased with full noise variance; `noisy_mean` is
unbiased with ~1/15 of it (leave-one-out is load-bearing — including the
input frame breaks the N2N cross-term); `clean` is exact; the distill average
has near-zero noise but carries the teacher's systematic error whole
(averaging removes jitter by ~√N and bias not at all).

Distillation targets: `python -m edge_denoise distill-targets`, teacher =
the same N2N checkpoint, 86 train+val sources, 16 replicas each, overlapping
64px tiles at stride 48 blended with a floored raised-cosine window
(identity round-trip exact; unit-tested), stored float32 `.npy` +
provenance manifest. The locked test split is refused by construction.

Evaluation: one `python -m edge_denoise repeatability` call holding all five
arms plus `sobloss` (the pilot's 30k-from-scratch champion) and the `n2n`
teacher — identical sources, seeds, crops, and 37 CD sites. Classical rows
reproduce the pilot bit-identically, confirming comparability.

## 2. Results (val, 10 sources × 10 seeds, 37 sites)

| method | PSNR dB | pixel σ ×10⁻³ | CD 3σ **scene** px | CD 3σ site px | CD bias px | CD abs-bias px | CD success | center σ px | shift σ px |
|---|---|---|---|---|---|---|---|---|---|
| single_frame | 14.13 | 195.2 | 1.452 | 2.236 | +0.091 | 0.373 | 98.6% | 0.363 | 0.353 |
| avg_of_8 | 23.12 | 68.9 | 0.550 | 1.558 | +0.080 | 0.197 | 98.6% | 0.255 | 0.131 |
| one_shot@n2n (teacher) | 35.42 | 6.50 | 0.539 | 1.039 | +0.138 | 0.319 | 99.2% | 0.200 | 0.145 |
| one_shot@sobloss (30k scratch) | 35.39 | 6.36 | 0.533 | 0.959 | +0.100 | 0.303 | 98.9% | 0.188 | 0.149 |
| one_shot@ft_noisy | 35.54 | 6.28 | 0.539 | 0.917 | +0.127 | 0.302 | 98.9% | 0.184 | 0.141 |
| one_shot@ft_oracle | **35.68** | 6.27 | 0.514 | 0.956 | +0.123 | **0.298** | 98.9% | 0.188 | 0.139 |
| one_shot@ft_avg | 35.60 | 6.22 | 0.511 | 0.993 | +0.134 | 0.311 | 99.2% | 0.194 | 0.138 |
| one_shot@ft_distill | 35.48 | 6.01 | 0.484 | 1.011 | +0.135 | 0.321 | 98.9% | 0.197 | 0.135 |
| **one_shot@ft_consist** | 35.14 | **4.52** | **0.406** | **0.723** | +0.144 | 0.371 | 98.9% | **0.159** | **0.122** |

(Full table incl. avg_of_2/4/16 and the iterative burst rows:
`runs/edge_denoise/repeatability_val_ft_ladder/summary.md`; σ-maps:
`sigma_maps.png` alongside.)

## 3. Paired analyses

**CD 3σ per scene** (n = 10; negative = better):

| arm | vs n2n: mean Δ px / better / t | vs sobloss: mean Δ px / better / t |
|---|---|---|
| ft_noisy | −0.044 / 9/10 / −1.32 | −0.020 / 6/10 / −1.02 |
| ft_oracle | −0.027 / 9/10 / −2.34 | −0.003 / 6/10 / −0.70 |
| ft_avg | −0.019 / 9/10 / −2.70 | +0.005 / 6/10 / +0.46 |
| ft_distill | −0.010 / 9/10 / −0.56 | +0.014 / 9/10 / +0.44 |
| **ft_consist** | **−0.094 / 10/10 / −3.04** | **−0.070 / 10/10 / −3.91** |

`ft_distill`'s small paired mean hides structure: it improves 9/10 scenes by
−0.011…−0.062 px but regresses the hardest scene (src 87, the pilot's worst
CD σ) by +0.144 px. `ft_consist` improves everything, most where it is
hardest: src 87 −0.337, src 21 −0.181, src 50 −0.101.

**Per-site |CD bias|** (n = 37 sites, paired vs n2n; negative = more
accurate):

| arm | mean Δ px | worse sites | t |
|---|---|---|---|
| sobloss | −0.016 | 19/37 | −1.13 |
| ft_noisy | −0.017 | 19/37 | −1.18 |
| ft_oracle | −0.021 | 17/37 | −1.82 |
| ft_avg | −0.008 | 15/37 | −0.76 |
| ft_distill | **+0.002** | 19/37 | +0.28 |
| ft_consist | **+0.052** | 25/37 | **+2.31** |

The bias column is the ladder's verdict on target choice: every
unbiased-target arm nudges accuracy the right way; the distillation target
freezes it at the teacher's level exactly; the consistency term buys its
variance win with a significant accuracy cost.

## 4. Reading against the pre-registered assessment

The 2026-09-02 assessment of the two-phase proposal made testable claims:

1. *"The averaged-denoised target inherits the teacher's bias; unbiased
   targets can do better"* — **confirmed precisely**: ft_distill's per-site
   Δ|bias| vs the teacher is +0.002 px (t = 0.28), signed +0.135 vs +0.138,
   while oracle/noisy arms shave −0.017…−0.021 px.
2. *"With frozen targets the loss decomposes into consistency + anchor, so
   the useful content is the consistency component"* — **confirmed**, with a
   twist the decomposition itself explains: ft_distill out-precisions the
   *oracle* (0.484 vs 0.514 scene). Because the student is initialized at
   the teacher, the anchor term starts near zero and the whole optimization
   budget flows into variance reduction; the oracle instead spends part of
   its budget moving the mean toward clean (hence its best-in-ladder bias
   and PSNR, and smaller variance gain). Same objective family, different
   split of the budget.
3. *"The oracle is the ceiling for pseudo-label fidelity; run it as the
   gate"* — the gate **passes but shallowly** (−5% scene CD): better
   gradient targets are worth having, not where precision lives. The
   error-shaping ceiling of method doc §4.2 is real.
4. *"λ_c is the theoretically cleaner sibling and should win"* —
   **confirmed and then some**: it is the only arm that beats everything on
   every precision column and every scene, and the first p < 0.05 precision
   effect in the project (predicted qualitatively by §6 — "the only term
   that changes the optimum"; the magnitude, −25% at λ_c = 1, was not
   predicted).
5. *Memorization guard* — all gains are on held-out content groups; the
   consistency win cannot be target memorization (no per-scene target
   exists for it). ft_distill's src-87 regression is the one place the
   frozen-target arm looks unstable.
6. *"Fine-tuning from the teacher makes sweeps 3–6× cheaper at the same
   answer"* — **confirmed**: ft_noisy ≈ sobloss everywhere it matters, at
   10k steps vs 30k.

## 5. Verdict and recommended next steps

**The two-phase pseudo-label pipeline is feasible and behaved exactly as
analyzed — and the experiment it motivated found something better than the
pipeline itself.** The distillation target adds machinery (a teacher pass
over the corpus, frozen targets, a staleness/bias liability) to deliver a
weaker version of what `lambda_consistency` does in one config line. The
recommended line drops phase-2 distillation and promotes the consistency
term, with the ladder's unbiased targets as optional accuracy support.

1. **λ_c sweep × seed replication** (the decisive study): λ_c ∈ {0.25, 0.5,
   1, 2} × ≥3 seeds on the fine-tune protocol (~20 min/run), tracking CD 3σ
   *and* per-site |bias| *and* PSNR to place the knee of the bias–variance
   dial. λ_c = 1 already trades +0.05 px/site accuracy for −25% precision;
   metrology use cases differ on whether that trade is free money or
   disqualifying, so the sweep should deliver the curve, not one point.
2. **Combine λ_c with an unbiased gradient target** (`clean` now,
   `noisy_mean` for deployment realism): the ladder shows the two effects
   are complementary — consistency reduces variance, oracle/avg targets
   reduce bias — one run each to test additivity.
3. **Gradient-domain consistency** (penalize `‖S f(y₁) − S f(y₂)‖²`): the σ
   maps concentrate variance on edges, so aiming the variance penalty there
   may buy the same CD improvement at lower bias/PSNR cost. Small code
   change; natural follow-up to #1.
4. Re-run the smoothness diagnostic (`tools/diagnose_smoothness.py`) on the
   λ_c winner before freezing anything — the consistency term is the arm
   most exposed to the blur-cheat failure mode, and PSNR (−0.28 dB) is too
   forgiving to rule it out alone.
5. Only after a winner exists at seed-replication scale: **one confirmatory
   run on the locked test split.**

## 6. Caveats

One training seed per arm; 10 dev scenes; 64px synthetic-Poisson patches;
λ_c = 1 unswept; the consistency arm's accuracy cost (+0.052 px/site,
t = +2.31) is significant and must be re-measured at every λ_c; slope/LER
transfer not re-diagnosed for these checkpoints (next-step #4); all
development on the val split — the test split remains untouched. The
fine-tune arms saw 40k effective steps of data (30k teacher + 10k) vs
sobloss's 30k, which may contribute to ft_noisy's small PSNR edge.

Reproduce: configs `edge_denoise/configs/miic_p10_dedup_ft_*.yml`; teacher
copy + distill targets under `runs/edge_denoise/ft_ladder/`; harness command
in each config header and in `ft_ladder/repeatability.log`.

## Appendix A — exact objectives per arm, phase by phase

Notation: clean source $x$; noisy replicas $y_1,\dots,y_{16} =
\mathrm{Poisson}(x\,p)/p$ at effective peak $p = 10$, independent per frame
given $x$; every tensor in one training sample is the **same random 64px
crop** of one scene, in model range $[-1, 1]$; $S$ = the 1/8-normalized Sobel
pair (2 channels, reflect padding, §4.1 of the method doc);
$d(a, b) = \mathrm{mean}\,(a - b)^2$ (mean-reduced L2); $f = f_\theta$ is the
network being trained. Per sample, a fresh permutation of the 16 replicas
supplies the input replica $i$, the fidelity-target replica $j$, and the
agreement replica $k$ — pairwise distinct. Batch 8, Adam $2\cdot10^{-4}$,
grad-clip 1.0, EMA 0.999, in every phase.

Weights are written symbolically. The values used in **every** run of this
report are

$$\lambda_{\mathrm{image}} = 1,\qquad \lambda_{\mathrm{gradient}} = 4,\qquad
\lambda_{\mathrm{consistency}} = 1\ (\text{where the term appears}),$$

set by the config keys `objective.lambda_image` / `lambda_gradient` /
`lambda_consistency` (abbreviated $\lambda_i$, $\lambda_g$, $\lambda_c$ in
the table).

At a glance — **one column per phase**, so a row reads left to right exactly
as the training ran. The network input is $y_i$ everywhere except where a
cell says otherwise; ft_consist's phase 2 additionally feeds $y_k$:

| arm | phase 1 — 30k, from scratch | phase 2 — 10k, continuing from phase-1's weights |
|---|---|---|
| single_frame / avg_of_m | no training: the estimate is $y_1$ or $\tfrac1m\sum_{m'} y_{m'}$ | — |
| n2n | $d\big(f(y_i),\,y_j\big)$ | — (its EMA weights are every ft arm's starting point) |
| sobloss | $\lambda_i\, d\big(f(y_i),y_j\big) + \lambda_g\, d\big(S f(y_i),\,S y_j\big)$ — single phase | — |
| hybrid (pilot) | as sobloss, input $[y_i,\,S y_i]$ — single phase | — |
| grad (pilot) | $\lambda_g\, d\big(\hat g,\,S y_j\big)$ with $\lambda_i = 0$, input $S y_i$ — single phase | — |
| ft_noisy | $d\big(f(y_i),\,y_j\big)$ (= n2n) | $\lambda_i\, d\big(f(y_i),y_j\big) + \lambda_g\, d\big(S f(y_i),\,S y_j\big)$ |
| ft_oracle | $d\big(f(y_i),\,y_j\big)$ (= n2n) | $\lambda_i\, d\big(f(y_i),y_j\big) + \lambda_g\, d\big(S f(y_i),\,S x\big)$ |
| ft_avg | $d\big(f(y_i),\,y_j\big)$ (= n2n) | $\lambda_i\, d\big(f(y_i),y_j\big) + \lambda_g\, d\big(S f(y_i),\,S \bar y_{-i}\big)$ |
| ft_distill | $d\big(f(y_i),\,y_j\big)$ (= n2n), then build frozen $T$ | $\lambda_i\, d\big(f(y_i),y_j\big) + \lambda_g\, d\big(S f(y_i),\,S T\big)$ |
| ft_consist | $d\big(f(y_i),\,y_j\big)$ (= n2n) | $\lambda_i\, d\big(f(y_i),y_j\big) + \lambda_g\, d\big(S f(y_i),\,S y_j\big) + \lambda_c\, d\big(f(y_i),\,f(y_k)\big)$ |

with $\bar y_{-i} = \tfrac1{15}\sum_{m\neq i} y_m$ and $T$ the frozen
per-scene teacher average (construction under ft_distill below).

**Classical rows** (`single_frame`, `avg_of_m`) — no training, no phases:
the estimate is $y_1$ or $\tfrac1m\sum_{m'} y_{m'}$ directly.

**n2n** — the shared phase-1 model:

- *Phase 1* (30k steps from scratch; input $y_i$):
  $$\mathcal L_1 = d\big(f(y_i),\,y_j\big)$$
  Plain Noise2Noise — no Sobel term, no agreement term.
- *Phase 2*: none. This checkpoint's EMA weights initialize every `ft_*`
  arm below; call the frozen phase-1 network $f_1$.

**sobloss** — pilot champion, reference row (predates the two-phase
protocol):

- *Single phase* (30k from scratch; input $y_i$):
  $$\mathcal L = \lambda_{\mathrm{image}}\, d\big(f(y_i),\,y_j\big)
  + \lambda_{\mathrm{gradient}}\, d\big(S f(y_i),\,S y_j\big)$$

**ft_noisy** — control:

- *Phase 1*: = n2n above.
- *Phase 2* (10k steps from the phase-1 EMA weights; input $y_i$):
  $$\mathcal L_2 = \lambda_{\mathrm{image}}\, d\big(f(y_i),\,y_j\big)
  + \lambda_{\mathrm{gradient}}\, d\big(S f(y_i),\,S y_j\big)$$

**ft_oracle**:

- *Phase 1*: = n2n.
- *Phase 2* (10k):
  $$\mathcal L_2 = \lambda_{\mathrm{image}}\, d\big(f(y_i),\,y_j\big)
  + \lambda_{\mathrm{gradient}}\, d\big(S f(y_i),\,S x\big)$$

**ft_avg**:

- *Phase 1*: = n2n.
- *Phase 2* (10k):
  $$\mathcal L_2 = \lambda_{\mathrm{image}}\, d\big(f(y_i),\,y_j\big)
  + \lambda_{\mathrm{gradient}}\, d\big(S f(y_i),\,S \bar y_{-i}\big),
  \qquad \bar y_{-i} = \tfrac1{15}\textstyle\sum_{m\neq i} y_m$$

**ft_distill** — the user's proposal, plus an image anchor:

- *Phase 1*: = n2n (frozen result $f_1$).
- *Target build* (between the phases; no training): per scene
  $$T = \mathrm{clip}\Big(\tfrac1{16}\textstyle\sum_{m} f_1(y_m),\,0,\,1\Big)$$
  on full 512² frames (overlapping 64px tiles, stride 48, floored
  raised-cosine blend; `python -m edge_denoise distill-targets`), then
  frozen. By linearity of $S$,
  $S\,T = \tfrac1{16}\sum_m S f_1(y_m)$ — matching the averaged image's
  Sobel is identical to matching the averaged Sobel maps.
- *Phase 2* (10k):
  $$\mathcal L_2 =
  \underbrace{\lambda_{\mathrm{image}}\, d\big(f(y_i),\,y_j\big)}_{\text{anchor (added guard)}}
  + \underbrace{\lambda_{\mathrm{gradient}}\, d\big(S f(y_i),\,S T\big)}_{\text{the proposal: Sobel vs frozen average}}$$

**ft_consist** — the winner:

- *Phase 1*: = n2n.
- *Phase 2* (10k; inputs $y_i$ and $y_k$, both through the *current* $f$,
  gradients through both sides):
  $$\mathcal L_2 = \lambda_{\mathrm{image}}\, d\big(f(y_i),\,y_j\big)
  + \lambda_{\mathrm{gradient}}\, d\big(S f(y_i),\,S y_j\big)
  + \lambda_{\mathrm{consistency}}\, d\big(f(y_i),\,f(y_k)\big)$$
  The agreement term is **image-domain** ($f$, not $S f$) as implemented;
  the Sobel-domain variant $d\big(S f(y_i),\,S f(y_k)\big)$ is next-step #3
  and has not been run.

**hybrid / grad** — pilot arms, single phase (30k from scratch): hybrid is
sobloss's $\mathcal L$ with network input $[y_i,\,S y_i]$; grad takes input
$S y_i$, outputs a Sobel field $\hat g$, trains
$\mathcal L = \lambda_{\mathrm{gradient}}\, d(\hat g,\,S y_j)$ with
$\lambda_{\mathrm{image}} = 0$, and inverts by FFT least squares at
inference.

Notes:

- Every phase-2 loss keeps the fidelity terms switched on deliberately: the
  agreement term alone is minimized by any constant output
  (config-rejected); a Sobel-only objective leaves the operator's null
  space (DC + the two Nyquist lines) unconstrained; frozen per-scene sheets
  over 76 train scenes invite memorization. How small
  $\lambda_{\mathrm{image}}$ can go in phase 2 is an open ablation — every
  run here used $\lambda_{\mathrm{image}} = 1$.
- **ft_consist is pairwise and symmetric**: one replica pair per sample per
  step, both $f(y_i)$ and $f(y_k)$ computed by the *current* model in the
  same graph with gradients through both sides (a mutual pull, not a match
  against a frozen reference). The pairwise expectation is
  $\mathbb E\,d(f(y_a), f(y_b)) = 2\,\overline{\mathrm{Var}}(f(y)\mid x)$ —
  an unbiased Monte-Carlo probe of the full across-replica variance — so
  training effectively visits all $\binom{16}{2}$ pairs of every scene; a
  k-replica batched variant would trade lower estimator noise for k forward
  passes and is not obviously better at fixed compute.
- Only the **gradient-term reference** differs across
  ft_noisy/oracle/avg/distill; the fidelity term, data stream, and recipe
  are shared bit-for-bit. Replica-distinctness ($i \neq j \neq k$) is what
  keeps every term's zero-cross-term argument valid (§4.3 / §6 of the
  method doc).

### A.1 Evaluation without ground truth (added 2026-09-03)

The user's deployment framing — no golden truth on real instruments; even
long averages carry drift and charging; the goal is single-shot precision
*better than long-time measurements* — assigns different statuses to the two
halves of the table. **Precision columns (CD/center/shift σ across retakes)
use no ground truth** and transfer to real data unchanged; on this corpus
the goal is already met (ft_consist single-shot 0.406 px vs avg-of-8's
0.550 at 8× dose, ≈ the extrapolated 16-frame level). **Bias columns exist
only in synthetic-land** (and even here "clean" is a real capture with
residual grain, report §6 of the pilot) — they should be read as
diagnostics of *systematic-error stability* (pattern-dependent bias does
not calibrate out; a constant offset does), and they are the last cheap
place to catch a blur-cheat. Consequences for the next steps: the λ_c sweep
optimizes precision margin over the classical ladder subject to a
bias-stability bound; the consistency term is the deployment-preferred
objective (needs neither truth nor averaging) with one new requirement —
on real drifting bursts the pair (and the N2N target) must be registered
first, or pixel-wise agreement punishes the drift itself and pushes toward
blur; the smoothness diagnostic on the λ_c winner is mandatory, not
optional.
