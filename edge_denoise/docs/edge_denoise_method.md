# Edge-Domain Denoising for SEM Metrology — Feasibility Study & Method

*2026-08-31 · code: [`edge_denoise/`](../) · experiment report: [`edge_denoise_report.md`](edge_denoise_report.md) (written when the pilot completes)*

**Verdict up front: the idea is feasible, and this document derives exactly in
what form.** Feeding a U-Net Sobel maps and asking for clean Sobel maps back is
mathematically sound — Noise2Noise training survives any fixed linear operator,
and the Sobel pair is exactly invertible (up to a characterized null space) in
$O(N\log N)$. But the derivations below show the *literal* pipeline
(gradient in, gradient out, reconstruct) carries two structural handicaps that a
*hybrid* variant avoids at no cost, so the package implements three
representations and the pilot experiment races them: `image` (= plain N2N; also
the missing first-class N2N pipeline), `gradient` (the literal proposal), and
`hybrid` (image + Sobel channels in, image out, gradient-weighted loss —
the recommended primary).

---

## 1. The problem: precision was never in the objective

The repository's own measurements motivate this experiment better than any
argument from first principles. On the deduplicated MIIC SEM validation split
(`runs/burst_diffusion/miic_p10_dedup*/repeatability_val/summary.md`):

| method | PSNR dB | pixel $\sigma\times10^{-3}$ | CD $3\sigma$ scene (px) | CD bias (px) | center $\sigma$ (px) |
|---|---|---|---|---|---|
| single frame | 14.13 | 195.2 | 1.452 | +0.091 | 0.363 |
| avg of 8 | 23.12 | 68.9 | 0.550 | +0.080 | 0.255 |
| avg of 16 | 26.08 | — | — | +0.048 | — |
| burst one-shot | 35.01 | 8.6 | 0.659 | +0.166 | 0.240 |
| **N2N one-shot** | **35.42** | **6.5** | **0.539** | +0.138 | 0.200 |

Two facts stand out:

1. **A 21 dB PSNR advantage buys almost nothing on CD precision.** The N2N
   model transmits ~11× less pixel noise than 8-frame averaging yet its CD
   $3\sigma$ is only at the avg-of-8 level (the locked-test comparison in the
   burst report §9.2 is a statistical tie, $p = 0.51$). The burst report's
   finding 7 explains why: the learned model's residual variance is
   **edge-concentrated** — its $\sigma$-maps are dark in flat regions and light
   up exactly on contours, the opposite of averaging's uniform noise. The MSE
   objective spends its capacity where metrology does not look.
2. **Learned arms carry ~3× the CD bias of frame averaging** (+0.14 to +0.17 px
   signed vs +0.05), a systematic edge shift contributed by the learned prior.

So the user's premise is confirmed by the data: *metrology information lives in
the edges, and the intensity-MSE training never asked for edges specifically.*
The question is what objective does, and whether it can be trained without
clean images.

## 2. Setting and noise model

Notation. Clean image $x$, noisy acquisitions $y_k = x + n_k$ with
$\mathbb{E}[n_k \mid x] = 0$, $n_k$ independent across $k$ given $x$. The
synthetic pipeline realizes $y = \mathrm{Poisson}(x\,p)/p$ at effective peak
$p = 10$ (variance $x/p$: white, signal-dependent, mean-preserving), with clean
sources pre-scaled to $[0.15, 0.85]$ so clipping bias stays negligible. All
tensors live in the model range $[-1, 1]$ (2× the $[0,1]$ intensity scale).
Nothing below requires Poisson specifically — only zero-mean, per-frame
independent noise.

## 3. Where metrology reads the image: edge-position variance

The repeatability harness measures CD exactly the way a CD-SEM does
(`burst_diffusion/repeatability.py::measure_site`): average $k$ scan lines into
a profile, derive a 50% threshold $\tau$ from the profile's robust extremes,
and locate the sub-pixel threshold crossings. Model the denoised output as
$v = x + r$ with residual $r = b + e$ ($b$ = deterministic bias given $x$, $e$
= zero-mean fluctuation transmitted from the input noise). Let $P(u)$ be the
band-averaged profile, $u_0$ the clean crossing, $m = P_x'(u_0)$ the clean
profile slope. Linearizing $P(\hat u) = \tau$:

$$\hat u - u_0 \;\approx\; -\frac{P_r(u_0) - \Delta\tau}{m},$$

where $\Delta\tau$ is the threshold's own shift. Because $\tau$ is re-derived
per waveform from the profile extremes, a DC offset moves $P_r$ and
$\Delta\tau$ identically and **cancels** — the estimator is DC-invariant, which
is why the gradient representation's lost offset genuinely does not matter for
CD/registration (§5.3). Taking variance over repeated acquisitions, with
$\sigma_e^2$ the per-pixel residual variance *at the edge* and $\bar\rho$ the
mean correlation of residuals between rows within the band:

$$\operatorname{Var}(\hat u) \;\approx\; \frac{\sigma_e^2\,\bigl(1 + (k-1)\bar\rho\bigr)}{k\,m^2},
\qquad
\mathbb{E}[\hat u] - u_0 \;\approx\; -\frac{b(u_0)}{m}.$$

CD is a difference of two crossings and the feature center their mean, so both
inherit this expression (correlation between the two edges shifts constants,
not structure). **Three levers control precision, and one controls bias:**

- $\sigma_e$ **at edges** — precisely where the report found learned residual
  variance concentrated;
- the restored **slope** $m$ — blurring trades PSNR for a shallower slope that
  *amplifies* whatever residual noise remains (a fixed Gaussian blur famously
  beats avg-of-16 on PSNR while ruining CD, burst report §9.1);
- the **row-to-row correlation** $\bar\rho$ of residuals along the edge — band
  averaging only helps by $1/k$ if the residuals decorrelate between scan
  lines; a denoiser with a wide receptive field can make $\bar\rho$ large;
- edge-local residual **bias** $b(u_0)$ — the systematic edge shift behind the
  +0.14 px CD bias of the learned arms.

An objective aimed at metrology should therefore penalize residual error and
its variability *in the edge band of the spectrum*, not per-pixel intensity
uniformly. That is exactly what a gradient-domain loss is (§4.2).

## 4. The gradient domain, made precise

### 4.1 The Sobel operator

`edge_denoise.gradient.sobel` applies the correlation kernels

$$K_x = \frac{1}{8}\begin{pmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{pmatrix},
\qquad K_y = K_x^{\mathsf T},$$

with reflect padding. The $1/8$ makes a unit-slope ramp respond with exactly 1
("intensity change per pixel", the unit edge slopes are read in). Facts, each
covered by a unit test:

- **Linearity & DC annihilation**: $S(ax + by) = aSx + bSy$; constants map to 0.
- **White-noise gain**: for i.i.d. pixel noise of variance $\sigma^2$ each
  gradient channel carries $\tfrac{12}{64}\sigma^2 = 0.1875\,\sigma^2$ (sum of
  squared kernel weights) — *attenuated*, but spatially correlated by the
  $[1,2,1]$ smoothing. For signal-dependent Poisson noise the same holds with
  $\sigma^2$ the local variance.
- **Spectrum** (whole-sample even extension, i.e. the reflect-padding
  geometry): $\hat S_x(\omega) = \tfrac{i}{2}\sin\omega_x\,(1+\cos\omega_y)$
  and symmetrically for $\hat S_y$, so
  $$|\hat S_x|^2 + |\hat S_y|^2 = \tfrac14\left[\sin^2\!\omega_x\,(1{+}\cos\omega_y)^2 + \sin^2\!\omega_y\,(1{+}\cos\omega_x)^2\right] \in [0, 1],$$
  vanishing at DC, rising as $|\omega|^2$ at low frequency, peaking (value 1)
  at $(\omega_x,\omega_y) = (\pi/2, 0)$ — the band where edge profiles live.

### 4.2 A gradient loss is a frequency-weighted loss (the central identity)

Let $r = f(y) - t$ be the residual against the target $t$. For the combined
objective

$$\mathcal L \;=\; \lambda_i\,\mathbb{E}\|r\|^2 \;+\; \lambda_g\,\mathbb{E}\|S r\|^2,$$

Parseval on the extended domain gives

$$\mathcal L \;=\; \int \Bigl[\underbrace{\lambda_i + \lambda_g\bigl(|\hat S_x(\omega)|^2 + |\hat S_y(\omega)|^2\bigr)}_{W(\omega)}\Bigr]\; \mathbb{E}\,|\hat r(\omega)|^2 \;\frac{d\omega}{(2\pi)^2}.$$

With the pilot's $\lambda_i = 1, \lambda_g = 4$ the weight $W$ runs from 1 at
DC to 5 in the edge band: the loss *is* the statement "offsets and shading
matter little; edges matter five times more." This identity also bounds what
to expect, honestly:

- **With unlimited capacity the minimizer does not move.** Pointwise,
  $\nabla_f\,\mathbb{E}\bigl[(f-x)^{\mathsf T}(\lambda_i I + \lambda_g S^{\mathsf T}S)(f-x)\bigr] = 0$
  still gives $f^*(y) = \mathbb{E}[x \mid y]$ wherever
  $\lambda_i I + \lambda_g S^{\mathsf T}S \succ 0$ (everywhere except the null
  modes of §5.2 when $\lambda_i = 0$). Gradient weighting is an
  **error-shaping** method: it redistributes the *approximation and
  optimization* error of a finite network away from edges — it does not define
  a different ideal denoiser. Gains should therefore be expected at the
  size of the model's error budget (the 0.5–1.5 px-CD-3σ gap between the
  learned arms and a hypothetical exact posterior mean), not as a
  step-change. This is also the correct reading of "burst ≈ N2N on
  metrology": same optimum, similar error shaping.
- The corollary is that the *loss weights are the experiment*: everything else
  in the pilot is held bit-identical to the N2N arm.

### 4.3 Noise2Noise survives linear operators (no clean images needed)

With a fresh-frame target $t = y_2 = x + n_2$, independence and zero mean give,
for the image term (Lehtinen et al., 2018) *and equally for the gradient term*,

$$\mathbb{E}\bigl\|S f(y_1) - S y_2\bigr\|^2
= \mathbb{E}\bigl\|S f(y_1) - S x\bigr\|^2 + \mathbb{E}\|S n_2\|^2
\quad(\text{cross term } 2\,\mathbb{E}\langle S f(y_1) - S x,\; S n_2\rangle = 0),$$

because $S n_2$ is zero-mean and independent of $y_1$. **Gradient-domain
supervision is exactly as clean-free as plain N2N** — the deployment story on
real instrument bursts is unchanged. The price is additive loss-floor
variance, which is why train-loss plateaus are expected and predicted:

$$\text{floor} \;\approx\; 4\sigma_{01}^2\left(\lambda_i + \tfrac{12}{64}\lambda_g\right)
\;\;(\text{model range; } \sigma_{01}^2 \approx 0.0385 \text{ for MIIC p10} \Rightarrow 0.154\ \text{for N2N},\ \approx 0.27\ \text{for the hybrid arm}).$$

The trainer logs per-term losses so these floors are checkable, exactly as the
burst trainer's 0.150-vs-0.154 check validated its objective.

## 5. Three representations, and why hybrid is primary

| | input | output | loss terms | recovers image by |
|---|---|---|---|---|
| `image` | $y$ | $\hat x$ | $\lambda_i$ (+ $\lambda_g$) | identity |
| `gradient` | $Sy$ | $\widehat{Sx}$ | $\lambda_g$ | FFT least squares + input mean |
| `hybrid` | $[\,y, Sy\,]$ | $\hat x$ | $\lambda_i + \lambda_g$ | identity |

### 5.1 Why give the network Sobel channels at all?

A first conv layer *could* learn $K_x, K_y$; supplying them costs two input
channels (~1k of 8.95M parameters) and buys an inductive bias: the earliest
features start in edge coordinates, and under a gradient-weighted loss the
optimization does not first have to discover the operator the loss is measured
through. This is the same reasoning as SPSR's gradient branch in
super-resolution (Ma et al., CVPR 2020). It is a finite-capacity argument, and
the pilot's `sobloss` arm (image input, gradient loss) exists precisely to
measure how much of the hybrid's effect is the loss alone.

### 5.2 The pure-gradient pipeline: exact reconstruction, and its two handicaps

**Reconstruction.** Reflect padding makes $S$ a circular convolution on the
whole-sample even extension (period $(2H{-}2)\times(2W{-}2)$; the mirrored
neighbor of boundary sample 0 is sample 1 — exactly what reflect padding
supplies), so

$$\hat u(\omega) = \frac{\overline{\hat S_x}\,\hat g_x + \overline{\hat S_y}\,\hat g_y}{|\hat S_x|^2 + |\hat S_y|^2}$$

solves $\min_u \|S_x u - g_x\|^2 + \|S_y u - g_y\|^2$ *exactly* in
$O(N\log N)$ — Poisson reconstruction (Perez et al., 2003) with the discrete
operator inverted consistently rather than approximated. A predicted field
need not be integrable (curl-free); the solve is then the orthogonal
projection onto realizable fields, which is the right way to discard the
inconsistent part of a network output. Verified: a sub-pixel bar edge at
20.714 px survives `sobel -> reconstruct` at 20.718 px (0.004 px), and the
white-noise gain, ramp response, and round-trips are unit-tested.

**Handicap 1 — the null space is bigger than the folklore says.** Both Sobel
responses factor as (difference in one axis) × ($[1,2,1]$ smoothing in the
other), and *each factor has a Nyquist zero*. Consequently
$|\hat S_x|^2 + |\hat S_y|^2 = 0$ on **the two full Nyquist lines**
($\omega_x = \pi$ for every $\omega_y$, and vice versa), not just at DC plus a
few corners: $\hat S_x$ dies on $\omega_y = \pi$ through its smoothing factor
exactly where $\hat S_y$ dies through its difference factor. Content there is
unrecoverable from Sobel data *by any method*. Worse, next to the lines the
inverse gain $1/|\hat S|$ diverges, so tiny network errors in those
weakly-supervised modes (their loss weight is the *same* vanishing
$|\hat S|^2$) would be amplified without bound. The implementation therefore
zeroes modes with $|\hat S_x|^2 + |\hat S_y|^2 < 10^{-3}\max$ — bounding the
amplification at ~30× while touching only the lines' immediate neighborhood.
For SEM content the lines carry finest-checkerboard detail that is
overwhelmingly noise; but this is a genuine information loss the hybrid
representation simply does not have.

**Handicap 2 — low-frequency error amplification.** The inverse gain
$1/|\hat S| \sim 1/|\omega|$ at low frequency is the unavoidable physics of
integrating a gradient (errors accumulate over distance). The training loss
supervises those modes with weight $|\hat S|^2 \sim |\omega|^2$ — the
supervision is weakest exactly where reconstruction amplifies most. Prediction
errors that are negligible in the gradient domain can therefore reconstruct
into large-scale shading blotches. This costs PSNR and pixel-σ; §3's
DC/threshold-invariance argument says it should cost little CD — the pilot
measures precisely this split.

**The lost DC.** The crop mean of the noisy input is an unbiased estimator of
the clean mean (Poisson is mean-preserving); restoring it adds
$\sigma_{01}/\sqrt{HW} \approx 3\times10^{-3}$ of flat repeatability noise at
$64^2$ — visible in pixel-σ, invisible to CD/registration (DC-invariance, §3).

### 5.3 Predictions (falsifiable, before the pilot ran)

1. `hybrid` improves CD 3σ and center σ over plain N2N, at a small PSNR cost
   (frequency weight is a zero-sum reallocation of error).
2. `gradient` roughly matches the others on CD, but loses visibly on PSNR and
   pixel σ (handicap 2 + DC noise).
3. `sobloss` lands between N2N and `hybrid` — the loss is most of the effect,
   the input channels a smaller share.
4. Learned CD bias (+0.14 px signed at N2N) shrinks where the gradient term is
   active, since edge-shape errors now dominate the objective.

## 6. The consistency loss: precision as an explicit objective

Repeatability can be optimized *directly*. For two independent acquisitions
$y_1, y_2$ of the same $x$:

$$\mathbb{E}\bigl\|f(y_1) - f(y_2)\bigr\|^2 = 2\,\mathbb{E}\bigl[\operatorname{Var}(f(y)\mid x)\bigr]$$

— exactly twice the mean per-pixel repeatability variance the evaluation
measures. `objective.lambda_consistency` adds this term (the burst data's ≥3
replicas per source supply the required third independent frame; the data
layer enforces input/target/consistency replicas pairwise distinct so the
penalty correlates with neither the input nor the target noise). It is a
bias–variance dial with a degenerate end point: alone, it is minimized by any
constant output, so config validation requires a fidelity term next to it, and
the pilot keeps it OFF ($\lambda_c = 0$) to change one thing at a time. The
trainer logs `val/consistency_sigma` = $\sqrt{\mathbb{E}\|f(y_1)-f(y_2)\|^2/2}$
(in $[0,1]$ units) either way, so precision is monitored live during every
training run — the readout the user asked for, available before any
repeatability evaluation.

## 7. Determinism, and why a single-pass regressor (not diffusion)

A metrology estimator should be a *pure function of the measurement*: any
internal stochasticity is added measurement noise. The edge_denoise estimator
is a single deterministic forward pass — same frame in, same image out, so the
repeatability evaluation isolates exactly the transmitted input noise. The
burst sampler is also deterministic, but its precision advantage
(iter-prediction over one-shot) came from *conditioning on a pseudo-average
in which the seed frame carries weight 1/15* — at 15× the inference cost and
~1 dB PSNR deficit to plain N2N (report §9.3). The N2N-vs-burst result
already established that for single-frame denoising the schedule apparatus is
not load-bearing; this experiment therefore builds on the regression family,
where the objective — not the sampler — is the variable under test. (The
burst schedule retains its dose-scalability value; nothing here replaces it
for $m \ge 2$ real frames.)

## 8. Pipeline configuration

- **Backbone**: `burst_diffusion.unet.UNet`, imported (not copied): ch 64,
  mult [1,2,2,2], 2 res-blocks, attention at 16² — 8.95M parameters, identical
  to every prior arm; only `conv_in`/`conv_out` widths change with the
  representation. The timestep conditioning is fed the constant 1.0, the same
  value the existing N2N arm saw, so it reduces to a learned bias.
- **Data**: `burst_diffusion.data.BurstCache` (audited content-group split;
  byte-identical splits across packages given identical data settings) +
  `edge_denoise.data.PairFactory` (seeded, resumable, DataLoader-free;
  aligned crops; input/target/second replicas pairwise distinct).
- **Training**: Adam(2e-4, β₂ 0.999), grad-clip 1.0, EMA 0.999, batch 8,
  64px crops, 30k steps — the burst recipe unchanged. Checkpoints are keyed
  (`kind: edge_denoise`), atomic, and restore all RNG state; `provenance.json`
  (commit, dataset content digest, environment, checkpoint hash) is written
  automatically at completion.
- **Inference**: `Denoiser.from_checkpoint(...).denoise(frames)` — one forward
  pass; the gradient representation reconstructs via §5.2 internally.
  Crops only, never resizes.
- **Evaluation**: `python -m edge_denoise repeatability` bridges into
  `burst_diffusion.repeatability` (extended with `RealizationProvider`), so
  classical ladders, burst arms, and edge arms land in **one table** with
  identical sources, seeds, crops, and CD sites.

## 9. Pilot experiment design

Dataset `data/MIIC-burst-p10-dedup` (96 distinct scenes, 76/10/10
content-group split). **Dev (val) split only — the locked test split is not
touched**; it stays reserved for a single confirmatory run once a method is
frozen. Arms, everything shared bit-identical:

| arm | config | representation | λ_i / λ_g | trains |
|---|---|---|---|---|
| N2N reference | (existing burst checkpoint `miic_p10_dedup_n2n`) | image | 1 / 0 | reused |
| A `hybrid` | `miic_p10_dedup_hybrid.yml` | hybrid | 1 / 4 | 30k steps |
| B `grad` | `miic_p10_dedup_grad.yml` | gradient | 0 / 1 | 30k steps |
| C `sobloss` | `miic_p10_dedup_sobloss.yml` | image | 1 / 4 | 30k steps |

Headline metrics, in order: CD $3\sigma$ scene median (beat 0.539 px), CD
signed/absolute bias (beat +0.138/0.319 px), center σ (0.200 px), shift σ
(0.145 px); PSNR/pixel-σ reported as the cost side. Ten seeds per source, 10
sources, ~37 CD sites — effects below ~15% on scene medians will not separate
at this n; the pilot is a direction-finder, not a confirmatory study. One
training seed per arm (the repo's standing caveat).

## 10. Relation to prior work

Brief pointers, from the author's knowledge (not a systematic review):
Noise2Noise training (Lehtinen et al., 2018) — the fresh-frame target;
gradient-domain image processing and Poisson reconstruction (Perez et al.,
2003) — §5.2's solve; structure-preserving super-resolution with a gradient
branch and gradient loss (SPSR, Ma et al., 2020) — the closest architectural
relative of `hybrid`; Sobel/edge losses are folklore in SR/deblurring;
deep-learning denoising ahead of SEM line-edge-roughness metrology (e.g.
Giannatou et al., 2019) — precedent for denoise-then-measure in this domain.
The specific combination here — gradient-domain *Noise2Noise* with an exactly
inverted discrete operator, evaluated on CD/registration *precision* rather
than PSNR — appears not to be standard practice, which is exactly why the
pilot exists.

## 11. Limitations and follow-ups

- Error-shaping ceiling (§4.2): gains bounded by the finite-capacity error
  budget; a null result against N2N is informative, not a failure of the math.
- $\lambda$ values (1/4) are principled but unswept; the consistency term is
  implemented but unexplored (the most direct precision lever — first
  follow-up if the pilot direction holds).
- Patch-scale (64²) synthetic Poisson noise; real bursts, fixed-pattern noise,
  physical units, and full-image tiling remain open exactly as for
  burst_diffusion.
- Edge-crop sampling bias (weighting training crops toward edge-rich windows)
  is an unexplored cheap variant.
- One seed per arm; scene-level n = 10.
