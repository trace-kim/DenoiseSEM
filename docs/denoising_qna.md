# Denoising SEM Frames — Questions & Answers

A record of the design-review questions asked over the project, with the
short answer first, the reasoning second, and the measurements that back
each answer. Part A (Q1–Q6) is the burst-averaging diffusion review of
2026-08 (formerly `burst_diffusion/docs/burst_diffusion_qna.md`). Part B
(Q7–Q12) is the 2026-09 review of the burst-fusion study: what a single frame
can and cannot carry, what a burst teaches at training time, and which
transforms help.

Companion documents: burst diffusion
[method](../burst_diffusion/docs/burst_diffusion_method.md) ·
[report](../burst_diffusion/docs/burst_diffusion_report.md) ·
[guide](../burst_diffusion/docs/burst_diffusion_guide.md); edge_denoise
[method](../edge_denoise/docs/edge_denoise_method.md) ·
[fine-feature report](../edge_denoise/docs/fine_feature_report.md) ·
[burst-fusion report](../edge_denoise/docs/burst_fusion_report.md).

Most numeric claims in Part A can be reproduced with
`python tools/probe_burst_predictions.py`; the detection bound of Part B with
`python tools/detection_bound.py`.

---

# Part A — Burst-averaging diffusion (2026-08)

## Q1. What exactly is the "single forward pass" (one-shot)?

**Short answer:** calling the trained U-Net once on the raw noisy measurement
at the highest noise level — `model(noisy_frame, t=T)` — and using its output
directly as the denoised image. No loop.

The U-Net is a function of an image and a noise level. One-shot is one
evaluation at $t = T$ ("this input is a single raw frame"). It is not a
separate mechanism from the iterative sampler: it is *exactly the sampler's
first network call* (equivalently, the sampler run with the length-one
schedule $[T]$, which is how `evaluate` computes the `one_shot` row). The
iterative rows in the results differ only in continuing from there — folding
the prediction into a running average and re-querying at lower $t$.

Why a single call already denoises here — but not in DDPM — is Q3 and Q5.

---

## Q2. What is Noise2Noise, and how does this work relate to it?

**Short answer:** Noise2Noise (Lehtinen et al., 2018) shows a denoiser can be
trained with *noisy targets* instead of clean ones, and converges to the same
function as clean-target training. Our Theorem 1 is that argument; the burst
structure extends it with a noise-level schedule, $t$-conditioning, and a
diffusion-style sampler.

Supervised denoising wants (noisy, clean) pairs, but a clean image often
cannot be acquired (dose limits, beam damage). Noise2Noise replaces the clean
target with a *second independent noisy observation* of the same scene. Under
$L^2$ loss the optimal predictor is the conditional mean, and independent
zero-mean noise vanishes inside a conditional mean:
$\mathbb{E}[y^{\text{tgt}} \mid y^{\text{in}}] = \mathbb{E}[x_0 \mid y^{\text{in}}]$.
The noisy target is an unbiased stand-in — gradient noise grows, the optimum
does not move. (The loss must match the noise: $L^2$ for zero-mean noise;
impulse noise like salt-and-pepper needs a median-seeking loss, which is why
the generator warns about it.)

Relation to this work, precisely:

- the `fresh` target rule ($K \notin S$) **is** Noise2Noise's independence
  condition; Theorem 2 shows the degeneracy when it is violated;
- the input side is a *partial average*, so one burst yields training pairs at
  $T$ input-noise levels instead of N2N's single level;
- the network is $t$-conditioned — one model spanning the schedule;
- the diffusion-style sampler is wrapped around that family — this is the
  novel part the experiment tested (and, in v1, the part that did not add
  value over one-shot; see Q4).

Honest framing: the winning `one_shot` result is essentially a noise-level-
conditioned Noise2Noise denoiser; the experiment's open question was whether
iteration adds anything on top.

---

## Q3. The loss trains the network to predict a *noisy frame*. Why does its output end up close to the *clean* image? Was the objective even set correctly?

**Short answer:** because MSE turns "predict a random thing" into "predict its
average", and the average of `clean + zero-mean noise` is the clean image.
The objective is verified correct three independent ways below.

The network is not asked to predict "the noisiest image" — it is asked to
predict **a randomly chosen fresh frame it has never seen**. The dice analogy:
predict a fair die roll under squared error, and your best answer is 3.5 (the
mean), never an actual face. During training, the *same input* appears with
*different targets* on different steps (frame 3 today, frame 11 tomorrow);
each pulls the output toward itself; the only stationary point is the mean of
the pulls — the clean image. The network never "chooses" to denoise:
denoising is the unique loss-minimizing compromise between targets it cannot
tell apart. Two conditions make this work: the target's noise must be
**zero-mean** (assumption A1) and **independent of the input** — which is
exactly why the target frame is *excluded* from the averaged subset
(`target_mode: fresh`; the included variant provably collapses to the
identity, method doc Theorem 2).

The three receipts (from the trained BBBC038 model, held-out source):

1. **Location of the prediction.** If training had produced a noisy-image
   generator, its output would be close to noisy frames and far from clean.
   Measured, it is the reverse:

   | distance | value |
   |---|---|
   | prediction vs clean | 36–43 dB (close) |
   | prediction vs any actual noisy frame | ~17.2 dB (far — exactly that frame's own noise distance) |
   | raw frame vs clean | 17.3 dB |
   | raw frame vs another raw frame | 14.3 dB |

   Note the last row: two noisy frames are ~14 dB apart *from each other* —
   even a perfect noisy-image generator could never be closer than ~17 dB to
   a particular unseen frame, because that frame's noise is unguessable.

2. **The loss floor.** $\mathbb{E}\lVert\hat\varepsilon - y\rVert^2 =
   \mathbb{E}\lVert\hat\varepsilon - x_0\rVert^2 + \sigma^2$ exactly, so the
   loss cannot go below the target's noise variance and reaches it only when
   the output is the clean estimate. Measured: final training loss 0.150 vs
   predicted floor 0.154 (MIIC); per-image,
   $\mathrm{MSE}(\text{prediction}, \text{fresh frame}) = 0.0191$ vs
   $\sigma^2 = 0.0186$. The objective was not just correct — it was
   *saturated*.

3. **Unit tests** pin the mechanics: the fresh target is asserted to be
   excluded from the averaged subset across hundreds of draws, the subset
   averaging is checked exactly, and the sampler reproduces true burst
   averaging when fed real frames.

---

## Q4. Isn't the iteration just a cumulative average of clean images? Why would that be *worse* than one-shot?

**Short answer:** it is not an average of clean images — it is
`(1 raw noisy frame + T predictions) / (T+1)`, the predictions are *correlated*
(so their errors do not cancel), and predictions after the first are made
from inputs the network never saw in training, which measurably degrades
them.

Three corrections to the premise:

1. **The raw frame stays in.** By the sampler's closed form,
   $x^{\text{avg}} = (y_1 + \sum_t \hat\varepsilon_t)/(T+1)$: the noisy
   measurement keeps weight $1/(T+1)$ forever.
2. **The predictions are not independent.** All $T$ predictions are
   deterministic functions of the same $y_1$. Averaging them is asking the
   same witness fifteen times — correlated errors do not cancel the way
   independent frame noise does (which is the only reason real burst
   averaging works).
3. **Predictions 2..T come from out-of-distribution inputs.** In training, a
   level-$t$ input is `clean + white grainy noise of variance` $\sigma^2/m$.
   The sampler's pseudo-average is `clean +` $\sigma^2/m^2$ `of real noise +
   smooth correlated model error` — much cleaner than $t$ promises, with the
   wrong noise texture. The network's correction is calibrated per level,
   like a lab tech told "shot at ISO 6400, compensate accordingly" and handed
   an ISO 100 photo: it miscorrects.

The controlled measurement (same network, same $t = 1$, different input):

| input at $t=1$ | prediction vs clean |
|---|---|
| REAL average of 15 frames (in-distribution) | **42.8 dB** |
| the sampler's own pseudo-average | **36.0 dB** |

6.8 dB lost purely to the input distribution. Fed real $m$-frame averages the
prediction improves monotonically (36.2 → 42.8 dB for $m = 1..15$) — the
denoiser family is healthy; only the self-generated inputs are foreign.
Across the validation set, the mean *first* prediction of the trajectory
scores 40.2 dB and the mean *last* prediction 34.0 dB.

Practical corollary: the $t$-conditioning is useful today for *real*
multi-frame inputs — acquire $k$ real frames, average, and query at the
matching level (`Sampler.run(x, schedule=[T+1-k, ...])`). (Whether that is
admissible in production is Q7.)

---

## Q5. In DDPM the U-Net "learns the Gaussian noise". By the same conditional-mean argument, shouldn't DDPM collapse to a **zero array** (the mean of the Gaussian)? And is our collapse-to-clean because clean-image information is embedded in the noisy frame?

**Short answer:** DDPM does not collapse because its target is *inside* its
input — the conditional mean $\mathbb{E}[\varepsilon \mid x_t] \ne 0$. Your
zero-array intuition is exactly right for the variant you describe: a *fresh*
Gaussian target would collapse DDPM to zeros. And yes — the embedded clean
information is precisely why our fresh-target choice works instead of
zero-collapsing.

First, a premise correction: DDPM's U-Net does not learn the Gaussian
*distribution* (that is fixed and known). It learns to **identify the specific
noise realization inside its input**. Since
$x_t = \sqrt{\bar\alpha}\,x_0 + \sqrt{1-\bar\alpha}\,\varepsilon$, the target
is algebraically entangled with the input, and

$$\mathbb{E}[\varepsilon \mid x_t] = \frac{x_t - \sqrt{\bar\alpha}\,\mathbb{E}[x_0 \mid x_t]}{\sqrt{1-\bar\alpha}} \ne 0 .$$

The unconditional mean of $\varepsilon$ is zero, but the network never
predicts unconditionally — it sees $x_t$. And that formula shows
$\hat\varepsilon$-prediction and $\hat x_0$-prediction are the *same estimate
in two coordinate systems*: "learning the noise" in DDPM **is** learning the
clean image, through the learned image prior that separates signal from
noise.

The full symmetry — each framework is forced onto the opposite target choice
by what its $\varepsilon$ *contains*:

| | DDPM | burst diffusion |
|---|---|---|
| what $\varepsilon$ is | pure Gaussian noise — no signal, mean $0$ | a real frame — clean image + noise, mean $x_0$ |
| predict an **included** $\varepsilon$ (inside the input) | works: the weighted mixture is recoverable via the image prior | fails: identity collapse (Theorem 2) |
| predict a **fresh** $\varepsilon$ (independent of the input) | fails: **zero-array collapse** (the questioner's scenario) | works: collapses onto $\mathbb{E}[x_0 \mid \text{input}]$ — the denoiser |
| so training must target | the included realization | a fresh frame |
| what $\hat\varepsilon$ means at inference | the input's own noise (a full-strength clean estimate in disguise) | the clean-image estimate directly |

Both methods *exploit* a collapse to the conditional mean; the art is
arranging the target so the collapse lands somewhere useful.

On the embedded-clean-information question — yes, and it plays two distinct
roles:

1. **In the target frame:** the fresh frame is $x_0 + $ unpredictable noise,
   so its conditional mean is $x_0$, *not* $0$. This is exactly why our
   fresh-target training does not zero-collapse the way fresh-target DDPM
   would.
2. **In the input frame:** the input also contains $x_0$, which makes the
   prediction specific to *this* image. With no input at all, the optimum
   would be the dataset-average image — a gray blur. The input supplies the
   "which image"; the target's structure supplies the "clean".

---

## Q6. Wouldn't self-rollout finetuning make the model learn *its own* noise distribution instead of the real equipment noise? Is it still valid?

**Short answer:** no — provided one invariant holds: **model outputs enter
only as inputs; the targets remain real measured frames, always.** Theorem 1
never uses how the input was manufactured, so the optimum on pseudo-inputs is
still $\mathbb{E}[x_0 \mid \text{input}]$, graded by reality.

A self-rollout finetuning batch: run the sampler a few steps on a real frame,
take an intermediate pseudo-average as the *input*, and train against **a
real fresh frame from the same burst**. The target's noise is still real,
zero-mean, and independent of everything — including the model's own
machinations — so the conditional-mean argument goes through unchanged. The
network learns to *finish the job from its own intermediate states*, but it
is never rewarded for agreeing with itself.

The failure mode the question correctly worries about is the converse: model
outputs on the **target** side (self-distillation). Then the loss rewards
self-agreement and the model can drift toward its own artifacts with nothing
anchoring it to the instrument. (Imitation-learning analogy: this is DAgger —
train on *your own policy's states* with *expert labels*. Own states, real
labels: safe. Own labels: drift.)

Where "learning the real noise" actually lives in this architecture: the
model never generates noise; it learns a correction *calibrated to the real
noise's statistics* (variance per level, signal-dependence, spatial texture)
because it minimizes MSE under exactly that noise on the target side and on
the real-average inputs. Self-rollout changes neither — real frames stay as
every target, and real-average inputs stay in the training mix (augment,
never replace).

Open design choices (validity does not depend on them; performance will):
what $t$ means for pseudo-inputs (nominal sampler step vs matched noise
level), rollout depth distribution, and EMA-vs-live weights for generating
rollouts. Success criteria are measurable: after finetuning, the A/B curves
of `tools/probe_burst_predictions.py` should merge (baseline: 42.8 vs
36.0 dB at $t=1$), real-input validation metrics must not regress, and the
goal is `iter_prediction` $\ge$ `one_shot`.

**Outcome (2026-08-30):** implemented as the opt-in `training.rollout`
stage (decisions: nominal-step $t$, stop levels uniform over $\{1..T-1\}$,
on-the-fly EMA-weight rollouts — reasoning in the method doc §5) and run
on both baselines. The goal holds on both datasets: `iter_prediction`
40.5 vs `one_shot` 39.9 dB (BBBC038) and 35.9 vs 35.6 dB (MIIC), with
real-input metrics essentially unchanged. A **step-matched** control (each
baseline plainly continued for the same 10k extra steps, no rollout)
leaves `iter_prediction` at its baseline value, so the change is
attributable to the rollout mechanism, not to more gradient steps. (Step-
matched, not FLOP-matched: a rollout step costs ~3.8× a plain one.) One refinement to the success
criterion as originally stated: the A/B curves *cannot fully* merge,
because the deterministic sampler makes every pseudo-state a function of
the single frame $y_1$ — B is information-bounded by one measurement
while A consumes fifteen. The validation-set A−B gap at $t=1$ closed from
10.9 to 4.3 dB (BBBC038) and 3.8 to 3.0 dB (MIIC); the remainder is that
ceiling, not residual distribution mismatch. Full tables: report §8.

*Audit note (2026-08-30, `burst_diffusion_audit_handoff.md`):* the MIIC
figures in this outcome were measured on the pre-deduplication MIIC split,
whose validation sources overlapped training content; treat them as
development numbers. The BBBC038 figures and the mechanism are unaffected;
every later MIIC result in this repository uses the content-deduplicated
split.

---

# Part B — Single-frame inference and what a burst can teach (2026-09)

Context. The burst-fusion study (`edge_denoise/docs/burst_fusion_report.md`)
registered drifting bursts from their own noisy frames and fused K frames at
inference: CD 3σ per scene 0.594 px (best single-frame arm) → 0.349 px with
four frames → 0.214 px with sixteen, blemishes at 78–89 % of their contrast
against 48–70 %, within 1.5 % of a clean-target oracle. It changed the
inference input from one frame to K frames, which production does not allow.
The questions below followed.

## Q7. Does the burst-fusion method take a single noisy frame at inference? Is a multi-frame input acceptable?

**Short answer:** no, it takes K frames of one burst (K = 4 or 16 for the
headline numbers), and no, single-frame inference is a hard production
requirement: time is money at acquisition, and the point of training on
bursts is to *gather frames once*, then denoise one frame. The multi-frame
results therefore do not apply to the production pipeline.

What was reported and what it means under that constraint:

- The same network does run on one frame (K = 1). That row is honest and not
  a win: CD 3σ 0.691 px against 0.594 for the dedicated single-frame control,
  with better pixel σ, PSNR (+0.8 dB) and bias. The dedicated single-frame
  arms of the fine-feature report remain the reference for single-shot use.
- "The model still beats averaging K frames" is true (K = 16: 0.214 px vs
  1.29 px for the drifting average, which also loses 25 % of its CD
  measurements to drift) but is not a reason to change a production pipeline
  that will not take K frames.
- What carries over to single-frame training on real bursts: the
  registration of drifting training bursts from their own frames (the stock
  global correlation peak fails by 5–15 px on periodic line patterns; a
  bounded search plus Gauss–Newton refinement on the *denoised* frames
  registers 200 held-out bursts to 0.04 / 0.13 px rms, and registering on the
  network's own outputs gives the same figure), and the loss that reads raw
  frames of the burst as targets through a warp of the *prediction* — no
  averaged target, no resampled target, no pre-registered data. At m = 1 that
  is exactly "one frame in, the other raw frames of the burst as targets".

---

## Q8. What is an "oracle"? A network trained noisy-in, clean-out cannot be a fundamental proof of anything.

**Short answer:** correct. "Oracle" means a network trained with the clean
image as its regression target; it bounds what an MSE-trained network of that
size can do, because no noisy or averaged target beats the clean one, and it
bounds nothing else. The fundamental limit is a detection bound, and it can
be computed without any model.

Two facts to keep apart:

- **Shrinkage is a property of the loss, not of the frame.** Any MSE-trained
  model outputs the conditional mean, and a conditional mean renders an
  uncertain feature at contrast × posterior probability. That is why every
  arm looks "too smooth", and it is escapable by changing the estimator
  (Q10).
- **Detectability is a property of the frame.** Whether *any* method can
  tell that a feature is there is a Neyman–Pearson question. The likelihood-
  ratio test with the feature's exact shape and position handed to the
  detector is the best detector that can exist; no trained model, no prior,
  no estimator beats it.

`python tools/detection_bound.py` computes that test for each of the 112
structured features of the ten dev scenes (Poisson likelihood ratio, present
vs absent, 400 draws per hypothesis, threshold set on the absent draws), on
one peak-10 frame and at multiples of that dose:

| single-frame SNR bin | features | median area | median contrast | ×1 dose: detected at 5 % / 1 % false alarm | ×2 | ×4 | ×8 | ×16 |
|---|---|---|---|---|---|---|---|---|
| 0.0 – 0.4 | 37 | 9 px | 0.025 | 12 % / 4 % | 16 / 6 | 24 / 10 | 41 / 20 | 64 / 42 |
| 0.4 – 0.6 | 43 | 15 px | 0.027 | 15 % / 5 % | 21 / 8 | 34 / 16 | 52 / 27 | 78 / 57 |
| 0.6 – 1.0 | 22 | 32 px | 0.029 | 24 % / 9 % | 39 / 19 | 61 / 36 | 84 / 61 | 98 / 91 |
| 1.0 – 2.0 | 10 | 78 px | 0.036 | 45 % / 19 % | 61 / 39 | 89 / 77 | 100 / 97 | 100 / 100 |

(SNR = contrast × √area / local Poisson σ, the fine-feature diagnostic's
definition.) On one frame at this dose, a detector that already knows what
it is looking for finds the median feature in one attempt out of seven and
the strongest stains in fewer than half, while accepting one false alarm in
twenty. Any method that "keeps" those features from one frame must also keep
noise blobs of the same kind at a comparable rate. This is not a property of
MSE, of U-Nets, or of how the training set was gathered; it is the photon
count of the frame. Consequences:

- what single-frame methods can still improve is how the features the frame
  *does* support are rendered (the SNR ≥ 1 bin), at the price of a
  false-feature rate that must be reported next to the retention number;
- dose, not the model, moves the bound: at 4× the electrons per frame the
  strongest stains are found 89 % of the time, at 8× 100 %, and the median
  feature reaches 52 %. The production dose is therefore the number that
  decides what a single frame can carry; the synthetic corpus sits at ten
  electrons per pixel (11 gray levels).

For comparison, the empirical bound: the clean-target oracle network keeps
0.61 of the median feature's contrast and 0.81 of the SNR ≥ 1 bin from one
frame; the control keeps 0.66 / 0.84 (the two differ in feature-set details).

---

## Q9. The burst shows how the noise recedes with averaging, i.e. the noise distribution. If a model extracts that distribution, can it remove the noise from a single frame?

**Short answer:** partly. The burst teaches the *prior* (what fine features
look like), and that is real. It cannot teach the noise *realization* in the
frame being denoised, and "extract the distribution and remove it" is where
the intuition breaks. This exact scheme was built and measured; it was not
abandoned as infeasible, it was found to add nothing over Noise2Noise.

- **It was built.** `burst_diffusion` is the scheme described: the training
  state is the average of m frames, the conditioning variable is m, the
  target is a fresh frame, so the network sees the whole curve of the noise
  receding with averaging; inference starts from one frame and walks the
  schedule. On single-frame input it lands at the plain Noise2Noise arm
  (35.42 dB, CD 3σ 0.539 px for both; report §9.3), and the 15-step walk
  keeps the blemishes at 0.61 versus 0.58 for one shot, with no change in
  any band (fine-feature report §4). The averaging curve adds nothing a pair
  of frames did not already teach.
- **Why it cannot add more.** Every arm already knows the noise distribution
  exactly — the frames are Poisson counts with a known rate, and the
  Noise2Noise loss is optimal for that model. What the distribution never
  gives is the noise *values* in frame 1: shot noise in frame 1 is
  independent of frames 2 to 16, so the burst teaches the variance at every
  averaging step, not the realization. The Q8 bound is precisely the
  "distribution known, feature known" ceiling.
- **Where the intuition is right.** The 16-frame averages carry the fine
  features, and a network trained bursts-in, single-frame-out learns what
  they look like; that prior can render a feature at full contrast once the
  frame gives evidence for it (Q10), but it cannot create detectability.
- **What would change the answer on real data.** All of the above assumes
  pure shot noise. Real frames also carry structured noise — fixed-pattern
  noise, line noise, charging streaks — which is deterministic or correlated
  across frames, so a burst reveals it and a single-frame model *can*
  subtract it. The synthetic corpus contains none of it; that capability is
  untested here and is the case where a single-frame model can genuinely
  gain.

---

## Q10. Would a real clean image improve things over the 16-frame average as the training target?

**Short answer:** no, and it was not what was meant. A clean target versus a
debiased 16-frame-mean target gave a median retention of 0.72 versus 0.70,
PSNR 36.24 versus 36.15 dB and signed CD bias +0.013 versus +0.024 px
(fine-feature report §8). The 16-frame mean is already an unbiased target
with a sixteenth of the noise; cleaning it further moves a conditional-mean
network by a few hundredths.

What was meant is a different *kind* of estimator, trained from the same
bursts:

- an MSE-trained network outputs the average of every scene consistent with
  the frame, so an uncertain feature comes out at a fraction of its contrast
  — the faded look, whatever the target;
- a network trained to output *one* scene consistent with the frame (a
  posterior sample, a MAP estimate, or an adversarially trained generator)
  renders the feature at full contrast whenever the evidence favors it. Its
  reference for full-contrast features can be the burst itself — the
  16-frame mean, or better its denoised version at 38–40 dB. No clean image
  is required.

Two costs come with that estimator: where the evidence is ambiguous it still
commits, so it renders phantoms at the rate the Q8 bound gives; and on edges
a committed position jitters between retakes by the posterior width, which
costs CD precision unless the committing term is applied only away from the
edges. **Decision:** judged over-engineered and a breakage risk for
production; not pursued.

---

## Q11. Would taking the log boost the fine-feature signal? Would a Fourier transform (amplitude and phase — random noise has random phase, fixed features have fixed phase) help? A combination?

**Short answer:** no, no, and no, for one shared reason: log, Fourier
transform, or both are invertible transforms of the same frame, and the Q8
likelihood ratio — hence the detection bound — is invariant to any such
transform. A network can also compute them internally; giving them as input
changes the representation, not the evidence. The fine-feature report tested
a linear re-representation directly (Sobel channels in, the `hybrid` arm):
identical to the plain arm.

- **Log.** A pointwise transform keeps every pixel's SNR: a stain of contrast
  Δ on a line of intensity λ has SNR Δ/σ before and (g′Δ)/(g′σ) after. What
  log changes is the loss weighting — it compresses bright values, so a dark
  stain on a bright line gets *less* weight in a squared error, not more.
  Practical problems on this data as well: a peak-10 frame has 11 gray levels
  and 11 % zeros on the dark lines, so log needs an offset; the principled
  form is the Anscombe transform, whose only real benefit is uniform noise
  variance, which a network trained on raw Poisson data already learns.
  Expected gain within 0.1 dB, zero on retention.
- **Fourier amplitude and phase.** "Noise has random phase, features have
  fixed phase" is true *across frames*, and that is exactly what averaging
  exploits — the signal phase is constant from frame to frame, the noise
  phase is not, and summing frames is the same operation in either domain.
  In a single frame you hold one draw of signal-plus-noise at every
  frequency, and nothing marks which part of the phase is which. With the
  signal spectrum known from the burst, the best single-frame filter in the
  Fourier domain is the Wiener filter, and the fine-feature report computed
  it: it transmits 0.001, 0.005 and 0.024 of the 1–2, 2–4 and 4–8 px bands;
  the networks beat that by 25× at 4–8 px because they use spatial structure,
  which a global frequency filter cannot. The Fourier view does help two
  other things: the periodic line pattern (already the easy part) and
  registration, where phase correlation is the standard tool.
- **Combination.** Two transforms that add no information do not add
  information together.

The fine features are the broadband, low-power deviations from the pattern;
they occupy exactly the frequencies where the frame's per-frequency SNR is
lowest.

---

## Q12. If the Fourier-phase idea has power in the multi-frame case, does it need a full average, or would two frames suffice?

**Short answer:** two frames help as two frames' worth of photons, and the
phase idea is not a separate lever on top of that. For shot noise the
registered sum of the frames is a sufficient statistic for the scene, so any
two-frame trick — cross-spectrum, phase agreement, "present in both, absent
in one" — can at best equal "register, add, and denoise the pair".

What two frames are worth, from the measurements already made:

| | one frame | two frames |
|---|---|---|
| ideal detector (Q8), strongest stains (single-frame SNR 1–2), detection at 5 % false alarm | 45 % | 61 % |
| ideal detector, median feature | 15 % | 21 % |
| fusion network CD 3σ per scene | 0.691 px | 0.489 px |
| fusion network blemish retention, median / strongest bin | 0.62 / 0.71 | 0.61 / 0.81 |
| PSNR | 35.8 dB | 36.6 dB |

A second frame buys about a quarter of the CD spread and lifts the strongest
stains; the median feature stays where it was. The blemishes visible in a
16-frame average need most of those frames: 8 frames detect the strongest
stains every time, and 16 frames reach 78 % on the median feature.

The one thing a second frame reveals that a sum does not is not the scene but
the noise: whatever is common to both frames and not part of the scene
(fixed-pattern noise, line noise) separates from whatever differs between
them (shot noise). The synthetic corpus has no such component; on a real
instrument that decomposition is what would let a single-frame model
subtract the structured part.

---

## Appendix A: the diagnostic probe (Part A)

`python tools/probe_burst_predictions.py` (defaults: BBBC038 checkpoint,
held-out source 12, 64 px center crop). Reference output at the 30k-step
baseline:

```
raw frame y1 vs clean:            17.31 dB
raw frame y1 vs another frame:    14.25 dB

A) network fed REAL m-frame averages (in-distribution, as in training)
    t   m   pred-vs-clean   pred-vs-a-real-fresh-frame
   15   1    36.15 dB       17.20 dB
   11   5    40.48 dB       17.23 dB
    7   9    41.77 dB       17.23 dB
    3  13    42.33 dB       17.24 dB
    1  15    42.82 dB       17.24 dB

B) network fed its OWN pseudo-averages (the iterative sampler trajectory)
    t   m   pred-vs-clean   state-vs-clean
   15   1    36.15 dB       23.15 dB
   11   5    36.18 dB       31.35 dB
    7   9    35.99 dB       33.87 dB
    3  13    35.98 dB       34.89 dB
    1  15    36.00 dB       35.18 dB

MSE(final prediction, actual fresh frame): 0.0191
single-frame noise variance (from its PSNR): 0.0186
```

Reading: **A**'s flat ~17.2 dB column = the prediction is far from every real
frame by exactly that frame's own noise — i.e. it is the clean estimate
(Q3). **A** improving down the rows = the posterior-mean family works across
levels. **B** flat at ~36 dB while **A** reaches 42.8 dB at the same $t$ =
the train/inference input gap (Q4). The final two lines = the loss-floor
signature (Q3).

*Caveat learned during the finetuning follow-up (Q6):* source 12 turned out
to be the one validation source with essentially **no headroom** — its
one-shot (36.1 dB) already saturates what a single frame supports there, so
its B column barely moves even after finetuning closes the dataset-wide gap
(validation-set mean B at $t=1$: 34.0 → 40.5 dB). Single-source probe runs
locate the mechanism, but judge gap-closing by validation-set means
(report §8), not by this one source.

## Appendix B: the detection bound (Part B)

`python tools/detection_bound.py --out runs/edge_denoise/detection_bound_val.json`
reproduces the Q8 table (dev scenes of `data/MIIC-burst-p10-drift`, whose
clean images and feature set are those of the fine-feature report; 400 draws
per hypothesis and feature, seed 0). The absent hypothesis replaces the
feature's pixels by the clean image with its 2–12 px band removed; the
present hypothesis is the clean image; both are Poisson-sampled on the
feature's bounding box with a 4 px margin at the stated dose, and the
detection rate is read at the 95th and 99th percentile of the absent
statistic. Per-feature values are in the JSON; the table reports medians per
SNR bin.
