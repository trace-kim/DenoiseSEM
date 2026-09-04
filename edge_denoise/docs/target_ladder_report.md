# Gradient-Target Ladder & Consistency — Experiment Report

*2026-09-02, revised 2026-09-04 · code: [`edge_denoise/`](../) · method &
derivations: [`edge_denoise_method.md`](edge_denoise_method.md) · precedes:
pilot report [`edge_denoise_report.md`](edge_denoise_report.md) · results:
`runs/edge_denoise/repeatability_val_ft_ladder/`*

> **Revision note (2026-09-04).** The first version of this report was
> audited ([`target_ladder_audit.md`](target_ladder_audit.md)); every
> quantitative finding of the audit was reproduced and is folded in here.
> What changed: the paired analysis is now run against the declared matched
> control `ft_noisy` (not N2N/sobloss); the paired CD effects are stated as
> 3σ (the first version printed 1σ values under a 3σ label); the accuracy
> cost of `ft_consist` is reported as *not established* (the site-level test
> pseudoreplicated 37 sites in 10 scenes); "better gradient targets help
> modestly" is withdrawn (it was the fine-tune protocol, not the target);
> Appendix A.1 no longer calls the CD/shift metrics ground-truth-free as
> implemented. Audit items not acted on yet are tracked in
> [`audit_backlog.md`](audit_backlog.md). The training artifacts and the
> aggregate table were found correct and are unchanged.

**TL;DR**

- **The consistency term is the first statistically significant precision
  result in this project — and it holds against the matched control.**
  `ft_consist` (sobloss objective + λ_c = 1, the term implemented-but-OFF
  since the pilot) lowers per-scene CD 3σ by **−0.151 px vs `ft_noisy`**
  (9/10 scenes, t = −3.59, **p = .006**, surviving Bonferroni over the four
  fine-tune-vs-control comparisons), by −0.282 px vs the N2N teacher (10/10,
  p = .014) and by −0.211 px vs the from-scratch sobloss champion (10/10,
  p = .004). Headline: **CD 3σ scene 0.406 px** (`ft_noisy` 0.539, N2N
  0.539, sobloss 0.533); site-pooled 0.723 (−21% vs control); center σ 0.159
  (−13%); shift σ 0.122 (−13%); pixel σ 4.52·10⁻³ (−28%, 10/10 scenes,
  t = −15.9). Against the control the gain is spread over nine scenes
  (src 21 −0.48, src 48 −0.22, src 50 −0.22 px); on the hardest scene
  (src 87) `ft_noisy` had already closed the gap and `ft_consist` adds
  nothing (+0.004 px).
- **The price: a PSNR cost is established, an accuracy cost is not.** PSNR
  −0.39 dB vs `ft_noisy` (10/10 scenes, p = .005). Site-weighted |CD bias|
  grows +0.069 px (0.371 vs 0.302) and signed bias +0.017 px (+0.144 vs
  +0.127), but with the scene as the unit (n = 10) neither is significant
  (|bias| p = .22 vs `ft_noisy`, p = .18 vs N2N). The first version's
  "t = +2.31, significant" treated 37 sites as independent. Whether the
  bias–variance dial of method doc §6 has a real accuracy cost at λ_c = 1
  is what the seed-replicated λ_c sweep must decide; it is not decided here.
- **The two-phase distillation proposal: the decomposition's predictions
  hold where they are testable, but the arm does not beat the control.**
  `ft_distill` reduces pixel σ strongly (10/10 scenes, t = −9.5) —
  consistent with the hidden consistency half of
  E‖S f(y) − T‖² = Var(S f(y) | x) + ‖E S f(y) − T‖² — and its accuracy is
  indistinguishable from the teacher's (Δ|bias| +0.002 px/site, +0.007
  px/scene, n.s.; signed +0.135 vs +0.138). But on CD precision it is
  **+0.10 px worse than `ft_noisy`** in the mean (p = .52): nine scenes
  improve by 0.02–0.12 px and src 87 regresses by +1.45 px 3σ. The useful
  content of the proposal is the consistency component; `ft_consist`
  delivers a related (image-domain, not Sobel-domain) term directly and
  without freezing the teacher's outputs in.
- **Better gradient targets do not help precision — withdrawn.** Against
  the matched control the supervised oracle (`gradient_target: clean`) and
  the leave-one-out noisy average (`noisy_mean`) are *not* better on CD 3σ
  (+0.050 / +0.073 px, 6/10 scenes each, p = .47 / .39) and not better on
  |bias| (p = .65 / .28). They do buy PSNR (+0.15 / +0.06 dB, p = .01 /
  .02). The "9/10 scenes vs N2N" of the first version was the fine-tune
  protocol itself: `ft_noisy` alone is −0.13 px vs N2N (9/10). The
  objective, not the target, is where precision lives.
- **The fine-tune protocol is an adequate sweep vehicle, not a proven
  equivalent of training from scratch.** 10k steps warm-started from the
  N2N teacher land where the 30k-from-scratch sobloss champion lands, with
  no detectable difference at n = 10 (`ft_noisy` − sobloss: −0.06 px 3σ,
  6/10 scenes, p = .33, roughly −0.19…+0.07 px at 95%; PSNR +0.15 dB,
  p = .01). Because every sweep arm shares the protocol, within-sweep
  comparisons are valid regardless.
- **`ft_consist` is not compute-matched.** Its second forward pass carries
  gradients, so 10k steps see ~160k network inputs (~20 min) against
  `ft_noisy`'s ~80k (~11 min). The sobloss row (30k from scratch, ~240k
  inputs, 0.533 px) argues against "more compute" as the explanation, but a
  20k-step `ft_noisy` (`miic_p10_dedup_ft_noisy_20k.yml`) now exists as the
  sweep's matched-compute λ_c = 0 arm. How to read the gain before that arm
  runs is an open discussion item (backlog).

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
Sobel null space). `λ_g = 4` in every fine-tune arm. Target statistics, for
reading the ladder: the fresh frame carries full noise variance; `noisy_mean`
~1/15 of it (leave-one-out is load-bearing — including the input frame breaks
the N2N cross-term); `clean` is exact; the distill average has near-zero noise
but carries the teacher's systematic error whole (averaging removes jitter by
~√N and bias not at all). Two footnotes the first version omitted: the stored
noisy replicas are clipped to [0, 1] and quantized to 8 bits, so "unbiased"
holds up to a clipping bias of about −0.002 in aggregate (larger at the
brightest intensities; backlog); and `ft_consist`'s consistency term runs a
second batch-8 forward pass *with gradients* each step, so it is not
compute-matched to the other arms (TL;DR, last bullet).

Distillation targets: `python -m edge_denoise distill-targets`, teacher =
the same N2N checkpoint, 86 train+val sources, 16 replicas each, overlapping
64px tiles at stride 48 blended with a floored raised-cosine window
(identity round-trip exact; unit-tested), stored float32 `.npy` +
provenance manifest. The locked test split is refused by construction.

Evaluation: one `python -m edge_denoise repeatability` call holding all five
arms plus `sobloss` (the pilot's 30k-from-scratch champion) and the `n2n`
teacher — identical sources, seeds, crops, and 37 CD sites in 10 scenes.
Classical rows reproduce the pilot bit-identically, confirming comparability.
Paired statistics: `python -m burst_diffusion paired --control ft_noisy …`
(scene-level; outputs `paired_vs_{ft_noisy,n2n,sobloss}.md` next to the
JSON).

## 2. Results (val, 10 sources × 10 seeds, 37 sites)

| method | PSNR dB | pixel σ ×10⁻³ | CD 3σ **scene** px | CD 3σ site px | CD bias px | CD abs-bias px | CD success | center σ px | shift σ px |
|---|---|---|---|---|---|---|---|---|---|
| single_frame | 14.13 | 195.2 | 1.452 | 2.236 | +0.091 | 0.373 | 98.6% | 0.363 | 0.353 |
| avg_of_8 (K = 2/scene) | 23.12 | 68.9 | 0.550 | 1.558 | +0.080 | 0.197 | 98.6% | 0.255 | 0.131 |
| one_shot@n2n (teacher) | 35.42 | 6.50 | 0.539 | 1.039 | +0.138 | 0.319 | 99.2% | 0.200 | 0.145 |
| one_shot@sobloss (30k scratch) | 35.39 | 6.36 | 0.533 | 0.959 | +0.100 | 0.303 | 98.9% | 0.188 | 0.149 |
| one_shot@ft_noisy (control) | 35.54 | 6.28 | 0.539 | 0.917 | +0.127 | 0.302 | 98.9% | 0.184 | 0.141 |
| one_shot@ft_oracle | **35.68** | 6.27 | 0.514 | 0.956 | +0.123 | **0.298** | 98.9% | 0.188 | 0.139 |
| one_shot@ft_avg | 35.60 | 6.22 | 0.511 | 0.993 | +0.134 | 0.311 | 99.2% | 0.194 | 0.138 |
| one_shot@ft_distill | 35.48 | 6.01 | 0.484 | 1.011 | +0.135 | 0.321 | 98.9% | 0.197 | 0.135 |
| **one_shot@ft_consist** | 35.14 | **4.52** | **0.406** | **0.723** | +0.144 | 0.371 | 98.9% | **0.159** | **0.122** |

(Full table incl. avg_of_2/4/16 and the iterative burst rows:
`runs/edge_denoise/repeatability_val_ft_ladder/summary.md`; σ-maps:
`sigma_maps.png` alongside.) Read the classical rows with their replication
in mind: `avg_of_8` has only **two** independent averages per scene (one
degree of freedom per site), and `avg_of_16` has one realization and hence
no repeatability estimate at all. The scene-median column hides that the
oracle/avg/distill differences from the control are within scene-to-scene
noise (§3).

## 3. Paired analyses

All tests are **paired per scene** (n = 10; the scene is the independent
unit because its sites share frames, registration and model output), two-
sided Student t on `arm − control`; CD values are **3σ** — the JSON's
`scene_sigmas_px` is 1σ and the first version of this report printed those
1σ values under a 3σ label. Negative = lower. With ten scenes the test is
weak: p > .05 is not evidence of equality.

**Primary comparison — every arm vs the matched control `ft_noisy`:**

| arm | CD 3σ scene: Δ px / lower / t / p | \|CD bias\| scene: Δ px / p | pixel σ: Δ ×10⁻³ / lower / p | PSNR: Δ dB / p |
|---|---|---|---|---|
| ft_oracle | +0.050 / 6/10 / +0.75 / .47 | +0.010 / .65 | −0.01 / 7/10 / .85 | **+0.146 / .011** |
| ft_avg | +0.073 / 6/10 / +0.90 / .39 | +0.023 / .28 | −0.06 / 8/10 / .013 | **+0.064 / .018** |
| ft_distill | +0.101 / 9/10 / +0.67 / .52 | +0.059 / .31 | **−0.27 / 10/10 / <.001** | −0.060 / .076 |
| **ft_consist** | **−0.151 / 9/10 / −3.59 / .006** | +0.155 / .22 | **−1.76 / 10/10 / <.001** | **−0.393 / .005** |
| sobloss (30k scratch) | +0.060 / 4/10 / +1.02 / .33 | +0.011 / .56 | +0.08 / 4/10 / .35 | −0.150 / .012 |
| n2n (teacher) | +0.131 / 1/10 / +1.32 / .22 | +0.051 / .31 | +0.22 / 0/10 / .002 | −0.116 / .009 |

Bonferroni over the four fine-tune arms: α = .0125; `ft_consist`'s CD result
survives it. `ft_distill`'s mean hides structure: nine scenes improve by
0.02–0.12 px and src 87 (the pilot's worst scene) regresses by +1.45 px.
`ft_consist` improves nine scenes (src 21 −0.48, src 48 −0.22, src 50 −0.22,
src 80 −0.14) and leaves src 87 unchanged (+0.004): the control had already
brought src 87 from N2N's level down, so the "largest gains on the hardest
scene" reading of the first version was the fine-tune protocol, not the
consistency term.

**Secondary comparisons (CD 3σ scene; the references of the first version):**

| arm | vs n2n: Δ px / lower / t / p | vs sobloss: Δ px / lower / t / p |
|---|---|---|
| ft_noisy | −0.131 / 9/10 / −1.32 / .22 | −0.060 / 6/10 / −1.02 / .33 |
| ft_oracle | −0.080 / 9/10 / −2.34 / .044 | −0.009 / 6/10 / −0.70 / .50 |
| ft_avg | −0.058 / 9/10 / −2.70 / .024 | +0.014 / 6/10 / +0.46 / .65 |
| ft_distill | −0.030 / 9/10 / −0.56 / .59 | +0.041 / 9/10 / +0.44 / .67 |
| **ft_consist** | **−0.282 / 10/10 / −3.04 / .014** | **−0.211 / 10/10 / −3.91 / .004** |

Per scene vs N2N, `ft_consist` is −1.01 px on src 87, −0.54 on src 21, −0.30
on src 50 — but the src 87 share belongs to the protocol (above).

**Accuracy (|CD bias|), the audit's correction.** The first version tested
per-site |bias| over 37 sites (`ft_consist` vs N2N: +0.052 px, t = +2.31,
"significant"). Sites within a scene are not independent; at the scene
level the same comparison is +0.104 px, t = +1.44, **p = .18**, and vs
`ft_noisy` +0.155 px, **p = .22**. Descriptively the consistency arm is the
least accurate row of the table (site-weighted 0.371 vs 0.302); as a
statistical claim the accuracy cost is *not established* with one seed and
ten scenes. The same discipline applies in the other direction: `ft_distill`
"inherits the teacher's accuracy" means *no detectable difference*
(Δ|bias| +0.002 px/site, +0.007 px/scene), not equality — no equivalence
margin was pre-registered.

## 4. Reading against the pre-registered assessment

The 2026-09-02 assessment of the two-phase proposal made testable claims:

1. *"The averaged-denoised target inherits the teacher's bias; unbiased
   targets can do better"* — **half confirmed.** `ft_distill`'s accuracy is
   indistinguishable from the teacher's (Δ|bias| +0.002 px/site, signed
   +0.135 vs +0.138). But the unbiased-target arms do *not* measurably beat
   the control on |bias| either (oracle +0.010, avg +0.023 px/scene, n.s.),
   so "can do better" is not shown at this scale.
2. *"With frozen targets the loss decomposes into consistency + anchor, so
   the useful content is the consistency component"* — **consistent with
   the data, not proven by it.** `ft_distill` shows the strongest pixel-σ
   reduction after `ft_consist` (10/10 scenes, t = −9.5), as the hidden
   consistency half predicts, but that did not translate into CD precision
   (+0.10 px vs control, src 87 regression). The first version's reading
   that `ft_distill` "out-precisions the oracle" (0.484 vs 0.514 scene
   median) is within noise — neither arm differs from the control — and the
   budget-split story built on it is withdrawn.
3. *"The oracle is the ceiling for pseudo-label fidelity; run it as the
   gate"* — the gate **does not pass on precision** against the matched
   control (+0.050 px, p = .47); it passes on PSNR (+0.15 dB, p = .011) and
   on nothing metrological. The error-shaping ceiling of method doc §4.2 is
   real.
4. *"λ_c is the theoretically cleaner sibling and should win"* —
   **confirmed**: the only arm that beats the control on every precision
   column, the first p < .05 precision effect in the project (p = .006 vs
   the control), predicted qualitatively by §6 ("the only term that changes
   the optimum"); the magnitude, −28% per-scene CD 3σ at λ_c = 1, was not
   predicted. Two caveats stand: compute is not matched (TL;DR), and the
   accuracy cost is unmeasured at seed-replication scale (§3).
5. *Memorization guard* — all gains are on held-out content groups; the
   consistency win cannot be target memorization (no per-scene target
   exists for it). `ft_distill`'s src 87 regression is the one place the
   frozen-target arm looks unstable.
6. *"Fine-tuning from the teacher makes sweeps 3–6× cheaper at the same
   answer"* — **adequate, not equivalent**: `ft_noisy` is within noise of
   sobloss at 10k steps vs 30k (§3); an equivalence claim would need a
   pre-registered margin and more scenes.

## 5. Verdict and recommended next steps

**The two-phase pseudo-label pipeline is feasible and behaves as analyzed,
and the experiment it motivated found something better than the pipeline
itself.** The distillation target adds machinery (a teacher pass over the
corpus, frozen targets, a staleness/bias liability, a 1/16 self-inclusion of
the input replica in its own target — Appendix A) to deliver a weaker
version of what `lambda_consistency` does in one config line. The
recommended line drops phase-2 distillation and promotes the consistency
term. The unbiased gradient targets are retained as a PSNR/accuracy option,
no longer as a precision lever.

1. **λ_c sweep × seed replication** (the decisive study): λ_c ∈ {0.25, 0.5,
   1, 2} × ≥3 seeds on the fine-tune protocol (~20 min/run), **plus the
   matched-compute λ_c = 0 arm** (`miic_p10_dedup_ft_noisy_20k.yml`, ~22
   min) so the consistency term is separated from its extra forward pass.
   Analyze with `python -m burst_diffusion paired --control <λ_c = 0 arm>`
   — scene-level, 3σ, all four metrics — and read CD 3σ, |bias|, PSNR and
   pixel σ together. Whether the bias–variance dial has a measurable
   accuracy cost, and where its knee is, is what this study decides; the
   present data show a PSNR cost and a descriptive bias increase only.
2. **Gradient-domain consistency** (penalize `‖S f(y₁) − S f(y₂)‖²`): the σ
   maps concentrate variance on edges, so aiming the variance penalty there
   may buy the same CD improvement at lower PSNR/bias cost. Small code
   change; natural follow-up to #1.
3. Re-run the smoothness diagnostic (`tools/diagnose_smoothness.py`) on the
   λ_c winner before freezing anything — the consistency term is the arm
   most exposed to the blur-cheat failure mode, and PSNR (−0.39 dB) is too
   forgiving to rule it out alone.
4. *(downgraded)* **λ_c combined with an unbiased gradient target.** The
   first version called the two effects complementary; against the matched
   control the target arms show no precision or accuracy gain, so there is
   no ladder evidence for additivity. At most one exploratory cell after #1,
   on the hypothesis that an oracle gradient target counteracts whatever
   bias the consistency term induces.
5. Only after a winner exists at seed-replication scale: **one confirmatory
   run on the locked test split** — preceded by generating enough noisy
   replicas on the evaluation sources that the `avg_of_8` / `avg_of_16`
   rows carry real repeatability estimates (backlog), so the
   "single-shot beats long averaging" claim can be tested rather than
   suggested.

## 6. Caveats

One training seed per arm; 10 dev scenes; 64px synthetic-Poisson patches;
λ_c = 1 unswept; `ft_consist` is not compute-matched to the control; its
accuracy cost is descriptive only (scene-level p ≈ .2) and must be
re-measured with seeds at every λ_c; `avg_of_8` rests on two averages per
scene and `avg_of_16` on one; the harness selects CD sites, matches
crossings and references shift against the clean image (Appendix A.1), so
its CD/shift columns are not ground-truth-free as implemented; slope/LER
transfer not re-diagnosed for these checkpoints (next-step #3); all
development on the val split — the test split remains untouched. The
fine-tune arms saw 40k effective steps of data (30k teacher + 10k) vs
sobloss's 30k, which may contribute to `ft_noisy`'s small PSNR edge.

Reproduce: configs `edge_denoise/configs/miic_p10_dedup_ft_*.yml`; teacher
copy + distill targets under `runs/edge_denoise/ft_ladder/`; the harness
command is in each config header (the ladder's `repeatability.log` holds
only the harness's progress output — evaluations now record their argv and
each edge arm's checkpoint hash in `repeatability.json`, but this one
predates that). Provenance notes from the audit: the `ft_consist` run was
executed twice into the same directory on 2026-09-02; the second run
overwrote the first's checkpoint and provenance and the evaluation used the
second (timestamps match). The two runs agree to ~1e-5 in loss (cuDNN
nondeterminism); training now refuses an occupied `run_dir` without
`--overwrite`.

## Appendix A — exact objectives per arm, phase by phase

Notation: clean source $x$; noisy replicas $y_1,\dots,y_{16} =
\mathrm{Poisson}(x\,p)/p$ at effective peak $p = 10$, independent per frame
given $x$ (stored clipped to $[0,1]$ and 8-bit quantized — see §1); every
tensor in one training sample is the **same random 64px crop** of one
scene, in model range $[-1, 1]$; $S$ = the 1/8-normalized Sobel pair (2
channels, reflect padding, §4.1 of the method doc); $d(a, b) =
\mathrm{mean}\,(a - b)^2$ (mean-reduced L2); $f = f_\theta$ is the network
being trained. Per sample, a fresh permutation of the 16 replicas supplies
the input replica $i$, the fidelity-target replica $j$, and the agreement
replica $k$ — pairwise distinct. Batch 8, Adam $2\cdot10^{-4}$, grad-clip
1.0, EMA 0.999, in every phase.

Weights are written symbolically. The values used are

$$\lambda_{\mathrm{image}} = 1,\qquad \lambda_{\mathrm{gradient}} = 4,\qquad
\lambda_{\mathrm{consistency}} = 1\ (\text{where the term appears})$$

in **sobloss, hybrid and every `ft_*` arm**; n2n has no gradient term
($\lambda_{\mathrm{gradient}} = 0$) and the pure-gradient pilot arm uses
$\lambda_{\mathrm{image}} = 0,\ \lambda_{\mathrm{gradient}} = 1$. They are
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
  Sobel is identical to matching the averaged Sobel maps. *Caveat (audit
  item 6):* the sum runs over **all 16** replicas, including the training
  input $y_i$, so the target contains $f_1(y_i)$ with weight 1/16 and the
  population decomposition's independence assumption holds only
  approximately; a leave-one-out target ($\tfrac1{15}\sum_{m\neq i}$) or a
  disjoint target burst would make it exact (backlog — moot unless the
  distillation arm is revisited).
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
  the Sobel-domain variant $d\big(S f(y_i),\,S f(y_k)\big)$ is next-step #2
  and has not been run. Calling it "the consistency half of the
  distillation loss" (TL;DR) is therefore an analogy — the distillation
  decomposition constrains $\mathrm{Var}(S f(y)\mid x)$, this term
  constrains $\mathrm{Var}(f(y)\mid x)$ — not an identity.

**hybrid / grad** — pilot arms, single phase (30k from scratch): hybrid is
sobloss's $\mathcal L$ with network input $[y_i,\,S y_i]$; grad takes input
$S y_i$, outputs a Sobel field $\hat g$, trains
$\mathcal L = \lambda_{\mathrm{gradient}}\, d(\hat g,\,S y_j)$ with
$\lambda_{\mathrm{image}} = 0$, and inverts by regularized FFT least squares
at inference (spectral modes below $10^{-3}$ of the peak gain are zeroed, so
"exact inverse" holds up to the operator's null space and its immediate
neighbourhood).

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
  passes and is not obviously better at fixed compute. The second forward
  pass is what makes this arm cost ~2× the control per step.
- Only the **gradient-term reference** differs across
  ft_noisy/oracle/avg/distill; the fidelity term, data stream, and recipe
  are shared bit-for-bit. Replica-distinctness ($i \neq j \neq k$) is what
  keeps every term's zero-cross-term argument valid (§4.3 / §6 of the
  method doc).

### A.1 Evaluation without ground truth (added 2026-09-03, corrected 2026-09-04)

The user's deployment framing — no golden truth on real instruments; even
long averages carry drift and charging; the goal is single-shot precision
*better than long-time measurements* — assigns different statuses to the two
halves of the table, and the first version of this section overstated one
of them.

**Precision columns are truth-free as metrics, but not as implemented.**
CD/center/shift σ across retakes are defined without a reference and would
transfer to real data as concepts. The harness, however, uses the clean
image in three places: CD sites and their expected edge positions are
selected on it (`find_cd_sites`), each predicted crossing is the one nearest
the clean crossing (`measure_site`), and the shift and the registration
inclusion gate reference it (`estimate_shift`). Only pixel repeatability is
truth-free as implemented. Deployment needs fixed or manual ROIs (or a
burst-mean reference) followed by revalidation — tracked in the backlog; for
method development on the synthetic corpus the clean-selected sites are the
right choice, because they hold every arm to identical sites.

**"Better than long-time averaging" is suggestive, not established.**
`ft_consist` single-shot is 0.406 px vs `avg_of_8`'s 0.550 at 8× dose in
the scene medians, but paired per scene it is better on 8/10 with p ≈ .20 —
and `avg_of_8` has only two independent averages per scene, so its σ is
itself barely estimated; `avg_of_16` has one realization and no σ at all.
More evaluation replicas are needed before this claim carries weight
(next-step #5).

**Bias columns exist only in synthetic-land** (and even here "clean" is a
real capture with residual grain, report §6 of the pilot) — they should be
read as diagnostics of *systematic-error stability* (pattern-dependent bias
does not calibrate out; a constant offset does), and they are the last
cheap place to catch a blur-cheat. Consequences for the next steps: the λ_c
sweep optimizes precision margin over the classical ladder subject to a
bias-stability bound; the consistency term is the deployment-preferred
objective (needs neither truth nor averaging) with one new requirement — on
real drifting bursts the pair (and the N2N target) must be registered
first, or pixel-wise agreement punishes the drift itself and pushes toward
blur; the smoothness diagnostic on the λ_c winner is mandatory, not
optional.
