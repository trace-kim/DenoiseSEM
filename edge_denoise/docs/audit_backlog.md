# Audit backlog — target-ladder report

*Opened 2026-09-04 from [`target_ladder_audit.md`](target_ladder_audit.md).
Triage: the audit's quantitative findings were all reproduced; the items
below are the ones judged valid but **not** on the critical path of the next
experiment (the λ_c sweep), plus one open discussion. Each entry says why it
matters and what "done" looks like, so it can be picked up cold. Remove an
entry when it is done; note the commit.*

What was done immediately (2026-09-04), for the record: the report's paired
analysis re-run against the matched control at scene level with 3σ units
(`python -m burst_diffusion paired`, new); the report corrected accordingly;
a compute-matched λ_c = 0 config (`miic_p10_dedup_ft_noisy_20k.yml`);
`repeatability.json` now records each edge arm's checkpoint path/hash/step and
the invocation; training refuses an occupied `run_dir` without `--overwrite`;
`provenance.json` records the warm-start checkpoint by hash.

## Open discussion (not a rejection)

### D1. How to read `ft_consist`'s gain before the compute-matched control runs

Audit item 7 says the 10k-step `ft_consist` vs `ft_noisy` comparison is
~160k vs ~80k network inputs and that the gain "cannot be attributed
entirely to the consistency term". My first triage rejected that *framing*
on the grounds that `sobloss` (30k steps from scratch, ~240k inputs, same
objective as the control) landed at 0.533 px, so "more compute" alone is
already an unlikely explanation. The user wants this kept open rather than
rejected: it is a challenge to the central result and the mechanism is not
fully understood yet. **To discuss after the do-nows**: what the second
forward pass does beyond compute (it is a second *sample of the data
stream* — a different replica of the same crop — so the gradient sees two
views per step; is that "compute", "data", or "the term"?); whether the
sobloss-from-scratch argument is sound given the different starting point;
and what result from the 20k-step `ft_noisy` arm would settle it (my
reading: near 0.406 → the term is not the cause; near 0.539 → closed).

*Discussion held 2026-09-04.* Mechanics: the second forward pass computes
$f(y_k)$, which enters only the agreement term; with λ_c = 0 its gradient is
identically zero, so a "two-view control" is `ft_noisy` itself. "Compute or
term?" therefore reduces to "is `ft_noisy` under-converged?". TensorBoard
says no: both fidelity terms sit at the Poisson noise floor the trainer
docstring predicts (image 0.1544 vs 0.154; Sobel 0.0283 vs 0.0289) from
the first fine-tune step in every arm, `ft_noisy`'s val fidelity loss is
flat to five decimals over 10k steps, and its val repeatability σ moves by
2·10⁻⁵ while `ft_consist`'s falls 0.0081 → 0.0073 with slightly *worse*
fidelity and PSNR — the signature of a changed optimum, not of better
optimization. **Pre-registered controls and predictions (before running):**
`ft_noisy_20k` (same objective, 2× steps) and `ft_noisy_b16` (same steps,
2× examples per step). Prediction for both: CD 3σ scene 0.53–0.55 px, PSNR
≥ 35.54 dB, val repeatability σ ≈ 0.0091. Falsifier: either at ≤ 0.45 px
with PSNR not below `ft_noisy`'s → the term is not the cause, rethink the
sweep.

*Outcome (2026-09-04, report §7).* Falsifier not met; band prediction wrong
on the CD median (controls 0.490 / 0.477 px, not 0.53–0.55), right on
fidelity, PSNR direction and pixel σ (controls −1–3%, `ft_consist` −26%
against them, 10/10 scenes). `ft_consist` beats both controls on 9/10
scenes (sign p = .02) but the paired t is p = .16 / .37 because of src 87
(one site, 7/10 CD measurements under `ft_consist`). Verdict: compute does
not explain the effect; the CD claim at matched compute is strong but not
established at one seed; the p = .006 headline was flattered by its
control. Sweep baseline is now `ft_noisy_b16` with ≥ 3 seeds. **Still open
for discussion**: whether the src 87 measurement failures under
`ft_consist` are a blur signature (check in the smoothness diagnostic), and
whether the user is satisfied that "term, not compute" is settled or wants
a further control (e.g. the agreement term computed with the second view
detached, which halves its backward cost).

## Deferred — deployment and evaluation

### B1. Clean-free CD site selection, crossing matching and shift reference

*Audit item 4.* `find_cd_sites`, `measure_site` (nearest-to-clean crossing)
and `estimate_shift` / the registration gate all use the clean image; only
pixel σ is truth-free as implemented. Fine for method development on the
synthetic corpus (identical sites for every arm). **Done when**: the harness
can take fixed/manual ROIs or a burst-mean reference instead of the clean
image, and the ft-ladder table has been re-run both ways to show the
precision columns agree. Needed before any real-data evaluation.

### B2. Enough evaluation replicas for `avg_of_8` / `avg_of_16` to have a σ

*Audit item 4.* With 16 replicas per source, `avg_of_8` has two independent
averages per scene (one dof per site) and `avg_of_16` one realization (no
σ). The "single-shot beats 8× dose" claim is 8/10 scenes, p ≈ .20.
**Done when**: the val (and later test) sources carry e.g. 64 synthetic
replicas so avg-of-8 has ≥ 8 independent realizations, and the classical
ladder is re-measured. Synthetic, cheap (`noising_pipeline`); do it before
the confirmatory test-split run.

### B3. Clipping bias of the stored noisy targets

*Audit item 5.* `noising_pipeline` clips Poisson samples to [0, 1] and
quantizes to 8 bits, so the "unbiased" fresh-frame / noisy-mean targets carry
a clipping bias (about −0.002 in aggregate, up to about −0.06 at the
brightest intensities). On this corpus oracle and noisy targets were
empirically indistinguishable, so it changed no conclusion. **User's note**:
this may behave differently on real data (different intensity distribution,
detector saturation), so it must not be forgotten. **Done when**: the
noising pipeline reports the expected clipping bias per dataset in
`stats.json`, and a real-data evaluation plan includes a check of target
bias at the bright end.

## Deferred — workflow defects

### B4. Resume with a changed config only warns

*Audit item 8.* `Trainer._restore` warns when the checkpoint's config
differs from the current one, then restores the model, optimizer, EMA and
RNG state anyway — so a same-shaped model can resume under a different
objective while being labelled with the new config. Not exercised by any
reported run, but a warning is easy to miss in a long log. **User's note**:
a warning here can lead to human error; this should become an error.
**Done when**: a config mismatch on resume raises unless an explicit
`--allow-config-change` is passed, with a regression test.

### B5. Noisy replicas are not part of the dataset fingerprint

*Audit item 8.* `provenance.dataset_fingerprint` hashes the clean sources
and the split, not the noisy replicas. The replicas are seeded and
regenerable, so this is a completeness gap, not a correctness one.
**Done when**: the fingerprint includes a digest over the noisy replicas (or
the generator's seed + parameters from `stats.json`).

### B6. Distillation manifest is not enforced and targets include the input replica

*Audit items 6 and 8.* `distill_targets.json` does not hash individual
targets or bind them to a dataset digest, and training does not check it;
each target averages all 16 teacher outputs, so the input replica is inside
its own target with weight 1/16. The distillation arm has been dropped from
the recommended line, so this is moot unless it is revisited. **Done when**
(if revisited): leave-one-out targets (or a disjoint target burst), a
per-target hash in the manifest, and a manifest check at training start.

### B7. Deterministic CUDA execution

*Audit item 8.* Two identical seed-0 runs differ at ~1e-5 in loss (cuDNN
nondeterminism). Rejected as an action — seed replication is the plan and
determinism costs throughput — but recorded so nobody re-audits it: if
bit-exact reruns are ever needed, use `torch.use_deterministic_algorithms`
and `CUBLAS_WORKSPACE_CONFIG`, and expect a slowdown.

### B8. Pure-gradient reconstruction is regularized, not exact

*Audit item "method-by-method".* `reconstruct_from_sobel` zeros spectral
modes below 1e-3 of the peak gain; the config comment says "exact inverse".
The docstring is correct; the config comment is loose. The arm lost in the
pilot. **Done when**: the config comment says "exact up to the null space's
neighbourhood", if the arm is ever revived.
