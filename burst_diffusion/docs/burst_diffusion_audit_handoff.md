# Burst-Diffusion Experiment Audit Handoff

**Audit date:** 2026-08-30  
**Scope:** Recent burst-diffusion workflow, experiments, artifacts, and paper-facing claims in this repository. DDIM is explicitly out of scope.  
**Audit posture:** Read-only inspection and diagnostic reruns. No existing model, pipeline, configuration, dataset, or result artifact was intentionally changed. Temporary diagnostic outputs may exist under `tmp/audit_*`; this handoff is the only intended source-tree/document addition.

## Executive conclusion

The current experiment record is **not paper-ready as confirmatory evidence**. The largest problem is experimental provenance: the corrected MIIC “locked test” reuses the exact permutation tail that served as validation during earlier method development. Several other claims overreach the evidence, especially unseen-layout generalization, diffusion-specific benefit, CD equivalence, registration superiority, and statistical significance at the algorithm level.

This does **not** mean the central denoising effect is spurious. Read-only reruns on hundreds of previously unused sources, together with tiled full-image diagnostics, continued to show a large advantage over 16-frame averaging under the repository's matched synthetic peak-10 Poisson setting. The aggregate advantage was roughly 9–11 dB, but the BBBC mean is strongly influenced by low-complexity images; on its 84/574 higher-complexity sources (`clean std ≥ 0.08`), the median advantage was about 4.38 dB. The defensible conclusion is therefore narrower:

> Under matched synthetic peak-10 Poisson corruption, the fixed t-conditioned Noise2Noise U-Net checkpoints substantially outperform 16-frame averaging in PSNR. For the rollout-finetuned checkpoint, recurrent prediction exceeds its one-shot output; its final recurrent prediction also exceeds that of the saved plain-40k checkpoint, conditional on this single training seed.

The current record does not yet establish a diffusion-specific advantage, real-SEM effectiveness, acquisition/layout-independent generalization, metrology equivalence or superiority, or reproducibility across training runs.

## Experiment history reconstructed from artifacts

- **2026-08-29:** Peak-10 Poisson burst datasets were generated, followed by 30k-step BBBC038 and original MIIC baseline training.
- **2026-08-30 01:17–02:36:** Each baseline was continued to 40k steps with 50% EMA self-rollout.
- **07:24–07:50:** Plain 40k step-matched controls were trained.
- **09:06:** Repeatability, CD, and registration artifacts were produced.
- **14:50:** Exact-content leakage in the original MIIC dataset was fixed after duplicate content was discovered.
- **14:45–16:09:** The corrected/deduplicated MIIC baseline, rollout, and control lineages were generated and trained.
- **16:11–16:12:** Corrected development/test accuracy and repeatability evaluations were run.
- **16:16:** Provenance was captured.
- **16:17:** The report commit was recorded.

The corrected baseline appears to have started before the fix commit, and provenance was captured only after training. That timing does not prove the wrong code ran, but the present artifact scheme cannot establish the exact execution-time source state.

## Pipeline understood and audited

1. Clean source images are selected, normalized, and prescaled to approximately `[0.15, 0.85]`.
2. Sixteen aligned synthetic Poisson observations are generated per source at peak 10 and clipped to `[0, 1]`.
3. Sources are split and cached. The corrected MIIC recipe contains 76 train, 10 development, and 10 test sources; BBBC contains 86 train and 10 validation sources, with no locked test.
4. An approximately 8.95M-parameter, timestep-conditioned U-Net is trained on aligned random 64×64 crops. Its input is a partial-frame average and its target is a fresh noisy frame excluded from that average.
5. Baseline training runs to 30k steps. It is then either resumed to 40k with 50% EMA-generated rollout inputs or continued as a plain step-matched control.
6. Evaluation compares single frame, average-of-16, one-shot model output, recurrent averaged state, and recurrent prediction. The main saved evaluation uses a deterministic center 64×64 crop.
7. Repeatability evaluates pixel variance, clean-assisted CD, and clean-referenced registration across repeated reconstructions.

The core mechanics were internally consistent in the paths reviewed:

- The fresh target is excluded from the selected averaging subset: `burst_diffusion/data.py:438-449`.
- Frame-count mapping and the weighted sampler update are consistent: `burst_diffusion/schedule.py:34-73`.
- Rollout states follow the same update rule: `burst_diffusion/rollout.py:31-81`.
- No schedule-indexing, target-contamination, or obvious sampler off-by-one error was found.
- Relevant targeted tests produced no failures. Several overlapping selections were run, including a focused burst selection reporting 138 passed and 113 deselected. Recover the exact shell commands before citing test counts; the overlapping counts must not be summed.

The dominant risks are experimental design, inference, and reproducibility rather than an obvious implementation defect.

## Findings that block the current paper framing

### P0 — The corrected MIIC split reuses historical validation positions and was not prospectively sequestered

The original MIIC validation artifact and corrected MIIC test artifact use the identical source-index list:

```text
[12, 15, 16, 24, 29, 31, 37, 62, 72, 88]
```

Evidence:

- `runs/burst_diffusion/miic_p10/eval/results.json:148-160`
- `runs/burst_diffusion/miic_p10_dedup/eval_test/results.json:148-160`
- Split implementation: `burst_diffusion/data.py:232-256`
- Original config: `burst_diffusion/configs/miic_p10.yml:9-14`
- Corrected config: `burst_diffusion/configs/miic_p10_dedup.yml:16-25`

The behavior is structural. With `test_fraction=0`, validation consumes the permutation tail. With the same split seed and a newly enabled test fraction, test consumes that same tail. Consequently, the corrected test occupies the old validation positions in the seeded permutation. Regeneration changed some source-to-index mappings, so it is not wholly identical by content, but it is not a pristine confirmatory holdout.

Decoded-pixel and artifact hashing found:

- 8/10 corrected-test contents occurred somewhere in the old MIIC experimental corpus.
- 3/10 were contents previously used in validation.
- 7/10 occurred in the prior training corpus; counts overlap because the old corpus duplicated content.
- Corrected test indices 29 and 31 have byte-identical clean images **and all 16 byte-identical noisy frames** in the old validation artifacts.
- Corrected test index 37 has byte-identical clean content to old validation index 16, with newly generated noise.
- Only corrected test contents 16 and 24 were absent from the entire old generated corpus.

Important nuance: the corrected model's own train split is exact-content-disjoint from its test split. The defect is at the research-process level: method choices and claims were developed after exposure to many of these contents, including three prior validation contents. Therefore the following language in `burst_diffusion_report.md:539-555` and `:592-595` is not supportable:

- “locked test”;
- “test evaluated exactly once”;
- “development and all method work used dev only”;
- “genuinely unseen” in a confirmatory sense.

This is a provenance failure, not evidence that the effect disappears. After excluding the three prior-validation contents, the remaining seven still showed approximately +9.70 dB for baseline one-shot versus average-of-16 and +0.805 dB for rollout iteration versus its one-shot output. This is only a retrospective robustness check: 5/7 of those remaining contents had still occurred in old training, so it is not an independent-test subset.

### P0 — Exact-content hashing does not establish layout-independent generalization

The corrected split defines exact decoded full-image hashes at `burst_diffusion/data.py:37-47` and groups them at `:207-215`. Training uses random aligned 64×64 crops (`burst_diffusion/data.py:430-449`), whereas saved evaluation uses one center 64×64 crop (`burst_diffusion/evaluate.py:94-123`). Exact full-image hashing therefore misses nearly identical evaluated regions and recurring IC templates.

Nearest corrected test-to-train clean center crops included:

| Test → train | Crop PSNR | Crop correlation |
|---|---:|---:|
| 16 → 53 | 34.49 dB | 0.973 |
| 24 → 68 | 35.05 dB | 0.992 |
| 88 → 61 | 34.32 dB | 0.991 |

Test contents 16 and 24 were the only two absent from the old corpus, yet both have close template matches in the corrected training split. The first two full images also have high global similarity, which is consistent with a recurring layout/template but does not prove shared acquisition or layout identity without provenance metadata. Removing all three near-neighbor scenes reduces rollout iteration's advantage over average-of-16 from 9.59 to 8.91 dB, so the effect remains large, but unseen-layout generalization is not established.

Required fix: group sources using acquisition/specimen/device/wafer/layout metadata where available, supplemented by perceptual and evaluated-crop similarity checks. A SHA-only split is insufficient for a repetitive IC corpus.

### P0 — BBBC has no locked test

`burst_diffusion/configs/bbbc038_p10.yml:10-15` defines validation but no test split. The same ten validation sources were available for baseline monitoring, rollout decisions, controls, and metrology. BBBC therefore cannot serve as an independent confirmatory dataset in the current two-dataset headline.

A new test must not be created merely by adding `test_fraction` while retaining the same seed: that would repeat the MIIC permutation-tail mistake and turn historical validation into “test.” The audit has now inspected all 574 previously unselected local BBBC sources, so there is no longer a pristine local pool. Freeze that just-inspected audit as retrospective evidence and obtain new external sources for prospective confirmation.

### P0 — The baselines do not isolate a diffusion-specific contribution

The one-shot method is a timestep-conditioned Noise2Noise forward pass, as the method document explains at `burst_diffusion_method.md:235-246`. The current comparison is dominated by learned-prior denoising versus raw temporal averaging.

Missing controls include:

- an equal-capacity single-level Noise2Noise model;
- a clean-supervised oracle using the same architecture;
- an Anscombe-plus-BM3D or comparably strong classical Poisson denoiser;
- a competitive learned Poisson denoiser;
- schedule/timestep ablations;
- FLOP-, latency-, or wall-time-matched recurrent controls.

A validation-tuned Gaussian check on corrected MIIC reached 27.48 dB on the saved test versus 26.06 dB for average-of-16. The learned model remained far ahead, but this confirms that averaging is a weak PSNR baseline. The present evidence does not show that diffusion-like multilevel conditioning is necessary or better than a matched denoiser.

### P0 — One training seed and one split do not support method-level p-values

The report acknowledges one training seed and one split seed at `burst_diffusion_report.md:147-151`. Source-level paired tests quantify variability across images **conditional on fixed checkpoints**. They do not quantify the variability of retraining the methods, nor can they establish that rollout reliably beats the plain continuation across initializations and splits.

Torch and CUDA RNGs are seeded (`burst_diffusion/train.py:102-104`), and RNG state is saved, but deterministic CUDA algorithms/cuDNN behavior are not enforced. Exact GPU reproducibility is not established.

For the paper, use at least 3–5 independent training seeds per arm and preferably multiple acquisition/layout-grouped splits. Treat training run and source as separate uncertainty levels. Do not interpret image-level p-values as algorithm-replication p-values.

## Statistical and metrology problems

### Non-significance was incorrectly interpreted as equivalence

The report calls rollout CD repeatability “equivalent” to average-of-8 because the paired test gives `p≈0.51` (`burst_diffusion_report.md:598-605`). Failure to reject a difference is not an equivalence result. No practical equivalence margin or TOST was prespecified.

Reanalysis of saved per-scene CD 3σ values gave:

| Contrast, model minus comparator | Mean difference | 95% CI | Paired t p | Wilcoxon p | Model wins |
|---|---:|---:|---:|---:|---:|
| Rollout iteration − avg8 | −0.100 px | [−0.428, +0.228] | 0.508 | 0.625 | 6/10 |
| Rollout iteration − avg4 | −0.521 px | [−1.270, +0.228] | 0.150 | 0.064 | 8/10 |
| Rollout iteration − one-shot | −0.138 px | [−0.310, +0.034] | 0.102 | 0.064 | 9/10 |

Therefore “equivalent to avg8,” “clearly better than avg4,” and “precision now properly significant” are not supported by the saved test-split CD data. Only the reported PSNR improvement of rollout iteration over one-shot has strong conditional evidence in this set.

Average-of-8 also has only `K=2` independent groups per source, while model rows have `K=10`. The c4 correction addresses expected sample-standard-deviation bias under normality; it does not remove the very large uncertainty of a two-realization variance estimate or establish Gaussianity.

### The registration claim changes under source-level analysis

The report says registration “reverses in the model's favor,” comparing pooled shift sigma 0.106 for rollout iteration against 0.166 for average-of-8 (`burst_diffusion_report.md:607-608`). The saved artifact also contains scene-level summaries:

- Average-of-8 scene-median sigma: 0.0508; pooled sigma: 0.1658.
- Rollout iteration scene-median sigma: 0.0951; pooled sigma: 0.1062.

Reconstructed per-source comparison gave 4/10 wins for the model, a mean difference of −0.020 px, 95% CI `[−0.118, +0.078]`, and paired `p=0.653`. The scene median descriptively favors average-of-8, but the source-level result is inconclusive rather than evidence of average-of-8 superiority. Pooled and scene-level summaries estimate different quantities; heterogeneous, highly uncertain `K=2` source estimates make neither superiority claim reliable. Registration should be analyzed at the source level with more realizations.

### Metrology is oracle-assisted, not a blind deployable pipeline

The repeatability code selects and validates sites on clean images (`burst_diffusion/repeatability.py:241-319`), chooses noisy crossings nearest the clean edge (`:203-238`), and registers against the clean reference (`:428-443`). Failed edge measurements can be excluded (`:447-480`). Corrected-test success ranges from 96.2% to 100%; rollout iteration is 98.0% and average-of-8 is 98.8% (`runs/burst_diffusion/miic_p10_dedup/repeatability_test/summary.md:16-29`). Censoring is limited but slightly method-dependent and cannot be dismissed outright. These remain simulation-oracle metrics and should be supplemented with a blinded/automatic workflow before making operational SEM metrology claims.

Pixel and CD repeatability can also improve through stable smoothing or hallucination. Repeatability alone is not accuracy.

## Synthetic-noise and domain-validity concerns

### Clipping violates the simple unbiased-noise premise locally

The clean images are prescaled to `[0.15, 0.85]`, but peak-10 Poisson samples are clipped to `[0, 1]`. At clean intensity near 0.85, the theoretical conditional clipping bias is about −0.059. The learned target is therefore the clipped conditional mean, not exactly the pristine clean intensity.

For BBBC, `data/BBBC038-burst-p10/stats.json:3-5` records:

- median absolute image bias: approximately 0.000366;
- aggregate signed mean bias: approximately −0.003114;
- 12/96 sources with absolute image-mean bias above 0.01;
- worst source bias approximately −0.0301.

An intensity-conditioned audit found 12.3% of BBBC pixels in bins with absolute conditional bias above 0.01 and 8.9% above 0.02. The generator warns on the median statistic (`burst_diffusion/generate.py:284-290`, `:322-327`), which hides biased bright structures and minority images. The report's “0.0004” and “≤0.0024 on both datasets” language (`burst_diffusion_report.md:63-67`, `:142-147`) understates this issue.

### The experiment is synthetic and narrow

The evidence covers one extreme peak-10 Poisson condition, one burst length, one patch size, perfectly aligned images, and no real bursts. Stored noisy images have only 11 possible intensity levels; approximately 2.56% of corrected MIIC pixels and 3.09% of BBBC pixels saturate at 255.

Not exercised:

- read noise or Poisson-Gaussian mixtures;
- fixed-pattern or correlated detector noise;
- drift, scan distortion, charging, contamination, dose variation, or beam damage;
- multiple peak/noise levels and burst lengths;
- real SEM acquisitions;
- physical-unit CD calibration.

The current work should be framed as a synthetic proof of concept until those conditions are tested.

## Overfitting assessment

No obvious optimizer overfitting was visible in the corrected MIIC learning trace; validation PSNR continued to improve through the fixed final checkpoint. However, the saved baseline checkpoint showed a train/holdout gap on center crops:

- MIIC train one-shot mean: 36.479 dB.
- MIIC validation+test one-shot mean: 35.055 dB.
- Gain over average-of-16: 10.315 dB on train versus 8.987 dB on holdout.
- Gap in gain: 1.328 dB, approximate 95% CI `[0.485, 2.170]`.

This is consistent with scene memorization, near-template reuse, or split difficulty, but is not by itself proof of damaging model overfit. The broad unused-source diagnostics below materially reduce concern about ordinary exact-source memorization. Acquisition/layout-level validation and independent training seeds are still required.

## Reproducibility and artifact weaknesses

- `burst_diffusion/provenance.py:95-119` fingerprints sorted clean-content hashes only. Provenance separately stores split index lists, but the digest does not cryptographically bind each index to clean/noisy content, manifest order, generation settings, or `sources.json`. Reordering can preserve the fingerprint while changing deterministic noise and split assignment.
- Evaluation checks only `num_steps` compatibility (`burst_diffusion/evaluate.py:194-201`), so a shape-compatible but incorrect config/dataset can yield plausible output without rejection.
- Final corrected provenance was captured after training from a dirty tree and stores a diff hash, not a reconstructable patch: `runs/burst_diffusion/miic_p10_dedup/provenance.json:187-210`.
- Environment capture includes Python/platform/Torch/CUDA but not a full dependency lock.
- Evaluation and repeatability artifacts do not consistently record command lines, checkpoint hashes, source manifests, git state, and analysis environment.
- Reported p-values and intervals live in Markdown rather than being generated by a tracked statistical analysis script.
- Runs, checkpoints, logs, images, and result bundles are ignored and therefore not reconstructable from the repository alone.
- The report states the upstream MIIC DOI and CC BY-NC 4.0 license (`burst_diffusion_report.md:60-63`), but preparation of the local 1,050-file derivative is not reproducibly captured. It contains only 185 unique decoded contents. The transformation history and compliance of any redistributed derivative/artifact bundle need explicit documentation.

Documentation also remains inconsistent:

- `burst_diffusion/docs/research_idea.md` presents contaminated MIIC numbers and describes rollout as future work.
- `burst_diffusion_guide.md:227-231` still points users to contaminated `miic_p10.yml` checkpoints.
- `burst_diffusion/README.md:171` describes train/validation splitting without the content-group/test nuance.
- `burst_diffusion_qna.md:258-273` repeats contaminated MIIC development results without warning.
- The report's accuracy heading promises mean/median while cells contain only means (`burst_diffusion_report.md:557-565`).

## Results that survived audit reruns

### Saved corrected-test result reproduced exactly

A deterministic rerun of the corrected MIIC rollout test reproduced the stored aggregate values:

- one-shot: 35.02 dB;
- recurrent/final prediction: 35.65 dB;
- average-of-16: 26.06 dB.

This supports artifact consistency, not test independence.

### Fresh unused-source diagnostics

The fixed saved checkpoints were evaluated with fresh synthetic noise on exact-unique sources absent from the historical experiment datasets. These diagnostics were read-only and **were not persisted as versioned experiment artifacts**, so the next agent should freeze their protocol and reproduce them before citing them. The same qualification applies to all derived audit-only numbers in this handoff, including near-neighbor similarity, the Gaussian comparator, registration reanalysis, clipping-bin percentages, and tiled full-image results.

| Dataset and unused pool | One-shot − avg16 | Iteration − avg16 | Iteration − one-shot |
|---|---:|---:|---:|
| MIIC, 89 sources absent from both old and corrected experiment sets | +10.115 dB mean; 89/89 wins | +10.601 dB mean; 89/89 wins | +0.486 dB mean; 77/89 wins |
| BBBC, 574 never-selected sources | +10.733 dB mean; 574/574 wins | +11.170 dB mean; 574/574 wins | +0.437 dB mean; 408/574 wins |

Additional aggregates:

- MIIC: average-of-16 26.169 dB, one-shot 36.284 dB, iteration 36.770 dB.
- BBBC: average-of-16 28.565 dB, one-shot 39.298 dB, iteration 39.735 dB.
- On a post-hoc/descriptive higher-complexity BBBC subset with clean standard deviation at least 0.08 (`n=84`), the median one-shot advantage remained about 4.38 dB. The threshold was not prespecified and must not be presented as confirmatory subgroup analysis.

These results strongly reduce concern about simple exact-source overfitting. The BBBC aggregate mean is not a uniform per-image effect: only 84/574 sources met the stated higher-complexity threshold, where the median one-shot advantage was about 4.38 dB. Because the pool has now been inspected, it should no longer be treated as pristine for later method tuning. Freeze it as a confirmatory audit now, and obtain new acquisitions for future confirmation.

### Tiled full-image diagnostics

The core one-shot effect also survived tiled whole-image evaluation on the current holdouts:

| Dataset | One-shot PSNR | Avg16 PSNR | Difference | One-shot SSIM | Avg16 SSIM |
|---|---:|---:|---:|---:|---:|
| MIIC | 36.238 | 26.148 | +10.090 dB | 0.9473 | 0.4800 |
| BBBC | 38.314 | 28.834 | +9.480 dB | 0.9723 | 0.4773 |

Thus the center-crop protocol is unnecessary for demonstrating the core synthetic denoising effect. Whole-image or tiled evaluation should replace it in the paper-facing protocol.

### Conditional paired PSNR calculations were numerically correct

On the saved corrected MIIC test and fixed checkpoints:

- Baseline one-shot minus average-of-16: +8.991 dB, 95% CI `[7.756, 10.227]`, `p≈5.0e-8`.
- Rollout iteration minus rollout one-shot: +0.624 dB, 95% CI `[0.360, 0.887]`, `p≈0.000458`.
- Rollout iteration minus plain-40k iteration: +0.864 dB, 95% CI `[0.365, 1.364]`, `p≈0.00355`.

These are valid image-level comparisons conditional on the fixed checkpoints and evaluated set. They are not estimates of across-training-run method uncertainty, and the test provenance prevents confirmatory interpretation.

## Claim disposition for the next agent

### Retain, with narrow wording

- A fixed learned model strongly beats raw frame averaging under the matched synthetic peak-10 Poisson protocol.
- The direction of the effect survives hundreds of unused exact-unique sources and tiled full-image diagnostics, although magnitude depends strongly on image complexity.
- For the rollout-finetuned saved checkpoint, recurrent prediction exceeds its one-shot output by roughly 0.4–0.6 dB on average. The saved rollout-versus-plain recurrent contrast conditionally favors rollout, but only repeated training runs can isolate a reliable finetuning effect.
- The core schedule, target exclusion, and rollout update implementation appear internally coherent.

### Withdraw or recast as exploratory

- “locked test,” “evaluated exactly once,” or confirmatory p-value language for corrected MIIC;
- BBBC as an independent test dataset;
- “genuinely unseen scene/layout/device” generalization;
- a diffusion-specific advantage;
- equivalence to average-of-8 CD precision;
- clear superiority to average-of-4 CD precision;
- registration superiority based on pooled sigma;
- algorithm-level significance from one training seed;
- real SEM or deployable metrology effectiveness;
- “information ceiling” as the established explanation for the remaining recurrent gap;
- the approximate noise-loss floor as mathematical confirmation of the method.

## Prioritized handoff actions

### Before any further tuning or paper edits

1. Freeze the 89-source MIIC and 574-source BBBC audit protocols: immutable manifests, exact and perceptual hashes, source provenance, noise recipe/seed, source-index mapping, checkpoint hashes, commands, software environment, and per-source results.
2. Mark the historical corrected MIIC test and BBBC validation results as exploratory/retrospective in every document.
3. Generate every statistic and table from a tracked script; define the primary endpoint and multiplicity policy before inspecting further alternatives.

### Minimum experimental additions for a defensible synthetic paper

1. Build acquisition/device/layout-grouped holdouts and audit full evaluated tiles for perceptual near-duplicates.
2. Use full-image/tiled evaluation as the primary protocol.
3. Run at least 3–5 independent training seeds per arm and, if feasible, multiple grouped splits.
4. Add equal-capacity single-level Noise2Noise, clean-supervised oracle, strong classical Poisson, and competitive learned-denoiser baselines.
5. Add schedule/timestep ablations and compute/latency-matched iterative controls.
6. Use hierarchical or clustered bootstrap/permutation inference that represents both source and training-run uncertainty.
7. If claiming CD equivalence, prespecify a physically meaningful margin, collect substantially more independent average-of-8 realizations, and use an equivalence/variance-ratio analysis.
8. Archive a reconstructable artifact bundle and a single script that regenerates all paper tables and figures.

### Required for real-SEM or metrology claims

1. Acquire a genuinely new, sequestered external device/wafer/specimen test set.
2. Evaluate real aligned and misaligned bursts, multiple dose/noise levels, and Poisson-Gaussian/correlated artifacts.
3. Report physical calibration, dose/acquisition time, runtime, memory, and failure rates.
4. Replace or clearly separate clean-oracle metrology from a blind operational measurement pipeline.
5. Validate accuracy as well as repeatability against a credible reference.

## Key evidence map

- Main report and current claims: `burst_diffusion_report.md`
- Mathematical/method description: `burst_diffusion_method.md`
- Split and training-crop logic: `burst_diffusion/data.py`
- Evaluation crop and comparator logic: `burst_diffusion/evaluate.py`
- Schedule/update rule: `burst_diffusion/schedule.py`
- Rollout construction: `burst_diffusion/rollout.py`
- Metrology implementation: `burst_diffusion/repeatability.py`
- Provenance implementation: `burst_diffusion/provenance.py`
- Original MIIC evaluation: `runs/burst_diffusion/miic_p10/eval/results.json`
- Corrected MIIC test evaluation: `runs/burst_diffusion/miic_p10_dedup/eval_test/results.json`
- Corrected repeatability artifact: `runs/burst_diffusion/miic_p10_dedup/repeatability_test/repeatability.json`
- Corrected provenance: `runs/burst_diffusion/miic_p10_dedup/provenance.json`
- BBBC noising statistics: `data/BBBC038-burst-p10/stats.json`
- Original/corrected source manifests: `data/MIIC-burst-p10/sources.json` and `data/MIIC-burst-p10-dedup/sources.json`

## Recommended paper-safe summary

> We present exploratory synthetic-noise evidence that a timestep-conditioned Noise2Noise U-Net can exploit a learned image prior to outperform raw 16-frame averaging under matched peak-10 Poisson corruption. For the rollout-finetuned saved checkpoint, recurrent prediction exceeds its one-shot output, and its saved recurrent output conditionally exceeds the plain-40k control. Broader unused-source and full-image diagnostics support the robustness of this synthetic effect. Independent acquisition-grouped tests, matched denoising baselines, multiple training seeds, and real SEM validation remain necessary before attributing the gain specifically to diffusion or claiming metrology superiority.
