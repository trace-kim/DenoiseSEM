# Target Ladder Report Training Workflow Audit

## Verdict

I would not sign off [target_ladder_report.md](target_ladder_report.md) as written.

The training artifacts themselves appear valid: I found no crashes, NaNs,
train/test leakage, incorrect checkpoint loading, or mismatched recipes. The
central result—`ft_consist` improves repeatability—survives a more appropriate
analysis.

However, the report contains several material statistical, causal, and
reproducibility flaws. In particular, the report does not analyze against its
declared matched control, mislabels 1σ effects as 3σ, pseudoreplicates bias
observations, and incorrectly describes most metrology evaluation as
ground-truth-free.

## Principal findings

### 1. The declared matched control was not used

The report identifies `ft_noisy` as the phase-2 control, with the fine-tuning
arms intended to differ by one objective/target change
([report](target_ladder_report.md#phase-2-controlled-fine-tuning-ladder)). Yet
its paired tests compare the target arms against N2N and scratch sobloss, not
`ft_noisy`.

Recomputing the proper paired comparisons across the ten validation scenes
gives:

| Arm versus `ft_noisy` | Mean Δ CD 3σ, px | Scenes lower | Paired p |
|---|---:|---:|---:|
| `ft_oracle` | +0.0502 | 6/10 | .470 |
| `ft_avg` | +0.0731 | 6/10 | .393 |
| `ft_distill` | +0.1005 | 9/10 | .519 |
| `ft_consist` | **−0.1514** | **9/10** | **.00587** |

Negative is better. Therefore:

- The claim that cleaner/lower-variance gradient targets consistently improve
  precision is not supported against the matched control.
- `ft_consist` does show a strong improvement against the matched control. This
  p-value also survives Bonferroni correction across the four
  fine-tune-versus-control comparisons.
- Against `ft_noisy`, `ft_consist` reduces pixel σ by 28%, site-pooled CD 3σ by
  21%, center variation by 13%, and shift variation by 13%, while losing about
  0.39 dB PSNR and increasing site-weighted absolute CD bias by about 0.069 px.

### 2. The paired “CD 3σ” effects are actually 1σ

The report's paired tables use `scene_sigmas_px`, which stores raw 1σ values;
only the aggregate summary multiplies these by three
([repeatability.py](../../burst_diffusion/repeatability.py)).

Consequently:

- `ft_consist` versus N2N is reported as −0.094 px “3σ”; the correct effect is
  **−0.282 px 3σ**.
- `ft_consist` versus sobloss is reported as −0.070 px; the correct effect is
  **−0.211 px 3σ**.
- Source examples such as source 87, −0.337 px, should be −1.011 px when labeled
  3σ.

The t statistics and p-values are unchanged by scaling, and the main aggregate
table is correct. The error is confined to the paired effect-size tables and
their prose.

### 3. The bias significance test pseudoreplicates sites

The report treats 37 measurement sites as independent and obtains p=.0266 for
the `ft_consist` bias increase. But sites within one scene share the same noisy
frames, registration and model output; the evaluation harness itself warns that
the scene is the independent unit
([summary.md](../../runs/edge_denoise/repeatability_val_ft_ladder/summary.md)).

Using ten equal-weight scenes:

- `ft_consist` versus N2N: p=.184.
- `ft_consist` versus `ft_noisy`: p=.223.
- Cluster-robust approximations are also non-significant.

There is a descriptive bias increase, but the report's claim that the accuracy
cost is statistically significant is unsupported.

Relatedly, “no significant difference” is repeatedly interpreted as exact
equality. Claims that distillation inherits teacher accuracy “exactly” or that
`ft_noisy` reproduces sobloss require a predefined equivalence margin and an
equivalence test, neither of which was used.

### 4. Most metrology metrics are not ground-truth-free

The deployment appendix says CD, center and shift precision do not require
ground truth and transfer unchanged to real samples. The implementation
contradicts this:

- CD sites and expected edge positions are obtained from the clean image
  ([repeatability.py](../../burst_diffusion/repeatability.py)).
- Predicted crossings are selected by proximity to clean crossings.
- Shift and the registration inclusion gate use the clean image as reference.

Only pixel repeatability is presently ground-truth-free. Deployment would
require fixed/manual ROIs or another non-clean reference procedure, followed by
revalidation.

The long-time-averaging claim is also premature:

- Avg8 has only two independent averages per scene, making its standard
  deviations unstable.
- `ft_consist` beats avg8 on 8/10 scenes, not every scene; paired p≈.20.
- Avg16 has one realization and therefore no repeatability estimate at all.

### 5. The “unbiased target” premise is not exact for this dataset

The report models stored noisy values as unbounded `Poisson(xp)/p` and calls
fresh and averaged noisy targets unbiased. Actual generation clips to `[0,1]`
and then quantizes to 8-bit
([pipeline.py](../../noising_pipeline/pipeline.py)).

For the actual corpus:

- Expected aggregate target bias is approximately −0.00222.
- Observed bias is −0.00223.
- At the dataset's brightest intensities, expected clipping bias reaches
  approximately −0.0586.

Averaging reduces variance but not this clipping bias. Therefore oracle and
noisy-target arms differ in conditional mean as well as target variance,
weakening the report's target-variance-only interpretation.

### 6. The distillation explanation is not algebraically faithful to the implementation

Two issues occur:

- The distillation loss constrains variation in `S f(y)`—the gradient
  domain—whereas `ft_consist` directly constrains `f(y_i)-f(y_k)` in image space
  ([train.py](../train.py)). Calling this the same hidden consistency term is an
  analogy, not an exact decomposition.
- Each distilled target averages teacher outputs over all 16 replicas
  ([distill.py](../distill.py)), while the training input is sampled from those
  same replicas ([data.py](../data.py)). The target therefore contains the
  current input's teacher output with weight 1/16. The report's population
  decomposition omits this covariance term.

Leave-one-out teacher targets or a disjoint target burst would match the
intended independence assumption.

### 7. `ft_consist` is not compute-matched

`ft_noisy` performs one batch-8 forward per step. `ft_consist` performs a second
batch-8 forward with gradients ([train.py](../train.py)).

Thus the 10,000-step comparison represents approximately 80,000 versus 160,000
network input examples and about 11 versus 20 active training minutes. The
outcome comparison is valid, but the gain cannot be attributed entirely to the
consistency term—or used for a clean efficiency claim—without a two-view or
equal-compute control.

### 8. Reproducibility and lifecycle flaws

The current artifacts appear internally consistent, but their provenance is
incomplete:

- Evaluation output stores names for edge arms, not checkpoint paths or hashes.
- Contrary to the report, `repeatability.log` does not contain the exact
  evaluation command.
- Dataset provenance hashes clean sources but not noisy replicas.
- The distillation manifest does not hash individual targets or bind them to a
  dataset digest, and training does not enforce the manifest.
- Initialization checkpoint hashes are not recorded.

There were also two complete seed-0 `ft_consist` runs in the same run directory.
The second overwrote the first checkpoint and provenance while TensorBoard
histories were merged. Evaluation timestamps match the second run, so this is
not evidence of result substitution, but the first execution is no longer
auditable. The two supposedly identical runs differed slightly because
deterministic CUDA execution is not enforced.

Separately, resume only warns about incompatible configuration changes and then
restores the old optimizer state ([train.py](../train.py)). A same-shaped model
could therefore resume under a different objective while being labeled with the
new configuration. This was not exercised by the reported runs but is a real
workflow defect.

## Method-by-method disposition

- `single_frame`, avg2/4/8/16: implementation correct; avg8/avg16 replication is
  insufficient for the claims made.
- N2N teacher: correct step-30,000, one-step, fresh-target checkpoint; local and
  original teacher hashes match.
- Sobloss: configuration and image-plus-Sobel objective match.
- Hybrid: `[y, Sy]` construction and training route match.
- Pure gradient: the Sobel train path works, but reconstruction masks all
  spectral gains below `1e-3`. It therefore discards some modes that training
  penalizes and is not the “exact inverse” described by the configuration.
- `ft_noisy`: correctly implemented and should have been the primary analysis
  control.
- `ft_oracle` and `ft_avg`: correctly implemented, but neither demonstrates a
  target-quality benefit against `ft_noisy`.
- `ft_distill`: all 86 train/validation targets exist, are finite float32
  512×512 arrays, and exclude test data; the independence and provenance
  concerns above remain.
- `ft_consist`: the third replica is correctly distinct, and the objective is
  implemented as specified. Its repeatability improvement is the most credible
  result, subject to compute matching and multiple training seeds.

The report also incorrectly says every run used `λ_image=1, λ_gradient=4`; N2N
uses zero gradient weight, and the pure-gradient arm uses `λ_image=0,
λ_gradient=1`.

## Checks that passed

- All fine-tune checkpoints reached step 10,000; the earlier arms reached step
  30,000.
- Checkpoint-embedded configurations match the current recipes.
- EMA weights are present and are used by evaluation.
- The five fine-tune arms have identical primary sampler/RNG states;
  consistency's auxiliary replica is distinct.
- No test split was used for training or distillation.
- No runtime errors or non-finite weights were found.
- Stored aggregate results match the report's principal table.
- Checkpoint selection was fixed-final-step rather than best-validation
  cherry-picking.
- A four-scene smoothness diagnostic found modest additional texture smoothing,
  not catastrophic blur; it does not replace the still-missing slope/LER/MTF
  evaluation.
- `python -m pytest tests/edge_denoise
  tests/burst_diffusion/test_burst_diffusion_repeatability.py -q`: **75 passed**,
  with two expected fixture warnings.

No files were modified during the audit.

The defensible conclusion is: **`ft_consist` is promising and improves
repeatability for this fixed checkpoint and validation set, but the
target-ladder mechanism, accuracy-cost significance, ground-truth-free
transfer, and long-time-averaging claims are not established by the current
experiment.**
