# Real N2N preprocessing experiment: interpretation and next steps

The experiment compares six runs based on `sem_real_n2n.yml`: affine +
percentile, translation + percentile, none + percentile, affine + none,
and the previously trained translation + none and none + none baselines.
It does not compare six different denoising objectives or training stages.
The observations below were supplied by the experimenter; remote event files
and checkpoints have not been inspected here.

## What the curves currently support

| Registration group | Reported training plateau | Reported validation plateau |
|---|---:|---:|
| None | approximately 0.0512 | approximately 0.0516 |
| Translation | approximately 0.0435 | approximately 0.0454 |
| Affine | approximately 0.0417 | approximately 0.0395 |

These numbers were grouped by registration, not reported separately for all
six treatments. They do not quantify a percentile-versus-none brightness
advantage by themselves.

**The early, smooth plateau is not inconsistent with the earlier DDIM run.**
The current trainer logs the average over `training.log_every` optimizer steps,
and each step's value already averages its accumulation microbatches. The
documented four-arm training commands used effective batch 64 per process
(16 x 4); verify the completed checkpoints' actual settings. A large amount of
averaging precedes TensorBoard's smoothing slider. Smoothing zero therefore
means no additional UI smoothing, not individual patch or optimizer losses.
The current image objective is masked mean squared error in model units
[-1,1]. DDIM's repository loss instead samples diffusion timesteps and fresh
Gaussian noise and sums pixel errors before the batch mean. Loss scales and
fluctuations are not directly comparable between these objectives.

**Smoothing 0.99 does not establish continued learning until 20k–30k steps.**
It applies a display filter to already logged points. Its apparent delay
depends on event spacing and the TensorBoard implementation. Judge convergence
from unsmoothed events and saved checkpoint outputs. UI smoothing is distinct
from EMA of model weights: this trainer uses EMA weights for validation when
enabled, while training loss comes from live weights.
[TensorBoard's scalar guide](https://www.tensorflow.org/tensorboard/scalars_and_keras)
describes turning smoothing down to inspect the unsmoothed values.

**Lower affine loss is consistent with better target alignment, but does not
prove better denoising or dimensional fidelity.** Under independent,
conditionally unbiased target noise, noisy-target squared error contains both
prediction error against the latent signal and target-noise variance. This is
the basis of [Noise2Noise](https://proceedings.mlr.press/v80/lehtinen18a.html).
Here, interpolation changes target noise variance/correlation, percentile
matching changes its scale and mean, and masks change the contributing pixels.
Corrections estimated using noisy input/target frames may also violate the
simple independence assumptions. The six losses do not have a guaranteed common
noise floor. A smoother, geometrically biased predictor can reduce loss while
worsening CD. Treat the plateau ordering as evidence about each training
objective, then evaluate all saved outputs with a common protocol.

**The small rise in validation loss after approximately 2k steps could indicate
mild overfitting or model drift, but is not established from this description.**
Its magnitude relative to variation matters. Validation below training for the
affine group is not inherently contradictory: the sites, fixed pairs, valid
support and EMA/live weights differ. Do not pool train/validation losses or
convert their absolute values into detector-noise sigma without checking units
and the target contribution.

**Brightness-none spikes need investigation, not an automatic outlier label.**
Both current real-data factories use fixed validation pairs and crop windows;
the inline factory inherits that policy. For N2N they repeatedly use input
frame 0 and target frame 2 (zero-based, when available), plus the same selected
sites/windows. Matching measurements are precomputed. A different randomly
selected bad validation frame therefore does not explain isolated spikes in
this code. Brightness mismatch could make the fixed pairs more sensitive to
occasional prediction offsets or optimization instability, but that is a
hypothesis. Replay the actual spike checkpoint and neighbouring saved
checkpoints on the same batch. Inspect per-example loss, prediction/residual
maps, input/target means, valid-pixel counts, correction parameters and output
ranges. Compare EMA and live weights, and check resumed/overlapping event logs
if the spike cannot be reproduced. Do not silently remove the events.

Code references: `edge_denoise/train.py` (`run`, `_validate`, `_loss_terms`),
`edge_denoise/real_data.py` (`val_batch`), `edge_denoise/real_matching.py`
(`MatchedRealPairFactory`), and `ddim/functions/losses.py`.

## Next experiment and corrected implementation plan

1. **Fix the experimental identities.** Use the six treatments above, keeping
   architecture, objective, normalization, native acquisitions and site splits
   comparable. Read inline matching from checkpoints and legacy registration
   from original prepared manifests. Keep legacy estimator/interpolation
   settings visible; a shared method name alone does not establish identical
   preparation. Keep checkpoint step, seed, optimizer and batching settings
   available in the resolved config.
2. **Choose checkpoints on validation data first.** Compare the earliest
   available saved checkpoint near the raw plateau, a later checkpoint, and
   the final checkpoint on the same validation sites. If no checkpoint was
   saved near step 2k, state that limitation; do not pretend the curve alone
   supplies its output images. Inspect spike checkpoints where available.
   Use native noise/brightness stability, contours, repeatability, and visual
   edge preservation together. Lower loss or lower output variance alone is
   insufficient. Extending training is not the immediate priority.
3. **Pilot a common offline comparison on one test hole-array site.** For 128
   native acquisitions, save sixteen nonoverlapping raw means (1–8, 9–16,
   ..., 121–128), and a separate full average for viewing. Run all six
   checkpoints on each individual acquisition. Preserve uint8 range audits,
   clipping warnings, and the same saved-pixel analysis for every arm.
4. **Include the actual registration comparison.** Analyze raw, block-mean
   and model-output series from decoded saved uint8 pixels without correction.
   Independently estimate each output's drift against the same full-average reference and plot
   output-minus-raw drift. Retain raw-frame translations for hole correspondence
   so output-induced movement is not corrected away. Use native saved pixels
   for mask/refined contours and ECD. Keep brightness/charging trends visible
   by comparing each output directly with its corresponding raw input. No
   acquisition gain/offset diagnostics run on model outputs.
5. **Publish one visual report and matching TensorBoard sequences.**
   Use full-field A/B images with acquisition sliders, a wipe divider and each
   image's own contours. Clicking a hole shows the measured area, ECD and its
   trace across acquisitions. Keep coverage, repeatability and detailed exports
   folded away until needed. Raw segmentation failure must not suppress usable
   model results. Offline TensorBoard image steps mean acquisition numbers;
   checkpoint summaries use separate tags. Opt-in live panels use fixed
   train/validation examples only; no test images enter training.
6. **Review the pilot before expanding across test sites.** Neither the full
   mean nor the eight-frame means are ground truth. No PSNR/SSIM or accuracy
   claim is warranted. Avoid selecting checkpoints using the test comparison.

Steps 1 and 3–5 are implemented in the revised comparison command and the
existing opt-in live panel path. Checkpoint selection, spike replay and the
real-data pilot require the actual remote artifacts and remain the next
experimental work. No new training run or model-objective change is requested
by this plan. See [the server command and configuration](real_sem_comparison.md).
