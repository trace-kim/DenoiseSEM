# Review of the real SEM preprocessing proposal

Only two observations are assumed: inter-acquisition drift and changing
brightness. Charging is a hypothesis, not an identified cause. No assumption
about saturation, growing features, Poisson noise, temporal independence, or
spatially uniform darkening is needed for the implemented diagnostic.

## The input stays native; fitting need not happen after patch sampling

For a model intended to denoise a native single acquisition while preserving
its position and level, leave the sampled input pixels unchanged. Register the
target into that input's coordinates and match the target's expected brightness
to that input. Apply these relative transforms when assembling each pair.

The transform parameters can be estimated beforehand from the much larger
registered field. They need not be re-estimated on each sampled patch. For
global gain/offset, applying the correction before or after cropping is
algebraically equivalent. For geometry, sample the native target once directly
in the input coordinates with enough interpolation support; avoid successive
resampling. Registration error and interpolation can still affect fine edges.

Small-patch fitting is less identifiable and can copy the input patch's random
mean/contrast fluctuation into its target. Full-frame estimates also have finite
noise dependence; they are not magically independent. For strict training
validation, estimate calibration outside the training patch (with a margin),
use independent calibration information, or assess coefficient sensitivity
across disjoint spatial regions. Never include the input itself in an averaged
target. Do not split repeated acquisitions of one site across train/validation.

If each frame maps to a common reference by `C_i = a_i * I_i + b_i`, a registered
target `J` from frame `j` for input frame `i` becomes:

```text
target_in_input_units = (a_j * J + b_j - b_i) / a_i
```

`sem_noise.brightness.match_target` implements that arithmetic without access
to the input image. This change implements noise-analysis calibration and this
utility; it does not wire calibration into the training dataset loader.

## What is sound and what needs correction in the explainer

- The core concern is valid: misregistered or systematically differently lit
  targets can teach an unwanted mapping. However, the claimed precise cause
  of a particular trained network's shifts needs experiments with that network.
- The Noise2Noise argument is about the appropriate conditional target
  expectation for the loss. Different target variances can be acceptable for
  squared error; that is not a guarantee for every noise process or loss.
  See the [original Noise2Noise paper](https://proceedings.mlr.press/v80/lehtinen18a.html).
- A gain plus offset is a useful empirical candidate, not a proven physical
  description of charging. Spatial residuals must test its adequacy. Histogram
  equalization and flexible local corrections would unnecessarily alter contrast.
- Fitting each frame against a different uncorrected leave-one-out mean gives
  a different reference brightness for every fit. Those coefficients cannot
  simply be treated as coordinates in one common scale. Use a fixed reference
  or explicitly gauge-normalized joint estimates instead.
- Smoothing/averaging reduces noise but does not make regression with noisy
  predictors unbiased. Millions of spatially correlated pixels do not imply
  millions of independent measurements. The quoted precision and universal
  0.03-pixel acceptance level are not established for these acquisitions.
- Bicubic interpolation is not universally geometrically harmless. Its effect
  depends on sampling, edge shape, kernel, and phase. It changes noise covariance
  and can change edge profiles. Registration fitted from noise also couples
  target and input noise.
- Half the variance of an adjacent difference is half the sum of the two
  variances minus their covariance, plus any residual signal contribution.
  It is not automatically the noise variance of either individual frame.
  Moving to lag two does not prove independence.
- A flat mean after calibration is partly a fitted outcome. A flat residual
  spectrum is not a necessary condition: real noise may be correlated, and
  interpolation introduces correlation. Inspect systematic residual structure
  without declaring all nonwhite power to be specimen change.
- Saturation, spots growing under the beam, scan shear, and a need for a learned
  registration helper are not observations supplied here. They should not drive
  this correction. Neither brightness curves nor a variance/mean plot uniquely
  identifies charging versus other causes.

## Implemented analysis

The existing `python -m sem_noise analyze ...` command automatically includes
the brightness comparison. Translation-accepted frames are measured on the
same valid overlap crop. The first accepted frame is the fixed brightness
reference; its identity is recorded. Original data are never rewritten.

1. Average disjoint blocks (up to 16 by 16 pixels; smaller on small images).
   Assign one third of blocks to validation by grid position.
2. Fit an offset using the median training-block difference. When block contrast
   is sufficiently above a conservative within-block variance proxy, also fit
   a robust global gain plus offset. That proxy is not a detector-noise estimate.
3. Select gain plus offset only if it improves validation RMSE by at least 2%
   over offset, with finite gain strictly between 0.25 and 4. Otherwise retain
   offset. Decline the selected correction if it worsens validation versus the
   uncorrected image. These are engineering gates, not confidence intervals.
4. Apply accepted parameters to unsmoothed registered floating-point copies,
   with no clipping, quantization, or histogram operations. No additional spatial
   warp is performed. Registration-disabled frames receive no estimated
   correction because matching coordinates have not been established.

This is a conservative first diagnostic. A noisy anchor and noisy predictors
can bias coefficients; validation is conditional on the prior registration.
Gain and offset are not separately identifiable on flat fields, so only offset
is fitted there. Block validation can reject gross inconsistency but cannot
prove a global model is correct at every edge. The current brightness analysis
uses the main translation registration, not the optional affine example fits.
Residual geometry can therefore remain visible in the difference panels.

The analysis maps copies to a fixed reference to compare brightness evolution.
That is a diagnostic coordinate choice, not an instruction to normalize training
inputs. Calibration changes noise variance (by gain squared for fixed gain)
and parameter estimation adds dependence. Existing native/aligned noise
statistics remain unchanged and are explicitly distinguished from these plots.

## Report outputs

Each site report now contains:

- `brightness_evolution.png`: original and corrected mean versus acquisition
  index, pixel residual RMSE, gain, and offset. Missing/rejected fits remain gaps.
- `brightness_difference_*.png`: registered uncorrected and corrected examples,
  their signed difference, and before/after residuals against the reference.
  Image panels share a display scale; paired residuals share a symmetric scale.
- `brightness.csv` and `brightness.json`: frame identities, fit status/reasons,
  coefficients, block validation scores, before/after means, and slope comparison
  on the same successfully calibrated frames (DN per acquisition index).
- `brightness_examples.npz`: full-resolution floating-point example arrays.

Correction differences use `after - before`. Residuals use `image - reference`.
Unavailable examples explicitly show unchanged copies, not successful fits.
The reference is a noisy acquisition, not ground truth. No claims about real
data improvement should be made until these outputs are examined on that data.
