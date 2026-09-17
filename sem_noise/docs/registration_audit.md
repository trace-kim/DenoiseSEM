# Registration audit: commits 5e42751 and 6033065

This audit describes the retained `--registration fit` mode. Subsequent work
added a separate default `--registration affine` mode with ECC geometry and
target-to-input brightness from shared region means; see the package README.

Audited the two commits against the requested native-resolution, zero-start,
two-pass joint fit. The gain-collapse problem is reproducible without any true
brightness change. Implementation fixes do not remove that estimator bias.
The user explicitly requested retaining the specified estimator and exposing
its limitation; no brightness gate, gain clamp, search, pyramid, tile estimator,
alternative noise model, or frame rejection has been introduced.

## Findings and disposition

| Priority | Finding | Consequence | Disposition |
|---|---|---|---|
| High | Noisy moving intensities are treated as error-free predictors in `gain * moving + offset ≈ reference`. | Lowering gain suppresses noise, so gain can collapse and offset compensate despite unchanged physical brightness. Pass-1 corrected images then suppress contrast in the pass-2 reference. Convergence does not diagnose this. | Estimator preserved by request. Added explicit report/README explanation, synthetic regression, and actual fit-pixel plots. |
| High | Backtracking compared the old cost on its whole mask with the trial cost after multiplying by the new validity mask. | Moving difficult pixels out of bounds or into masked regions could appear to improve the fit without improving any retained residual. | Both parameter vectors now use the same common pixels and the actual Huber loss at a frozen scale for each comparison. |
| High | A fixed two-pixel clipping dilation did not cover a Gaussian kernel with default four-sigma truncation; moving masks covered only bilinear neighbours although intensities used cubic interpolation. | Clipped values and zero-filled mean borders could contaminate supposedly valid fit observations. Cubic prefiltering also spreads sentinel values beyond local support. | Mask full Gaussian and cubic support; numerically extend masked inputs from valid pixels before filtering. Filled pixels remain masked. Regression changes a masked block from 4095 to 1e12 and requires identical fit parameters. |
| Medium | Backtracking could set `converged=True` merely because repeated halving made a rejected step tiny. | A stalled fit could look trustworthy. | Only the unshortened normal-equation step establishes convergence. Added explicit termination reason and a stalled-step regression. |
| Medium | The Jacobian sampled a finite-difference gradient with bilinear interpolation while the residual used a cubic interpolant. | The solver and reported covariance used derivatives of a different image function. | Differentiate the actual cubic interpolant with symmetric small coordinate perturbations. This changes no fit pixels and performs no registration search. |
| Medium | Corner magnitude uncertainty dropped horizontal/vertical cross-covariance. | Corner error bars could be too large or too small. | Propagate the full relevant covariance, with an analytic regression. |
| Medium | The report requested `corner_max_px_se`, but the saved field was `corner_max_se`. | Largest-corner error bars silently vanished from the plot and table. Only the maximum corner was shown in HTML. | Correct the lookup and show all four corner components/magnitudes with errors. |
| Medium | Error-bar autocorrelation used an unpadded FFT and did not remove the residual mean. | Opposite borders were treated as adjacent and a residual DC component could inflate the estimated area. | Center valid residuals and pad for the requested lag window. Verified against direct nonperiodic autocorrelation on a masked image. |
| Medium | “Shift only” also applied gain and offset. | The displayed comparison did not match the requested translation-only correction and could conceal contrast suppression. | Translation-only now uses gain 1, offset 0. Intermediate examples add a separate affine-only panel so geometry and brightness can both be inspected. |
| Medium | The report showed tracks and differences but no actual intermediate images or brightness regression data. | A small residual or flat corrected mean could obscure why gain/offset looked wrong. | Added original/blurred/reference/corrected images, four matched-scale differences, exact final fit-pair densities, Huber weights, residual plots, and native-array exports. |

Rank-deficient systems are also explicitly detected before solving/inverting.
Constant images retain their rows and unidentifiable errors remain null in
JSON rather than becoming spurious finite confidence intervals. New input
validation covers invalid blur scales and mask shapes.

## Gain reproduction

Use eight 128×128 frames, a fixed clean signal
`70 + 2*sin(x/9) + 2*cos(y/11)`, independent Gaussian noise of standard
deviation 12 DN, and NumPy `default_rng(91)`. True gain is one, true offset is
zero, and true motion is zero for every frame.

Before changes, non-anchor pass-1 gains were approximately 0.286–0.322;
pass-2 gains were approximately 0.123–0.201, with offsets about 55.8–61.2 DN.
Many fits were marked converged. After implementation fixes, pass-2 gains
remain approximately 0.125–0.200 and offsets 55.8–61.1 DN, as expected when
preserving this estimator. Thus the numerical fixes must not be described as
a cure for gain attenuation.

For the simpler aligned, independent, non-robust regression, the relationship is

```text
estimated_gain ≈ true_gain * Var(blurred moving signal)
                           / (Var(blurred moving signal) + Var(blurred moving noise))
estimated_offset ≈ mean(reference) - estimated_gain * mean(moving)
```

This is explanatory, not a correction formula for the joint robust fit. It
does not account for moving geometry, interpolation, robust weights, or the
dependence introduced by constructing the reference from the same frames.
The statistical distinction is documented in
[SciPy's discussion of errors in explanatory variables](https://docs.scipy.org/doc/scipy-1.13.0/reference/odr.html).

The reported real-data pair near 0.26/52.31 is consistent with this failure
mechanism, but has not been traced to a specific real-data frame in this audit.
The low-contrast synthetic reproduction establishes a defect in interpretation,
not proof that every low gain is artificial.

## Remaining limits and specification details

- The eight-parameter local fit still starts at identity in each pass. Fine
  texture, repetitive patterns, or motion outside its capture range can produce
  a wrong local minimum; even a converged fit is not guaranteed correct.
- The covariance remains a weighted least-squares approximation with a scalar
  residual-correlation adjustment over ±8 pixels. It does not account for
  errors-in-variables bias, reference construction uncertainty, long-range
  scan-line dependence, or model/local-minimum error. Comparing a parameter
  solely with its error bar is insufficient to establish a physical effect.
- The reference includes each tested frame and uses pass-1 gain/offset as well
  as geometry. Both self-inclusion and contrast suppression can affect pass 2.
  This construction has been preserved and its assumptions made explicit.
- All distinct, manifest-included frames are reported. Pre-existing exact
  duplicate and manifest exclusions remain in the input audit. The first
  pass-1 frame is an identity row, not a fitted independent observation.
- Native/aligned noise measurements still apply integer/bilinear centre
  translations only. Full affine and gain/offset corrections are diagnostic
  image products; applying them to noise statistics would itself change noise
  variance and correlation, particularly when gain is biased.
- Fully clipped images or fewer than 64 valid fit pixels cannot support this
  fit. Such input failures still fail the site rather than manufacture numbers.
- Difference scales remain shared across panels of each frame, not across all
  frames. Their limits and masks are recorded. Region tables retain the
  requested shift-only/full-fit rows plus the pre-existing before rows.

## Report additions and compatibility

`intermediates/` contains up to three examples: the first included frame,
lowest-gain frame, and largest-affine-corner frame, deduplicated. This selects
examples for display only; it never selects which frames contribute to a fit.
Each example exports NPZ arrays and standalone PNG figures. The HTML embeds
the figures, while native per-frame difference PNGs remain relative links.

The brightness line is the saved joint fit. Every valid final native-resolution
pixel contributes to its density/weight plots; there is no second regression.
Tests require the exported mask and residual RMS to match the saved fit.

No config keys changed. `termination_reason` is an added fit field. Numerical
results, uncertainties, valid masks and pixel counts can change after fixes.
The `shift_only` image/table semantics intentionally change to translation
alone. Additional examples consume storage; cubic-consistent derivatives cost
more interpolation work than the old approximate Jacobian. Existing output
directories are never overwritten, so real-data reports must be regenerated
into a new output directory.

## Verification

- `python -m pytest tests/sem_noise -q`: **72 passed**, with two pre-existing
  precision-loss warnings in constant-image moment calculations. Covers recovery and error-bar
  calibration, the new numerical regressions, bias reproduction, and complete
  report generation with exact fit-pixel consistency checks.
- Generated synthetic reports for both known affine/brightness motion and
  equal-brightness low-contrast repeats. Known gains of 1.00 down to 0.85 were
  recovered at approximately 1.0000 down to 0.8503. Both sites completed.
- Visually inspected intermediate image, difference, and brightness-fit
  figures. Input images are unchanged and generated artifacts are outside
  tracked source files.
