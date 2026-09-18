# Real SEM noise analysis

Characterize repeated noisy measurements of the **same site** before training.
Defaults assume 128 acquisitions per site, with independent results for each
of 15–30 sites. This package imports none of the training packages and requires
no GPU. Input images are read only; results go into a new output directory.

The default **target-to-input diagnostic** measures translation and affine
geometry separately from brightness. For each included raw input A, another
noisy frame B is sampled into A's coordinates and matched to A's brightness
using two corresponding region means. A full-image percentile gain/offset fit
is also reported alongside the original method for comparison. A stays untouched. The report shows
the regions, actual means, intermediate images, four native-resolution
difference panels, and 4×4 residual tables. The old joint eight-parameter fit
is available explicitly with `--registration fit` for comparison.

## Start with the PNG data

```powershell
python -m pip install -e ".[analysis,dev]"
python -m sem_noise analyze --input data/sem-real --output output/sem-noise-01
```

The simple layout is `data/sem-real/site_01/frame_0000.png` through
`frame_0127.png`, then `site_02/`, etc. Filenames must encode acquisition order.
Open `output/sem-noise-01/index.html`. It links to each site's offline report,
figures, numerical tables, and full-resolution maps. Reports embed their
figures; the per-frame difference images are separate PNG files next to the
report, so download the entire output directory. `sem-noise` is also installed
as a console command.

If another session uses the existing editable environment, use a separate
environment in this worktree. Do not repoint its editable install while that
session is working. Existing outputs are never overwritten. A failed site does
not prevent other sites from running: exit code 1 and the index identify partial
failures; exit code 2 indicates a command/input/configuration error.

### Unknown layout or acquisition order

Make an inventory recursively, then **edit site labels and frame indices to
match the equipment's acquisition record**:

```powershell
python -m sem_noise inventory --input data/sem-real --output tmp/sem-order.csv
python -m sem_noise analyze --input data/sem-real --manifest tmp/sem-order.csv --output output/sem-noise-01
```

The inventory suggests one site per containing directory and natural numeric
filename order (`scan2.png` before `scan10.png`). It cannot infer sites from
unknown names or reconstruct lost acquisition order. It never uses filesystem
modification time as a measurement timestamp. Without a manifest, discovery
uses immediate site folders, or treats images directly in the input directory
as one site. Mixed root images and site folders are rejected. Use a manifest
for deeper layouts. Do not combine different patterns into one site.

Example CSV (paths relative to `--input`):

```csv
site,path,frame_index,page,timestamp_s,include,voltage_kv,dwell_us
pattern_A,batch7/random_name.png,0,0,0.0,true,5,1
pattern_A,batch8/another_name.png,1,0,2.4,true,5,1
pattern_A,batch8/damaged_scan.png,2,0,4.8,false,5,1
pattern_B,other/first.png,0,0,120.0,true,5,1
```

Only `site,path,frame_index` are required. `frame_index` is a unique nonnegative
ordinal per site, retaining gaps for missing acquisitions. Rows are sorted by
this column. `page` defaults to zero and is useful for TIFF/NPY stacks.
`include=false` explicitly excludes a damaged frame. Extra metadata columns are
preserved; variation within a site is flagged. Group incompatible acquisition
settings separately. Reports do not pool sites into an equipment-wide fit.

`timestamp_s` is optional, but must be present for every frame in a site and
strictly increase when supplied. Relative or Unix seconds work. Without it,
a known uniform interval and pixel size can be supplied:

```powershell
python -m sem_noise analyze --input data/sem-real --manifest tmp/sem-order.csv --output output/sem-noise-timed --frame-interval-s 2.4 --pixel-size-nm 1.5
```

Frame interval is the time between acquisitions, **not pixel dwell time**.
Timestamps take precedence. Uniformity is checked within 1% using time
differences divided by acquisition-index differences. Irregular timing disables
seconds-based Allan curves and temporal FFTs; frame-lag results still describe
the ordered sequence. Without timing, duration in seconds is unavailable and
frequency is cycles/frame. Sites can have independent time origins.

### ROI and acquisition settings

Use `sem_noise/configs/default.yml` as a starting point. CLI options override YAML:

```powershell
python -m sem_noise analyze --input data/sem-real --manifest tmp/sem-order.csv --config sem_noise/configs/default.yml --output output/sem-noise-cropped --roi 0 960 0 1024
```

ROI is `y0 y1 x0 x1` with exclusive stops. Adapt these example dimensions to
the actual PNGs. Crop burned-in labels, scale bars, and non-imaging borders
**before registration**. Set `black_level` and `white_level` to the true exported
clipping bounds: unscaled 12-bit values in uint16 PNGs have upper bound 4095,
not 65535. Otherwise integer storage limits are used and flagged; float images
have no implicit clipping bounds. Pixels at the bounds are masked out of the
registration fit and out of the noise-distribution masks. Measurements are not
gamma-corrected, equalized, normalized per frame, denoised, or photometrically
compensated; the fitted gain and offset are applied only to the difference
images and the brightness track.

Retain voltage/current, detector, dwell time, frame time, scan direction, pixel
size, working distance, and any automatic contrast/filtering/averaging settings.
These can explain differences between otherwise comparable datasets.

### Raw image histograms

Every site report includes an expandable **Raw image histograms** section for
the first, middle, and last included acquisitions, even when registration is
disabled or a pair estimate fails. These histograms use every original pixel
inside the configured ROI (the whole image when no ROI is set), including
clipped values, before blur, registration, brightness correction or comparison
masking. Axes are linear intensity in DN and pixel count.

Eight-bit images use one bin per integer intensity over the full storage range
(0–255 for uint8). Other dtypes use 256 shared bins spanning the examples'
observed range. The table shows the source dtype, pixel counts, minimum/maximum,
P1/median/P99, mean, standard deviation and counts at the clipping bounds.
`raw_histograms.png` and `raw_histograms.csv` save the figure and exact bin counts.

## Default: raw target-to-input matching

```powershell
python -m sem_noise analyze --input data/sem-real --output output/sem-pairs-01 --registration affine
```

Install `.[analysis]` in the environment running the analysis. This path uses
`opencv-python-headless>=4.14` for ECC with separate reference and moving masks;
no desktop OpenCV windows or GPU are needed.

1. Fit translation against the first included frame using ECC on 1-pixel
   blurred copies. Each translation starts from the last successful estimate.
   Refine with full affine ECC, initialized by that frame's translation.
   Affine includes translation; it is applied as one transform, not two warps.
   The unit function also supports direct affine fitting from identity.
2. Build a geometry-only mean to choose shared physical brightness regions.
   Select its blurred lower and upper intensity quartiles once per site;
   blue/low and orange/high masks are displayed and saved. This mean is a
   region-selection aid, not a replacement target. It has no fitted brightness
   correction. Clipped pixels and interpolation/blur support are excluded.
3. Diagnose one target per input, half the included burst away cyclically.
   If stored sampling matrices map reference coordinates to raw frames, the
   matrix sampling B onto A is `W_B @ inverse(W_A)` in homogeneous coordinates.
   Warp the original B once. Move only the region labels into A's coordinates;
   never resample A.
4. Measure low/high means in native DN from raw A and affine-aligned B on
   exactly the same valid region pixels. Compute
   `g = (high_A - low_A) / (high_B - low_B)` and
   `b = low_A - g * low_B`. The answer for this pair is `g * warped_B + b`.
   No pixelwise brightness regression, clipping, or quantization is applied.

Both images in these pairs are noisy acquisitions. Means over corresponding
regions reduce the influence of independent pixel noise. The method assumes
the selected regions describe stable content and a global linear brightness
change. Close region means make gain sensitive to noise; inspect the reported
means and masks. A zero target-region contrast cannot determine gain and is
reported as a failure, not replaced with gain 1. A failed geometry estimate
likewise remains visible, and no correction PNG is fabricated for that pair.
If only the translation comparison fails, successful affine/brightness results
are still reported; the translation panel is grey and its RMS is blank.
ECC scores are correlations, not standard errors or guarantees of the right
alignment. Numerical parameter error bars are not claimed for this method.

### Compare both brightness estimators

For every diagnostic A/B pair, a second, independent estimator fits
`Q_A(p) = g * Q_B(p) + b` by ordinary least squares at the 10th, 15th, …, 90th
percentiles (17 points). It uses all pixels of each raw image, including
clipping bounds, **before registration**. It selects no physical sub-area,
overlap mask, or brightness region. If the user explicitly configured an ROI,
the whole supplied ROI is used; no further crop is introduced.

The original two-region algorithm and its corrected-image/difference outputs
are unchanged. `target_pairs.csv` retains their `gain` and `offset_dn` columns
and adds `quantile_gain`, `quantile_offset_dn`, `quantile_fit_rms_dn`, pixel counts,
and independent `quantile_status`/`quantile_error` fields. The table and gain/
offset tracks show both estimators. A failed geometry or original brightness
measurement does not suppress a measurable full-image percentile estimate.
Equal target percentiles make the new gain unidentifiable and retain an
explicit failure, without replacing either estimate.

The first/middle/last examples add a percentile-point plot with both fitted
lines and before/after histogram overlays. Both histogram corrections apply
the saved gains/offsets to the same full raw target, without interpolation or
clipping. The input remains untouched. Histograms use 64 shared linear bins for display only;
the percentile calculation uses all raw values directly. `pair_quantiles.csv`
saves every percentile point, including its value after the new correction.

Different noise strengths can affect distribution width and the inferred gain.
Curvature in the percentile plot shows deviations from a global linear mapping;
the reported fit RMS measures those deviations, not spatial image residuals.

The native/aligned **noise statistics still use translations only**, with no
gain/offset correction. If translation fails, the affected frame remains
unshifted in those statistics and the report warns explicitly. No frame is
dropped merely because its geometric or brightness estimate failed.

Every pair has a row in `target_pairs.csv`, and each successful pair has four
native-resolution difference panels on one shared colour scale: before,
translation only, affine only, and affine plus brightness, all minus fixed A.
The colour limit is calculated separately for each pair from the largest
absolute valid difference across all four panels, so every displayed stage
fits within its shared symmetric range. `pair_regions.csv` gives a 4×4 RMS
table for each panel.
Subtraction is performed in signed floating-point DN, after conversion from
the input storage dtype. The 8-bit palette PNG is a display encoding, not an
unsigned difference array. Each pair also reports signed min/max, absolute
difference P99 and RMS for each stage, making the scale's extreme values visible.
A single large difference sets the common limit; a ±220 DN range can be valid
for uint8 inputs. Interpolation and brightness-corrected values are not clipped
back into the original storage range.

The main `report.html` shows only the first, middle, and last included input
examples, alongside tracks covering all pairs. The linked `pair_report_full.html`
contains every pair's difference image, residual tables, geometry and brightness
measurements, including all failure reasons. CSV/JSON outputs still cover all
pairs. Raw/blurred intermediate images and scatter plots are generated for
the three examples and appear in both reports.

Each example has three pixel-to-pixel scatter plots: raw B, translation-corrected
B, and affine-corrected B versus untouched A. Every corresponding valid pixel
is plotted on the same mask and shared linear axes; there is no binning or
subsampling. Only the affine panel overlays the line through the two measured
brightness-region means. The gain/offset calculation is unchanged; no regression
is fitted to the scatter points.

| New artifact | Contents |
|---|---|
| `pair_report_full.html` | All pair comparisons and measurement tables; linked from the concise site report |
| `geometry.csv` | Translation, affine sampling matrices, ECC scores, corner effects, failure reasons |
| `target_pairs.csv` | Input/target indices, pair matrix, both region means/counts, original and percentile gain/offset values, independent statuses, residual RMS, image links |
| `pair_quantiles.csv` | The 17 full-image percentile pairs used by each new brightness fit, with corrected target percentiles |
| `pair_registration.json` | Geometry and pair measurements with method/selection conventions |
| `pair_regions.csv` | Native difference RMS in each 4×4 region |
| `pairs/brightness_regions.npz` | Geometry-only mean, common validity and fixed low/high labels |
| `pairs/input_NNNN_target_MMMM_differences.png` | Four native panels for that pair |
| `pairs/input_NNNN_target_MMMM.npz` | Example original input/target, actual blurred copies, corrected targets, labels, masks and matrix |
| `pairs/input_NNNN_target_MMMM_distributions.png` | Example percentile plot and full-image before/after histogram comparison of both methods |

Reusable functions live in `sem_noise/pair_matching.py`: `estimate_geometry`,
`pair_transform`, `warp_target`, `select_brightness_regions`, `regions_on_input`,
`measure_brightness`, `measure_quantile_brightness`, and `match_target`. `match_target` returns aligned and
brightness-matched targets and validity; it never writes into the input arrays.
Pass its regions on the input grid, and pass actual clipping masks when known.

This implements diagnosis and reusable matching units. Training still uses
the existing `RealPairFactory`; it does not automatically consume these files.
In particular, this change does not modify training inputs or silently change
the consistency loss. The reference/target distinction and pair direction are
the same ones that future training integration must preserve.

## Legacy joint registration (`--registration fit`)

The following sections describe the retained older mode. It is not the default
and its gain bias is the reason the separate region method was added.

### What is fitted

For every frame `M` of a site, against a fixed reference `R`, one least-squares
fit on native-resolution pixels. Both copies are blurred by `registration_sigma`
(1 px). Nothing is subsampled, there is no search, no pyramid, and the fit
starts from zero shift. The model, with `u = x − cx` and `v = y − cy` measured
from the ROI centre (x right, y down), is

```text
corrected(x, y) = gain · M(x + dx + a11·u + a12·v,  y + dy + a21·u + a22·v) + offset  ≈  R(x, y)
```

Eight parameters: `dy, dx` (where the reference centre's content sits in the
frame, i.e. its drift), the four affine terms `a11 a12 a21 a22`
(dimensionless; their effect is reported as the displacement they add at the
four ROI corners, in pixels), `gain` and `offset`. The loss is Huber
(scale 1.345 × the residual's median absolute deviation, re-estimated each
iteration); pixels at the clipping bounds, the full Gaussian support
(round(4 × sigma) pixels), and the cubic interpolation support are masked.
Masked values are extended from the nearest valid pixel before filtering so
clipping sentinels cannot leak through the global spline prefilter. These
filled values never become fit observations. Gauss–Newton with backtracking runs until every
geometric step is below 0.001 px (at most 40 iterations); a frame that does not
reach that is marked `converged = false` and reported as it stands.
`termination_reason` distinguishes the iteration limit, a stalled line search,
a singular system, and the actual step tolerance. Backtracking compares the
Huber loss on the same overlapping pixels for both parameter vectors; a tiny
rejected/backtracked step does not establish convergence.

Two passes:

1. **Pass 1** — reference = the first included frame. Its own row is the
   identity by definition, not a fit.
2. **Pass 2** — reference = the registered mean of all frames, built with the
   pass-1 fits on the first frame's grid and brightness scale (pixels covered
   by every frame). Every frame, including the first, is fitted against it.
   **Pass 2 is the reported result**; pass 1 is saved for comparison.

The mean includes the frame being fitted (1/N of it), which favours the pass-1
solution and correlates reference noise with the moving frame. Its importance
depends on frame count, contrast, and noise; it is not generally negligible.

### Gain attenuation: a limitation of the specified estimator

The fit minimizes `gain * moving + offset - reference`. Moving intensities
are noisy predictors: reducing gain suppresses their noise as well as their
contrast. With correctly aligned independent images, ordinary least squares
approximately gives `gain = true_gain * signal_variance /
(signal_variance + moving_noise_variance)`, using the blurred-image variances.
Huber weighting does not remove this errors-in-variables bias. The offset
then compensates toward the reference mean. Building the pass-2 reference
from contrast-suppressed pass-1 images can compound the effect.

A synthetic equal-brightness case (mean about 70 DN, weak sinusoidal structure,
12 DN independent frame noise) reproduces gains near 0.3 in pass 1 and still
lower gains in pass 2. Such fits can converge and have small conditional error
bars. Low gain is not by itself evidence of detector/brightness drift.
The estimator is intentionally preserved: no gain clamp, brightness gate or
noise-model assumption is added. See [the audit](docs/registration_audit.md) and
the report pixel-pair plots. An errors-in-variables estimator would be a
separate methodological change requiring a noise model.

### Error bars

Each parameter's standard error is an approximate weighted least-squares covariance of the
fit, `σ̂² (JᵀWJ)⁻¹`, scaled by the frame's **residual correlation area**: the
sum of the normalized residual autocorrelation over ±8 px lags. The 1-px blur
alone makes that area about 12.6 px² for white noise (`4πσ²`), so an
uncorrected error bar would be about 3.5× too small; intrinsic noise
correlation and leftover structure enlarge it further. The area is written per
frame (`residual_correlation_area_px2`). Correlation beyond ±8 px, such as
scan-line noise that is coherent along a whole row, is not captured, so the
error bar on `dy` is optimistic on such data. Corner displacements, rotation
`(a21 − a12)/2`, shear `(a12 + a21)/2` and scale changes carry linearly
propagated errors. These describe the fit on the two images at hand, not
calibrated stage motion. They do not include gain attenuation, uncertainty in
the pass-1 reference construction, or error from a wrong local minimum.

### Difference images and region tables

For every frame, one native-resolution PNG (`differences/frame_NNNN.png`) with
three panels side by side, in original DN on one shared symmetric colour
scale:

1. **before correction** — frame minus reference;
2. **shift only** — centre translation only, with gain 1 and offset 0, minus
   reference;
3. **full fit** — all eight parameters, minus reference.

The full fit adds affine and brightness corrections. Intermediate examples
separate their effects with an additional affine-only panel. All three use one
pixel set (inside both footprints, covered by the mean, not touching clipped
values; grey elsewhere) and one limit, the largest absolute valid difference
across the three maps, printed in the caption and in `registration.csv`
(`colour_limit_dn`). The PNG is an 8-bit palette image: index 127 is zero,
0 and 254 are ∓limit, 255 is invalid. `regions.csv` gives the RMS difference in
each cell of a 4×4 grid for the same three panels on the same pixels.

Native-resolution PNGs of noise-like data do not compress: expect roughly
8–10 MB per frame at 2048×2048 (about 1.2 GB for a 128-frame site).

### Brightness track

`gain` and `offset` per frame with error bars, and the frame mean before and
after applying them on the pixels shared with the reference, together with
the reference mean on those pixels. A flat after-track is the fitted outcome,
not independent proof: the residual images show what a global gain and offset
cannot explain.

### Intermediate examples

The report shows the first included frame, the lowest-gain frame, and the frame
with the largest affine corner displacement (duplicates appear once). Each has
original, blurred, reference, shift-only, affine-only, and fully corrected
images, with shared grayscale limits. Four matched-scale difference panels
isolate the geometric and brightness changes.

The brightness plots use every final valid fit pixel: x is the geometrically
warped blurred moving intensity and y the blurred reference intensity. The
line is the saved joint fit, not a second regression. Density, final Huber
weights, and residual-versus-intensity panels expose noise and downweighting.
Binning is for display only. `intermediates/frame_NNNN.npz` preserves native
arrays, masks, weights and the eight parameters; adjacent PNGs are standalone
figures. These examples add storage proportional to up to three native frames
and their intermediate arrays.

### The noise statistics stay translation-based

`--registration fit` applies only the fitted **centre shift** to the
noise statistics; `--registration none` takes frames as aligned and skips the
fit, the difference images and the brightness track. No affine warp and no
gain/offset ever enter the native/aligned noise statistics, because a warp
changes noise variance and correlation and a gain scales it.

### Convergence radius

Starting from zero shift, Gauss–Newton follows the residual downhill. For
isolated edges and features that works over many pixels; for dense fine
texture with a correlation length of a few pixels, a drift larger than that
length can settle in a wrong local minimum. Such a frame shows a large
`residual_rms_dn`, structure left in its full-fit panel, and often
`converged = false`. Compare with `registration_pass1.csv`; both passes start
from zero.

## Measurements and interpretation

| Question | Outputs | What to check |
|---|---|---|
| Registration | dy/dx, affine terms and corner effects, all with error bars; residual RMS; difference images; region tables | Numbers against error bars; structure left in the full-fit panel; region cells that improve only under the full fit |
| Brightness | gain/offset with error bars, mean before/after | Trend versus error bar; residual images for what gain/offset cannot explain |
| Noise distribution | Residual and adjacent-difference histograms, Gaussian Q–Q, skewness, kurtosis, MAD scale, >3σ tails | Flat regions versus moving edges; no automatic distribution-family claim |
| Signal dependence | Binned mean–variance curve and affine fit | Empirical exported-DN variance model; not calibrated electron gain |
| Temporal independence | Pixel ACF, offset-removed ACF, adjacent differences | Correlated repeats can bias pair-based estimates and training |
| Slow changes | Mean, contrast, residual RMS, Laplacian RMS, early/late maps | Charging, beam/focus/specimen changes, and noise can all contribute |
| Averaging behavior | Pixel and mean-intensity Allan deviation, shuffled control, 1/√N reference | Detect departures from independent averaging without using a fake clean target |
| Spatial structure | Pair-difference 2-D PSD, x/y ACF, row/column banding ratios | Scan-line noise, anisotropy, and residual pattern energy |
| Data quality | Clipping, shape/dtype checks, duplicate hashes, unstable-pixel candidates | Inspect flags; they do not uniquely identify detector defects |

### The two pixel domains

Each site is measured in two domains on the same valid crop:

1. **Native:** nearest-integer translations preserve measured pixel values.
   Residual subpixel motion contributes especially near edges.
2. **Aligned:** bilinear subpixel translations improve structural overlap but
   attenuate and correlate noise. Treat them as a separate diagnostic.

An unregistered mean/std baseline shows motion-contaminated variation. Crop
coordinates relative to the ROI and the original shape are saved. For a
fractional shift fy, fx, independent white-noise variance is multiplied by
`((1−fy)²+fy²) * ((1−fx)²+fx²)`. Its mean is reported, but not applied as a
universal correction to possibly correlated, signal-dependent SEM noise.

### Noise estimators

Pixels clipping in any frame are removed from distribution/moment masks,
including bilinear footprints touching clipped samples. The low-gradient
mask thresholds a Gaussian-smoothed repeat mean at `flat_fraction` (default
50th percentile), then erodes one pixel. It may still contain texture. Up to
`sample_pixels` masked pixels and `distribution_samples` residuals are sampled
reproducibly. The two domains have independently derived masks; region selection
can contribute to their differences. Inspect masks before increasing sampling.

Per-pixel temporal variance uses `ddof=1`. Histograms use
`(I_t − mean(I)) * sqrt(N/(N−1))`, correcting self-inclusion variance loss for
independent stationary repeats. This does not correct temporal correlation or
make the repeat mean ground truth; higher moments remain approximate. No
pooled-pixel normality p-values or IID confidence intervals are reported.

Adjacent differences use `(I_(t+1) − I_t)/sqrt(2)`. Equal-variance independent
noise and a stable registered signal are needed for its variance to equal
single-frame variance. Positive correlation lowers this estimate. Differencing
symmetrizes the distribution, so it cannot identify original-noise skewness.

The empirical fit is `variance = a * signal + b`, using even retained positions
for signal, disjoint odd positions for variance, and weighted intensity bins.
Negative intercepts are retained: detector offsets and model mismatch can cause
them. Correlated repeats can bias this split estimate. Neither `a` nor `b` is
calibrated electron gain/read noise without a verified linear raw export and
detector calibration. Narrow signal ranges do not support a useful slope fit.

The unstable-pixel screen marks std above both five times the flat-region median
and median plus eight MAD scales. Moving edges and signal-dependent noise can
also trigger it. It is not a calibrated hot-pixel detector; clipped pixels are
excluded. Stable fixed-pattern response is confounded with the specimen without
independent dark/flat references.

### Time and space estimators

For block size m, overlapping Allan deviation is
`sqrt(mean((mean(I[t+m:t+2m]) − mean(I[t:t+m]))²) / 2)`.
The pixel version averages squared differences across sampled pixels; the
mean-intensity version uses whole-crop frame means. Powers of two extend to
one quarter of the longest contiguous run (m=32 for 128 complete frames).
Overlapping-window and possible disjoint-pair counts are exported separately;
overlapping windows are dependent. The shuffled control breaks time order but
does not make a nonstationary sequence stationary. A minimum is not a validated
optimal averaging count or training target.

Excluded/missing indices are never collapsed into adjacent time steps. ACF uses
pairs at the stated acquisition lag, blocks never cross gaps, and FFTs require
a contiguous sequence without known timing irregularity. Finite temporal-mean
subtraction biases IID ACF slightly negative, roughly −1/(N−1). The
mean-intensity periodogram uses Hann windowing and linear detrending. Without
line/pixel timing and scan geometry it cannot identify within-frame frequencies
or diagnose 50/60 Hz pickup.

Spatial spectra average up to `spatial_pairs` disjoint adjacent differences on
an unmasked central crop capped by `spatial_max_side`. Exact coordinates and
pair counts are saved. The two-sided Hann periodogram is
`abs(FFT(window * difference))² / sum(window²)`, FFT-shifted, in cycles/pixel;
its array mean estimates window-weighted variance. Directional ACF and banding
use the same patch after scalar-mean removal. Row ratio is
`width * Var(row_means) / Var(pixels)`, analogously for columns; roughly one is
the IID reference. Residual specimen/motion energy can contribute to spectra.

## Output and resource use

The default affine mode adds the pair artifacts listed above. Files marked
`fit` below belong only to the legacy joint-fit mode.

| Artifact | Contents |
|---|---|
| `index.html`, `summary.csv`, `summary.json` | Site comparisons, status, and all numerical results |
| `provenance.json` | Config, dependency versions, analysis-source SHA-256 |
| `input_manifest.json` | Relative paths, acquisition order, metadata, file/pixel hashes |
| `site_001/report.html`, `*.png` | Offline report and standalone figures |
| `site_001/summary.json`, `inputs.json` | Site metrics, units/crops, flags, input audit |
| `site_001/registration.csv` (`fit`) | Pass 2: the eight parameters with standard errors, residual RMS, corner effects, rotation/shear/scale, convergence, brightness means, panel RMS and colour limit, per frame |
| `site_001/registration_pass1.csv` (`fit`) | The same numbers against the first frame |
| `site_001/registration.json` (`fit`) | Both passes with the 8×8 covariance per frame, conventions, and the site summary |
| `site_001/regions.csv` (`fit`) | 4×4 region RMS for before / shift only / full fit, per frame |
| `site_001/differences/frame_NNNN.png` (`fit`) | Native-resolution three-panel difference image per frame |
| `site_001/frames.csv` | Raw quality, registration summary columns, both domains' frame metrics |
| `site_001/native_*.csv`, `aligned_*.csv` | Intensity bins, averaging, temporal/spatial ACF |
| `site_001/maps.npz` | Means/std, early/late maps, masks, spatial PSD, plus the geometry-only mean/regions (`affine`) or pass-2 reference (`fit`) |

Decoded-content hashes include full image shape and canonical numerical pixels
before ROI selection. Exact repeated content within a site is excluded after
its first included occurrence, preserving index gaps. Cross-site duplicates are
flagged in the overview. No training split is created: later splits must keep
all repeats of a site together and inspect cross-site duplicate groups.

One temporary disk-backed stack is processed at a time, normally
`frames * height * width * 4` bytes: about 2 GiB at 128×2048×2048. Float64 cache
values use eight bytes. Scratch is inside the site's output and is closed and
removed after success or a handled failure. Registration processes one frame
at a time; pair diagnostics save full-resolution difference images for every
included input and intermediate arrays for three example inputs.
The legacy joint fit holds one frame's design
matrix in memory (about 0.3 GB at 2048²) and takes a few seconds per frame per
pass at that size on one core; a 128-frame 2K site runs in roughly half an hour
plus the difference images. Full-resolution maps are not downsampled; bounded
samples/crops are used for temporal, distribution and spectral calculations.

PNG (8/16-bit grayscale) is the intended input. Grayscale TIFF/multipage TIFF,
BMP, JPEG (flagged as lossy), and numeric H×W/T×H×W NPY also work. RGB PNG,
BMP, and JPEG are accepted when all three decoded channels are exactly equal
at every pixel; one channel is used directly without conversion or normalization.
RGB images with unequal channels, color TIFFs, alpha/palette
images, nonfinite values, mixed within-site shape/dtype, unsupported TIFF page
shapes, and integer intensities outside ±2²⁴ are rejected. Some compressed TIFFs
need `imagecodecs`. Preserve original PNGs; screenshots or processed exports
can change the statistics. Exported-DN analysis does not undo such processing.

## Validation and equipment-level limits

```powershell
python -m pytest tests/sem_noise -q
python -m sem_noise demo --output tmp/sem-noise-demo --frames 128 --size 128
python -m sem_noise analyze --input tmp/sem-noise-demo --manifest tmp/sem-noise-demo/manifest.csv --output output/sem-noise-demo
```

The demo creates **synthetic** white Gaussian, Poisson–Gaussian, and
drift/charging/temporally correlated/row-noise PNG sites. `truth.json` records
parameters. This validates behavior, not tomorrow's equipment characteristics.

128 repeats only characterize their recorded duration and conditions. They do
not establish hour/day stability or supply clean ground truth. For stronger
equipment-level conclusions, revisit sites after longer delays, include
dark/uniform references, and repeat settings across sites. Check registration,
early/late changes, temporal correlation, and both pixel domains before choosing
Noise2Noise pairs or using a repeat mean as a target.

## References

- [NIST, Handbook of Frequency Stability Analysis](https://www.nist.gov/publications/handbook-frequency-stability-analysis): stability estimators; here the adjacent-block statistic is applied to measured intensity, not oscillator phase.
- [Foi et al., Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data (2008)](https://pubmed.ncbi.nlm.nih.gov/18784024/): signal-dependent raw-data noise models. This pipeline instead estimates empirical variance from repeats; the reference does not establish that an SEM PNG export follows that physical model.
- Huber, *Robust Statistics* (1981): the Huber loss and its 1.345 constant used by the fit.
