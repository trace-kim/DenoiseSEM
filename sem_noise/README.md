# Real SEM noise analysis

Characterize repeated noisy measurements of the **same site** before training.
Defaults assume 128 acquisitions per site, with independent results for each
of 15–30 sites. This package imports none of the training packages and requires
no GPU. Input images are read only; results go into a new output directory.

## Start with the PNG data

```powershell
python -m pip install -e ".[analysis,dev]"
python -m sem_noise analyze --input data/sem-real --output output/sem-noise-01
```

The simple layout is `data/sem-real/site_01/frame_0000.png` through
`frame_0127.png`, then `site_02/`, etc. Filenames must encode acquisition order.
Open `output/sem-noise-01/index.html`. It links to each site's offline report,
figures, numerical tables, and full-resolution maps. Reports embed their images
and require no web service. `sem-noise` is also installed as a console command.

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
have no implicit clipping bounds. Measurements are not gamma-corrected,
equalized, normalized per frame, denoised, or photometrically compensated.

Retain voltage/current, detector, dwell time, frame time, scan direction, pixel
size, working distance, and any automatic contrast/filtering/averaging settings.
These can explain differences between otherwise comparable datasets.

## Measurements and interpretation

| Question | Outputs | What to check |
|---|---|---|
| Registration error | x/y drift, drift steps, overlap correlation, competing peaks, local tile residuals | Whole-frame translation versus local distortion; periodic-pattern ambiguity |
| Noise distribution | Residual and adjacent-difference histograms, Gaussian Q–Q, skewness, kurtosis, MAD scale, >3σ tails | Flat regions versus moving edges; no automatic distribution-family claim |
| Signal dependence | Binned mean–variance curve and affine fit | Empirical exported-DN variance model; not calibrated electron gain |
| Temporal independence | Pixel ACF, offset-removed ACF, adjacent differences | Correlated repeats can bias pair-based estimates and training |
| Slow changes | Mean, affine gain/offset, contrast, residual RMS, Laplacian RMS, early/late maps | Charging, beam/focus/specimen changes, and noise can all contribute |
| Averaging behavior | Pixel and mean-intensity Allan deviation, shuffled control, 1/√N reference | Detect departures from independent averaging without using a fake clean target |
| Spatial structure | Pair-difference 2-D PSD, x/y ACF, row/column banding ratios | Scan-line noise, anisotropy, and residual pattern energy |
| Data quality | Clipping, shape/dtype checks, duplicate hashes, unstable-pixel candidates | Inspect flags; they do not uniquely identify detector defects |

### Registration and the two pixel domains

Translations are estimated from smoothed copies with tapered boundaries,
unnormalized FFT cross-correlation, and subpixel DFT refinement, followed by
interior least-squares refinement of shift, gain, and offset on registration
copies. The interior fit reduces taper bias without changing measured pixels.
The first pass
uses the first included frame; the second uses leave-one-out provisional means
to reduce reference noise. The coordinate origin remains the first included
frame if its refinement passes. This does not remove all data-dependent
registration bias. Frames outside `max_shift_px` per axis or below
`min_correlation` are excluded and recorded. Correlation is measured on smoothed
overlap and is not a calibrated probability of correctness.

`registration_max_side` caps registration copies, not measurements. Approximate
initial numerical shift spacing is `ceil(max_side / registration_max_side) /
upsample_factor` pixels; the interior fit is continuous. Neither is a claim of
**accuracy or uncertainty**. Similar separated
peaks within the search range flag ambiguity without automatically rejecting
the frame. Repeated lines/grids can produce plausible wrong matches despite
these checks; inspect a distinctive ROI and compare acquisition priors.

Local checks sample up to 16 frames with a 3×3 grid, tiles at least 24×24, and
residual shifts at most 3 pixels. No nonrigid warp is applied. Residuals can
reflect charging distortion, rotation, scan jitter, low texture, or ambiguity.
`--registration none` provides a stationary/flat-reference control with drift
metrics unavailable.

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

Pixels clipping in any accepted frame are removed from distribution/moment
masks, including bilinear footprints touching clipped samples. The low-gradient
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

| Artifact | Contents |
|---|---|
| `index.html`, `summary.csv`, `summary.json` | Site comparisons, status, and all numerical results |
| `provenance.json` | Config, dependency versions, analysis-source SHA-256 |
| `input_manifest.json` | Relative paths, acquisition order, metadata, file/pixel hashes |
| `site_001/report.html`, `*.png` | Offline report and standalone figures |
| `site_001/summary.json`, `inputs.json` | Site metrics, units/crops, flags, input audit |
| `site_001/frames.csv`, `registration.csv` | Raw quality, shifts/exclusions, both domains' frame metrics |
| `site_001/local_registration.csv` | Sampled tile residuals and quality |
| `site_001/native_*.csv`, `aligned_*.csv` | Intensity bins, averaging, temporal/spatial ACF |
| `site_001/maps.npz` | Numerical means/std, difference maps, masks, spatial PSD |

Decoded-content hashes include full image shape and canonical numerical pixels
before ROI selection. Exact repeated content within a site is excluded after
its first included occurrence, preserving index gaps. Cross-site duplicates are
flagged in the overview. No training split is created: later splits must keep
all repeats of a site together and inspect cross-site duplicate groups.

One temporary disk-backed stack is processed at a time, normally
`frames * height * width * 4` bytes: about 2 GiB at 128×2048×2048. Float64 cache
values use eight bytes. Scratch is inside the site's output and is closed and
removed after success or a handled failure. Moment/map buffers still require
several hundred MiB at 2048², plus filesystem caching. Full-resolution maps are
not downsampled; bounded samples/crops are used for temporal, distribution,
spectral, and registration calculations. Disk also holds every site's maps.

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

- [scikit-image registration API](https://scikit-image.org/docs/stable/api/skimage.registration.html): subpixel translation and normalization behavior; this pipeline uses unnormalized correlation on smoothed/tapered copies.
- [NIST, Handbook of Frequency Stability Analysis](https://www.nist.gov/publications/handbook-frequency-stability-analysis): stability estimators; here the adjacent-block statistic is applied to measured intensity, not oscillator phase.
- [Foi et al., Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data (2008)](https://pubmed.ncbi.nlm.nih.gov/18784024/): signal-dependent raw-data noise models. This pipeline instead estimates empirical variance from repeats; the reference does not establish that an SEM PNG export follows that physical model.
