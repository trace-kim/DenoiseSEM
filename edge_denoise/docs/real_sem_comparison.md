# Compare six real N2N registration/brightness treatments

Run on the server containing the data and checkpoints. Edit
`edge_denoise/configs/sem_real_compare.yml`: replace its six example checkpoints
and test-site folders with the actual paths. All six runs use the
`sem_real_n2n.yml` objective; the experiment varies training target matching:

| Arm | Training registration | Training brightness |
|---|---|---|
| affine_percentile | Affine | Percentile |
| translation_percentile | Translation | Percentile |
| none_percentile | None | Percentile |
| affine_none | Affine | None |
| translation_none | Translation (existing baseline) | None |
| none_none | None (existing baseline) | None |

These are **six preprocessing treatments of the same real N2N recipe**. The
comparison does not mix synthetic, mean-target, gradient-loss or consistency
recipes. No retraining is needed. See [interpretation and next steps](real_sem_comparison_findings.md)
for the reported plateaus, validation spikes, and the corrected experiment plan.

Each checkpoint entry declares the training treatment explicitly:

```yaml
checkpoints:
  affine_percentile:
    checkpoint: runs/edge_denoise/sem_real_n2n_affine_percentile/ckpt_latest.pt
    registration: affine
    brightness: percentile
  translation_none:
    checkpoint: runs/edge_denoise/sem_real_n2n_translation_none/ckpt_latest.pt
    registration: translation
    brightness: none
    prepared_manifest: data/SEM-real-translation/real_dataset.json
```

Inline matching settings are checked against the checkpoint. For old baselines,
the original prepared manifest supplies registration; missing `real_matching`
does **not** imply no registration. When `prepared_manifest` is omitted, it is
located using the checkpoint's dataset directory (or its `burst/` subfolder).
Set the path explicitly when data have moved on the server. The manifest hash
must match the checkpoint's dataset fingerprint when present; older checkpoints
without that fingerprint receive a visible linkage warning. Only metadata are
read here, not the training arrays.

The command checks that architecture, N2N objective, normalization, image size,
raw acquisition content and train/validation/test splits agree across arms.
Prepared manifests can differ because of registration; their underlying raw
content and splits must match. Test acquisitions found in a training/validation
split are rejected. One arm or several checkpoints of the same treatment can
also be compared using distinct labels. Reported checkpoint steps and EMA
selection remain explicit. Select checkpoints using validation sites before
the test comparison; do not tune their selection on test results.

```bash
python -m pip install -e '.[dev,analysis]'
python tools/real_sem_compare.py --config edge_denoise/configs/sem_real_compare.yml --site site_01
```

Inspect the pilot's `index.html`, `comparison.json`, and measurement counts
before expanding. Choose a **new output directory**, then omit `--site` to run
every configured site. Relative paths resolve from the repository root on
Windows and Linux. The command does not launch training. Real-data piloting
requires server access; local regression tests generate their own pixels.

Each site has 912 quantitative images with six checkpoints: 128 raw frames,
16 block means and 768 predictions. Noise reporting is followed by contour/CD
measurement on each series. Older versions printed `writing report` and then
ran that measurement pass silently; a long pause at that message could therefore
be contour processing. Progress now distinguishes noise/report completion,
contour/CD frame counts, combined rendering and TensorBoard export. Per-series
`timings_s`, `segmentation_stage_totals_s` and each frame's
`segmentation_timings_s` are saved in `comparison.json` with the analysis results.

Contour processing shares the image's gradient and cubic spline preparation
across holes, and keeps one segmentation backend loaded per series. Profile
peak selection, neighbour consistency and chord measurements run in
batches; each polygon's convex hull is computed once. These retain the same
nearest-peak rule (including prominence, plateaus and ties), rejection masks
and measurement definitions. This CPU path needs no extra dependency.
`device: cuda` selects the denoiser; GPU contour refinement is enabled separately
as described below. Pixel
resolution, interpolation order, both contour methods and sample counts are
unchanged. This reduces repeated computation; it does not remove the cost of
eight noise analyses and inference for all checkpoints. An already running
process will not acquire this optimization. Runs do not currently resume;
use a new `output_dir` when starting another comparison.

### Single-GPU contour acceleration

On the server, install CuPy for its CUDA Toolkit. For CUDA 12.x:

```bash
python -m pip install 'cupy-cuda12x>=13.4,<15'
```

For CUDA 13.x use `cupy-cuda13x>=14,<15` instead; install only one CuPy wheel.
These wheels expect a working matching Toolkit, including runtime compilation
headers. See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
for managed CUDA installations and troubleshooting. No SAM weights are needed
for the default classical backend.

Select one visible GPU for both inference and analysis. Outside a scheduler:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_sem_metrology.py --device cuda:0
CUDA_VISIBLE_DEVICES=0 python tools/real_sem_compare.py --config edge_denoise/configs/sem_real_compare.yml --site site_01 --metrology-device cuda:0
```

For Slurm, request one GPU and keep the scheduler's `CUDA_VISIBLE_DEVICES`;
use logical `cuda:0` in the commands. Do not replace a scheduler-assigned device
list with a physical index. Set a fresh `output_dir` before starting the comparison.
Alternatively set `metrology_device: cuda:0` in the comparison YAML. `null`
respects `segmentation_config`; the default remains CPU. Use `--metrology-device cpu`
to compare with the CPU implementation.
Automatic inference/mask devices resolve to the chosen refinement GPU. Explicit
GPU settings that conflict are rejected, keeping the comparison on one device.

The benchmark needs no checkpoints. It warms up both implementations, checks
CPU/GPU agreement, and reports complete `segment_image` wall time including
transfers and CPU work. Its default is a quantized 1024×1024 array of 64 holes.
For a representative native image from an existing run:

```bash
python tools/benchmark_sem_metrology.py --device cuda:0 --image output/sem-real-comparison-pilot/site_01/raw/frame_001.png --repeats 5
```

Repeat `--image` to include an eight-frame mean and model output. If the
comparison has a `segmentation_config`, pass that same file as `--config` to
the benchmark; crops are respected. This benchmark uses classical masks.
`output/sem-metrology-benchmark.json` contains warm-up costs, individual times,
median stage times, speedup, and parity results. A label/rejection-mask mismatch,
missing measurements, or contour/ECD/axis difference above 0.000001 px returns
a nonzero exit. Check parity on representative saved images before a full run.
Mocks validate the local implementation; real CuPy kernel parity and H100
speedup must be measured on the server.

GPU processing uses float64 cubic spline coefficients, normal-profile sampling,
Gaussian derivatives, nearest qualifying peaks and parabola fitting, plus the
image-gradient diagnostic. Profiles from different holes share bounded batches;
the saved image is uploaded once. Adaptive search radii, native pixels, peak
prominence and rejection rules are retained. Mask detection, contour tracing,
coherence/gap handling, polygon geometry, registration and report rendering
still use the CPU. Speedup therefore depends on the stage breakdown; small
images may not benefit. This does not accelerate denoiser inference or the
independent noise pipeline.

The selected device and per-stage seconds/image are saved in the combined
report, JSON/CSV and TensorBoard. CUDA errors fail explicitly, with no silent
CPU fallback. The CUDA path currently supports `gradient_peak` only; `threshold`
and `erf` remain available on CPU. Each series owns one CUDA device, stream and
memory pool, released at its end. No image pixels are cached between frames and
no other GPUs are used by this refiner. This ownership allows future independent
image workers without introducing multi-GPU execution now.

For unusually large profile buffers, set `refine.cuda_batch_samples` in the
optional segmentation YAML; the default is 1048576 samples per batch. Lowering
it reduces temporary memory, at the cost of more batches. It does not change
profile spacing or image resolution.

Each site is a flat folder of exactly 128 naturally ordered images, with
consistent dimensions and uint8 storage. RGB inputs must have exactly equal
decoded channels; one channel is selected without luminance conversion. JPEG
compression remains part of the observed signal. Provenance records acquisition
order, source file and decoded pixel hashes, checkpoint hash/step, fixed
normalization, EMA selection, tile settings, and resolved analysis settings.
Sites are already held out; this command never splits data.
All checkpoints receive the same native test inputs. Declared training
registration and brightness settings are provenance, not inference corrections.
The treatment table is exported in `arms.csv`/JSON, included in the combined
noise report, and logged to TensorBoard alongside the metrics.

The sixteen means use acquisitions 1–8, 9–16, ..., 121–128. Accumulation is
float64, with one `np.rint` rounding. No registration or brightness correction
precedes averaging. All saved means and predictions are lossless uint8 PNGs
with identical RGB channels. The separate full acquisition average is **visual
reference only**, excluded from every quantitative calculation, including the
correspondence template and registration reference.

Inference disables the patch clamp through tile blending, restores checkpoint
intensity units, and audits the final prediction before rounding or clipping.
Every excursion below 0 or above 255 is counted without tolerance; values
exactly at the boundaries are valid. Flagged images are clipped, rounded, saved
and retained in analysis. The combined report and affected model/site sections
display this warning with filenames, extrema, counts and percentages:

> **Prediction outside the uint8 intensity range. Review the model training and output scaling. The measurements below use the clipped uint8 image and may be affected by clipping.**

The same flags and statistics persist in `prediction_ranges.csv` and JSON.
NaN/infinity aborts with a failed prediction in the audit and a nonzero exit;
no image is fabricated. Partial artifacts remain for diagnosis. A noise-analysis
failure produces an explicit partial-failure report and nonzero exit.

Optional `analysis_config` and `segmentation_config` use existing package YAMLs.
Noise uses the existing standard `sem_noise` defaults, including affine and
brightness diagnostics, with the **same analysis settings for every arm**.
An existing analysis YAML can override those settings uniformly. Independent
translation tracks use the existing ECC estimator against the same first
eight-frame mean: raw frames, block means, and each model's saved outputs.
Output-minus-raw drift and fit failure counts reveal changes in output geometry;
these are diagnostics, not ground-truth accuracy. Output estimates never replace
the raw translations used for correspondence. Metrology always uses native
saved pixels. Duplicate
decoded observations are retained and identified, so quantization or clipping
cannot silently remove observations. Standalone noise commands retain their
deduplication default. As documented in sem_noise, distribution masks may
exclude boundary pixels; entire clipped images remain in the analysis.

Segmentation defaults to classical dark-hole detection with gradient
refinement. Existing configs can select other backends and their normal weight
requirements. Measurements use saved DN divided by 255; segmentation input
black/white settings are overridden to 0/255. Optional input crops are native
slices. Both coarse and refined measurements are retained, with explicit
failure statuses when refinement has too few usable vertices.

The first eight-frame average defines correspondence, not truth. Model outputs
reuse their raw acquisition's translation. Unique nearest-centroid matches use
a gate of 45% of median template nearest-neighbour spacing. A supplied
`match_gate_px` must remain strictly below half that spacing. With one hole,
the default is 10 px and a warning identifies the absent spacing estimate.
With no template holes, there is no repeatability estimate. Missing/ambiguous
matches, registration failures, border regions and unusable refinements are
explicit; detected unmatched regions are counted per frame. Translations move
only coordinates for matching and overlays, never measurement pixels.

Per-hole CSVs include ECD (primary CD), major/minor axes, valid/failed counts,
mean, sample SD (`ddof=1`), and `3σ`. Fewer than two valid observations gives no
SD. Comparison medians use holes with at least two valid measurements in all
series, separately for each contour method. Different holes' dimensions are
never pooled as repeatability. Plots retain charging trends; no temporal
detrending, PSNR or SSIM is applied. Dimensions default to pixels;
`pixel_size_nm` converts them to nm while registration remains in pixels.
`contours.json` contains coordinates for four fixed overlay crops; dimension
observations retain every template hole.

Set `frame_interval_s` for uniform timing, or provide all 128 explicit times:

```yaml
sites:
  site_01:
    source_dir: data/SEM-test/site_01
    timestamps_s: [0.0, 0.5, 1.0]  # Example only: supply all 128 values.
```

Times must be finite and strictly increasing. Block times are means of their
eight acquisition times. Without timing, plots use acquisition positions and
block centers (4.5, 12.5, ...). The noise manifest preserves times; irregular
timing follows sem_noise's existing diagnostics.

The report includes sixteen panels: each block's first raw acquisition, its
predictions and its eight-frame average. Display limits are 0–255 and native
PNG links are available. Brightness/drift tracks, per-hole CD trajectories,
fixed crops with translated contours, measurement counts, clipping tables and
detailed noise reports accompany the panels. Completed runs reuse the figure
PNGs and `metrics.json`/CSV table for a separate TensorBoard directory:

```bash
tensorboard --logdir output/sem-real-comparison-pilot/tensorboard_comparison
```

Models log at checkpoint steps; baseline scalars use step zero. Training logs
are untouched. Pre-clipping extrema, counts, fractions and warning text are
included.

For future real training, opt in in `sem_real_n2n.yml` or another real recipe:

```yaml
training:
  real_comparison_images: true
```

At existing validation intervals, up to four fixed train and four validation
examples show input, uint8 prediction, raw eight-frame mean and full average
(visual reference only). Native uint8 crops and means are precomputed without
pair-sampling randomness; sites need at least eight frames. Range violations
and nonfinite prediction failures are logged before display clipping. Model
mode and torch RNG state are restored. Test pixels never enter live panels;
full metrology remains offline. The default is false, with no objective or
sampling changes.
