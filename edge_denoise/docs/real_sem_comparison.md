# Visual comparison of real SEM denoisers

Run on the remote server containing the data and checkpoints. The base config is
`edge_denoise/configs/sem_real_compare.yml`. Directory changes belong in CLI flags;
the default test site is `/data/260904_raw_data/test/260904_0947-13`.

All brightness, noise/temporal variation, registration, contour and metrology
measurements use **decoded saved uint8 images**. Raw RGB channels must be
identical. Predictions are restored to the checkpoint's intensity units,
range-checked, clipped/rounded to uint8, saved as lossless identical-channel RGB
PNGs, and decoded before analysis. Pre-export range flags describe image
production; intermediate floating-point predictions never supply measurements.

No brightness matching, gain/offset fitting, autocontrast, or geometric warping
is applied to the comparison images. Segmentation receives a fixed 0–255 scale
with contrast stretching disabled. Floating arithmetic and subpixel contour
coordinates are derived from the delivered uint8 pixels only.

## Compare different model pipelines

The workflow accepts single-frame `edge_denoise` checkpoints with different
objectives, representations, backbones and native tile sizes, including N2N,
Sobel/consistency fine-tunes, mean-target fine-tunes, and gradient/hybrid models.
All five methods for the next phase have real-data training paths; see
[the sequential suite](real_sem_next_phase.md). Burst-input fusion needs a
separate comparison and is rejected here.

Use named checkpoint arguments without editing YAML. `--only-checkpoints`
omits the six example arms from the base recipe. New names read registration
and brightness from each checkpoint and its verified prepared manifest; these
are never guessed from a folder name or applied to inference outputs.

```bash
python tools/real_sem_compare.py --only-checkpoints \
  --checkpoint n2n=runs/edge_denoise/260921_real_n2n_affine_percentile/ckpt_latest.pt \
  --checkpoint ft_noisy=runs/edge_denoise/260922_real_ft_noisy_affine_percentile/ckpt_latest.pt \
  --checkpoint ft_consist=runs/edge_denoise/260922_real_ft_consist_affine_percentile/ckpt_latest.pt \
  --site-dir /data/260904_raw_data/test/260904_0947-13 \
  --output-dir output/260922_real_models_comparison \
  --device cuda:0 --metrology-device cuda:0 --no-tensorboard
```

Run this after those checkpoints exist; replace the example dates with the
actual run names. In YAML, registration and brightness may also be omitted
to read the recorded treatment automatically. Explicit declarations are still
checked. All models must have the same original acquisition content, site
splits, channels and intensity normalization. Test-frame leakage, manifest
fingerprints, output shape and uint8 measurement provenance remain checked.
`arms.csv`, `arms.json` and Run details record each model's representation,
target, loss weights, consistency domain and crop size; `comparison.json` retains
its full recipe. Arm exports also retain initialization and training budgets.

Default `--split test` excludes both training and validation acquisitions.
For selection on a prepared validation site, use `--split val --site-dir PATH`
and a separate output directory. Every input must match recorded validation
content; this mode cannot accept training or unknown content. It is recorded
in the report settings and arm metadata. Rebuilds preserve the recorded split.

The **ECD · all models** chart starts with every model selected, independently
of panes A/B. Select a hole from its dropdown or click a matched contour. Each
model keeps one color; raw, average8 and average128 can be enabled separately.
Toggle mean ±3σ bands if useful. The per-hole table shows mean, sample SD, 3σ
and usable/total counts, sorted by 3σ. Missing measurements remain gaps and
fewer than two observations have no SD. Clicking a model's point opens its
saved acquisition in pane B. The all-hole summary uses the common holes across
all models and matches the exported statistics, regardless of chart visibility.
Compare measurement failures and mean ECD changes as well as precision.

Existing completed reports get these chart changes with the `--render-only`
command below; it reuses their saved measurements without inference or analysis.
This shows all models already in that report. To add other checkpoints, produce
a new comparison containing those named models.

## Rebuild an existing comparison first

Use the existing `comparison.json` and saved PNGs. This does not need the raw
dataset, checkpoints, or denoiser inference. Older reports need one new contour
pass because they saved outlines for only four matched holes.

```bash
python tools/real_sem_compare.py \
  --from-comparison output/260921_real_n2n_comparison/comparison.json \
  --output-dir output/260921_real_n2n_otsu_comparison \
  --contour-method otsu \
  --metrology-device cuda:0
```

This runs Gaussian + Otsu inside the main comparison pipeline: reference feature
IDs, raw images, average8, average128, and every saved model output all use it.
The result is the normal `index.html`, CSV/JSON measurements, and TensorBoard
report. It does not launch a separate preview or require inference. Use
`--metrology-device cpu` for a CPU installation.

New runs using the shipped `sem_real_compare.yml` select Otsu by default.
Rebuilds without a method override preserve the method saved in the input;
older reports without method metadata retain the existing method. Use
`--contour-method current` to select the previous segmentation/refinement
pipeline explicitly. Use `--otsu-config sem_segment/configs/contour_preview.yml`
to load the same three detector settings used by the standalone preview.

Replace the original report directory with the one already on your server.
The new output directory must not exist. Saved uint8 images are copied unchanged
and the original report is retained. Failed raw segmentation cannot prevent
viewing local model contours. Images or measurements missing from an incomplete
old run are reported explicitly; the command does not invent them.

After this first rebuild, changes to presentation can reuse the new measurements:

```bash
python tools/real_sem_compare.py \
  --from-comparison output/260921_real_n2n_otsu_comparison/comparison.json \
  --output-dir output/260921_real_n2n_otsu_comparison \
  --render-only --no-tensorboard
```

`--render-only` accepts a new output directory too. It does not estimate drift,
segment images, or load a model. It requires the revised report format.
Detector/device overrides require remeasurement and cannot accompany
`--render-only`.

This also upgrades existing reports with stable image navigation, per-hole mean
and 3σ ECD statistics, all-acquisition contour bands, and image colorbars. Saved
PNGs and measurements are reused unchanged. Reload `index.html` after rebuilding
(hard-refresh a browser tab that still has the older viewer cached).

For repeated contour experiments on a **completed** report, use:

```bash
python tools/real_sem_compare.py \
  --from-comparison output/260921_real_n2n_comparison/comparison.json \
  --output-dir output/260921_real_n2n_otsu_fast \
  --contour-method otsu --metrology-device cuda:0 \
  --contours-only --no-tensorboard
```

`--contours-only` recomputes the full-average feature IDs, every image's contours,
matching and metrology, while reusing stored brightness, temporal variation and
translation diagnostics. It validates that those measurements and their image
assets are present, and rejects incomplete reports. As with `--render-only`,
the saved images must still be the images described by the report. The source
report and its files are preserved. Omit this flag for complete reanalysis.
`--no-tensorboard` skips duplicate image encoding for TensorBoard; HTML, native
image links and CSV/JSON measurements are still produced. Both flags work without
checkpoints or inference. Use `--tensorboard` to enable TensorBoard again.

## Otsu detector and measurements

The default is dark foreground, Gaussian sigma 1 px, and minimum component area
25 px. These settings are identical for every source. Smoothing uses a float64
working copy of the decoded saved uint8 crop. Ordinary Otsu is recalculated for
each image. Four-connected foreground components are filtered only by minimum
area, with no candidate limit, maximum area, watershed, or hole filling.
Interior loops remain visible. The original saved/displayed image is unchanged.

The main report uses the existing polygon-area/ECD calculation on complete Otsu
masks, subtracting interior holes. No contour refinement runs in this mode.
Border-crossing paths remain open in both the HTML viewer and TensorBoard and
have no area/ECD. They are not closed for convenience. The viewer identifies
the active detector, settings and per-image threshold; refinement controls are
disabled for Otsu reports. These are mask measurements, not subpixel edge
measurements or demonstrated real-SEM accuracy. Synthetic checks show that
smoothing can erase thin geometry and shift a thresholded boundary.

The saved record adds `contour_method`, `otsu_settings`, per-frame
`otsu_threshold_dn` and `gaussian_backend`, and optional per-region `open_paths`.
Existing coarse/holes/refined fields keep their meanings, and old reports remain
readable. `segmentation_settings` retains crop, physical scale, metrology and
current-method settings; `contour_method` selects the active detector. The
standalone preview remains available as an additional visual comparison.

## Produce a new comparison from checkpoints

The six arms are affine/translation/none registration crossed with
percentile/none brightness matching during training. Every model receives the
same native test frames at inference. Training treatments are metadata and are
checked against checkpoints and prepared manifests.

Choose the date prefix used for the experiments. For example, the command below
looks for `runs/edge_denoise/260921_real_n2n_affine_percentile/ckpt_latest.pt`, and
the analogous five folders:

```bash
python tools/real_sem_compare.py \
  --config edge_denoise/configs/sem_real_compare.yml \
  --experiment-prefix 260921_real_n2n \
  --site-dir /data/260904_raw_data/test/260904_0947-13 \
  --output-dir output/260921_real_n2n_comparison \
  --device cuda:0 \
  --metrology-device cuda:0
```

The prefix is an example, not a discovered checkpoint date. Override older
baselines individually, without changing the YAML:

```bash
python tools/real_sem_compare.py \
  --config edge_denoise/configs/sem_real_compare.yml \
  --experiment-prefix 260921_real_n2n \
  --checkpoint translation_none=runs/edge_denoise/OLDER_DATE_real_n2n_translation_none/ckpt_latest.pt \
  --checkpoint none_none=runs/edge_denoise/OLDER_DATE_real_n2n_none_none/ckpt_latest.pt \
  --output-dir output/260921_real_n2n_comparison
```

Replace `OLDER_DATE` with each baseline's actual date. The default test site and
GPU settings come from the base config. Other useful flags:

| Flag | Purpose |
|---|---|
| `--runs-dir PATH` | Parent directory for experiment-prefix lookup. |
| `--checkpoint-name ckpt_best.pt` | Checkpoint filename inside each prefixed folder. |
| `--model affine_percentile` | Compare only this configured arm; repeat for several. |
| `--prepared-manifest NAME=PATH` | Supply the original manifest when a checkpoint's dataset moved. |
| `--site NAME` | Select one named site from a multi-site config; with `--site-dir`, name the overridden site. |
| `--segmentation-config PATH` | Crop, physical scale and metrology settings; also detector/refinement settings for `current`. |
| `--contour-method otsu` | Use Gaussian + Otsu in the main report; `current` selects the previous method. |
| `--otsu-config PATH` | Detector YAML with `polarity`, `sigma_px`, and `min_area_px`. |
| `--tile-batch N` | Inference tile batch size; default 32. |
| `--analysis-batch N` | Maximum images per CUDA Otsu batch; default 16. |
| `--analysis-memory-mb N` | Estimated batch working-set budget in MiB; default 8192. |
| `--io-workers N` | Saved-image decoding workers; default 2. |
| `--contours-only` | On rebuilds, reuse completed brightness/noise/drift analysis and remeasure contours. |
| `--no-tensorboard` | Produce the HTML/CSV/JSON report without TensorBoard image encoding. |
| `--difference-limit-dn 32` | One symmetric display limit for output-minus-raw images, in DN. |
| `--metrology-device cpu` | CPU Otsu/native analysis or current-method refinement. |

The six-arm base recipe remains the N2N preprocessing study. Comparisons check
normalization, native source content and train/validation/test splits. They reject
test acquisitions found in training/validation data. Checkpoint steps and EMA
selection remain explicit. Select checkpoints using validation data before
examining test sites.

## GPU execution

For delays after `contours/CD complete`, see the
[performance audit and resolution plan](real_sem_performance_audit.md). The
comparison now logs record serialization, summaries, individual exports,
viewer assets and TensorBoard encoding/flush, with 30-second stage heartbeats.
It writes a small `timings.json` beside `comparison.json`; final save durations
are included there. The audit includes full-run and render-only timing commands.

Checkpoint saves serialize only new contours into compact external parts.
Finalization assembles the compatible `contours.json` by copying those encoded
rows; it does not repeatedly convert earlier models' coordinates. Record and
contour writes use atomic replacement. Existing completed reports retain their
v3 format, and `--render-only` reuses their saved contour export. See the audit
for measured local serialization savings; no additional full comparison is
required solely to profile these changes.

The base config uses logical `cuda:0` for inference, native brightness/temporal
statistics, Gaussian smoothing, ordinary per-image Otsu thresholds, four-connected
component labeling and minimum-area filtering. CUDA execution uses the existing
optional CuPy dependency; install the wheel matching
the server's toolkit as described in [sem_segment](../../sem_segment/README.md).
CUDA errors are explicit, with no silent CPU fallback. Under Slurm, preserve
the scheduler's GPU visibility and use its logical device numbering.

Gaussian filtering batches images without smoothing along the acquisition axis.
Every image has its own 256-bin histogram and threshold. Labels and minimum-area
filtering stay on GPU; only retained int32 label maps and scalar diagnostics
return to CPU. Metrology reuses those labels instead of labeling the mask again.
One producer decodes and processes the next batch while the caller traces and
measures the previous batch. One batch is queued ahead of the consumer; `--analysis-batch`
and an estimated 96 bytes per pixel working-set budget bound their size. A single
image exceeding the budget fails explicitly. This is a batching estimate, not a
hard limit on CuPy's memory pool or concurrent inference allocations. Reduce the
batch/budget if other jobs or models occupy GPU memory.

The ordinary detector and float64 measurement arithmetic are unchanged. Sigma
zero disables only smoothing; CUDA thresholding and labeling still run. Native
statistics keep float64 Welford accumulators on GPU and return the final SD map
and frame summaries. OpenCV ECC, scikit-image contour tracing, polygon geometry,
image encoding and report writing remain on CPU. No new registration estimator
is substituted. `--contours-only` avoids repeating ECC when testing contours.

`comparison.json` records actual batch sizes, execution backends, CUDA-event
stage times, transfers and series wall time. Per-frame CUDA timings are amortized
batch costs; overlapping stage totals do not add up to wall time. TensorBoard
generation has its own elapsed time. The GPU implementation uses one selected
GPU; this command does not distribute work across all four H100s.

Validate the hardware path on the server without copying anything here:

```bash
python tools/benchmark_sem_analysis.py \
  --from-comparison output/260921_real_n2n_comparison/comparison.json \
  --device cuda:0 --frames-per-source 16 --analysis-batch 16 \
  --output-json output/260921_real_n2n_otsu_benchmark.json
```

This compares CPU reference and CUDA masks, outline coordinates, thresholds,
counts, ECD and native brightness/temporal statistics on saved images from every source. Sampling includes acquisitions
9 and 29 where available. It separates warm-up from timing, prints per-source
throughput and stage costs, and exits unsuccessfully on differences. It measures
decoding, detection and mask metrology; the separate native-statistics agreement
check is outside that timer, as are inference, ECC and report rendering.
`--otsu-config` selects a detector YAML; `--device cpu` exercises the command
locally. Local tests use numerical CuPy mocks; hardware speed and floating-point
agreement must be established by this remote check.

## Inspect the report on the server

Open the generated `index.html`, or serve that folder using your normal remote
browser/access arrangement:

```bash
python -m http.server 8765 --bind 127.0.0.1 \
  --directory output/260921_real_n2n_otsu_comparison
```

No transfer of acquisitions or copy/paste of server results to the development
PC is needed. The report also works as a local static folder in a browser that
permits local scripts; preserve its assets and image subfolders.

- Choose sources A and B: raw, any model, sixteen nonoverlapping average8 images,
  or the single average128 reference. Start with the full field, use 1:1 pixels,
  scroll to zoom and drag to pan. Both views share native coordinates.
- Frame sliders replace images in the same viewport. With linking enabled, raw
  acquisition `i` selects model acquisition `i` and its containing average8
  block. Moving a block slider selects its first raw/model acquisition; each of
  the eight individual acquisitions remains accessible. Labels always identify
  the exact range. Average128 is static.
  The displayed pair stays visible until both requested images are decoded and
  their contours are ready; images, outlines, labels and measurements switch
  together without a blank frame or fade. Rapid requests are coalesced to the
  latest position. A missing image keeps the previous pair and reports the error.
  Zoom, pan and wipe reuse the displayed raster. The decoded-image cache is
  limited to 64 MiB, in addition to the displayed pair and current load; the
  viewer does not preload an entire sequence.
- The overlapping wipe has a separate divider slider. Each image and its own
  contour overlay are clipped together. Contours can be hidden, mask-only,
  refined-only, or both for the current method. Otsu has mask-only or hidden
  contours because no refinement is performed.
- Clicking a region highlights the measured area and fits the complete hole
  with padding. The small full-field view shows its location. ECD, area, status,
  refined coverage and an acquisition-linked diameter trace explain each number.
  The ECD plot includes separate A/B means and mean ±3 sample-SD bands, with
  a table of usable/total observations, mean, σ, 3σ and interval endpoints in the
  report's px or nm units. Statistics use the selected hole's valid observations;
  missing values are excluded and fewer than two observations have no SD.
  These bands describe observed variation, not confidence intervals or accuracy.
- Expand **All-acquisition contour band** and choose a source to overlay every
  saved outline, colored by acquisition/block center. The Measurement selector
  chooses mask or refined boundaries. Unmatched regions and interior rings are
  included; partial paths stay open and failed refinement stays dashed. Native
  coordinates preserve motion. **Open vector plot** opens a standalone SVG with
  an acquisition colorbar that preserves subpixel vertices when enlarged. The
  single average128 reference has no temporal band.
- Brightness tracks directly compare native output and raw means. The paired
  difference retains any brightness bias. Image differences use a common signed
  DN scale; saturation affects the display only. Differences may include removed
  structure and are not a ground-truth noise estimate.
  Each main pane and the overview have labeled intensity colorbars (0–255 DN).
  Difference panes use the saved symmetric DN limit, with blue = negative,
  white = zero, red = positive; raw/average panes retain their intensity bars.
  Temporal variation maps have a shared 0-to-maximum sample-SD colorbar in DN.
- Expand geometry/temporal variation or coverage when needed. Detailed warnings,
  treatment identities, CSVs and JSON stay in the details section.

## Measurement meaning and failures

`ECD = 2 * sqrt(A / pi)`, where `A` is polygon area minus interior rings. ECD is
area-equivalent diameter, not a horizontal width or a fitted-circle diameter.
Multiply by `pixel_size_nm` for nm. The comparison always labels this definition
as ECD, regardless of other CD definitions available in standalone metrology.

For the current method, refined polygons retain coarse vertices where refinement fails. Those spans are
drawn dashed; successful refined segments are solid. Insufficient refinement has
no accepted refined ECD. A decrease in sampled gradient strength marks a region
for inspection, not as proof of worse physical accuracy. Border regions are
visible but excluded from complete-hole metrology.

The uint8 average of all 128 raw images supplies hole IDs and the common drift
reference. It is not ground truth. Each frame is segmented independently; model
measurements never inherit the reference's boundary. If the reference has no
complete holes, local contours and their areas remain inspectable while
cross-frame identity/repeatability is unavailable. Registration estimates move
coordinates only for matching; images and displayed local contours stay native.

Raw images, average8 and models expose separate measurement coverage. The viewer
compares common holes of its selected models; export summaries use common holes
across all models. Raw failure does not suppress model-only summaries. Sample SD
is calculated within each hole with at least two usable observations, then
summarized across contributing holes. Missing values are gaps, never zeroes.
Average128 has one observation and no repeatability estimate. Native temporal
variation includes motion and charging as well as noise; neither low variation
nor low ECD SD alone establishes the best denoiser.

The standalone `sem_noise` acquisition workflow retains its original correction
diagnostics. This comparison does not invoke it. A legacy `analysis_config` is
accepted for `registration_sigma` and `frame_interval_s`; its acquisition
correction settings are not used. All comparison brightness/temporal statistics
use the full saved field; a segmentation crop affects contours only.

## TensorBoard

```bash
tensorboard --logdir output/260921_real_n2n_visual_audit/tensorboard_comparison \
  --samples_per_plugin images=128
```

The separate comparison log contains native image and full-contour sequences.
The samples flag retains all 128 acquisition images per tag; TensorBoard's
default image sampling would show only a subset.
Their event steps are acquisition numbers; average8 uses the first acquisition
of each block and average128 uses zero. Summary metrics use checkpoint steps in
separate tags. The HTML viewer provides linked frame controls, wipe, clickable
regions and exact measurement status. Existing training logs are untouched.

For future training, `training.real_comparison_images: true` retains the
lightweight fixed train/validation panels at optimizer steps: input, uint8
prediction, raw average8 and full average. They do not use test images or run
offline contour analysis at every validation interval.
