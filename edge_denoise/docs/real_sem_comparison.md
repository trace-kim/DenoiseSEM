# Visual comparison of real SEM denoisers

For precomputed average2/average4 inputs (64/32 images per site), use
[the block-dataset command and directory overrides](real_sem_block_datasets.md).
The comparison reads their provenance automatically and uses the actual input count.

Run on the remote server containing the data and checkpoints. The base config is
`edge_denoise/configs/sem_real_compare.yml`. Directory changes belong in CLI flags;
the default test site is `/data/260904_raw_data/test/260904_0947-13`.

The B1 leave-one-out target GPU equivalence check, timing pilot and both resumable
training commands are in [the next-phase handoff](real_sem_next_phase.md#b1-leave-one-out-targets-precision-wp1wp2).

All brightness, noise/temporal variation, registration, contour and metrology
measurements use **decoded saved uint8 images**. Raw RGB channels must be
identical. Predictions are restored to the checkpoint's intensity units,
range-checked, clipped/rounded to uint8, saved as lossless identical-channel RGB
PNGs, and decoded before analysis. Pre-export range flags describe image
production; intermediate floating-point predictions never supply measurements.

No brightness matching, gain/offset fitting, autocontrast, or geometric warping
is applied to single-frame comparison images. Explicit registered multi-frame
series are newly produced images with recorded transforms; the source PNGs stay
unchanged. Segmentation receives a fixed 0–255 scale
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

New runs using the shipped `sem_real_compare.yml` select `otsu_refined` by default.
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

## Precision: refined ECD and variation components (WP3/WP4/WP6)

`otsu_refined` retains Otsu region IDs and mask ECD, then refines each outer ring
along its normals against the **decoded, unsmoothed uint8 crop / 255**. The CPU
and CUDA Otsu producers retain that crop, so refinement needs no second decode.
One CUDA refiner is reused per series. Failed refinement is reported with its
coverage; it never becomes an accepted mask-only refined measurement. Interior
rings retain their mask geometry, as in the existing metrology implementation.

`--refine-estimator gradient_peak|threshold|erf` records the estimator in the
report/settings and detector label. `gradient_peak` is the initial default,
not a validation-selected winner. The existing CuPy refiner accelerates
`gradient_peak`; `threshold` and `erf` require explicit `--metrology-device cpu`.
Unsupported CUDA estimators fail rather than silently falling back. Select the
estimator on a **validation** site using coverage and failure rate, not lowest SD.

From an already completed validation comparison, run all three candidates:

```bash
VALIDATION_REPORT=output/real_models_validation/comparison.json
for estimator in gradient_peak threshold erf; do
  metrology=cpu
  if [ "$estimator" = gradient_peak ]; then metrology=cuda:0; fi
  python tools/real_sem_compare.py --from-comparison "$VALIDATION_REPORT" \
    --output-dir "output/260928_validation_${estimator}" --contours-only \
    --contour-method otsu_refined --refine-estimator "$estimator" \
    --metrology-device "$metrology" --no-tensorboard || break
done
python tools/benchmark_sem_analysis.py \
  --from-comparison output/260928_validation_gradient_peak/comparison.json \
  --device cuda:0 --frames-per-source 16 --contour-method otsu_refined \
  --output-json output/260928_refined_benchmark.json
```

The benchmark includes refinement, compares its ECD/coverage as well as masks,
and reports refinement stage time. H100 speed and real-data agreement remain
unverified until it runs remotely. Existing `otsu` and `current` methods retain
their behavior. Rebuilds keep their recorded method/estimator unless overridden.

Per-hole exports now include `cd_std_detrended` (linear acquisition-order trend,
residual variance with n-2 degrees of freedom), `cd_std_successive` (sample SD
of consecutive differences / sqrt(2)), and correlations with native frame mean
and the Otsu threshold. Block averages use block centres. Fewer than three
usable observations yield unavailable components; differences never bridge a
failed observation. Constant brightness/threshold makes correlation unavailable.
The old sample SD is unchanged. Summary exports include medians of all three
SDs, observations per hole and deterministic 95% percentile bootstrap intervals
over holes (2,000 resamples; fewer than two holes gives no interval). These
intervals do not remove temporal dependence or establish dimensional accuracy.

The viewer starts refined reports on refined ECD, offers both boundary toggles,
and displays the components/intervals. A single pixel-centre-to-raster helper
places contours, labels and fit boxes consistently; stored geometry and PIL
overlays are unchanged. `--render-only` refreshes that display on old reports.

WP3/WP4/WP6 verification (2026-09-28): full `python -m pytest -q`:
**1,058 passed, 1 skipped** in 159.25 s; the optional browser test is skipped.
An additional 9 viewer checks pass. Synthetic saved disks recover diameter within
0.05 px for each CPU estimator; mocked CUDA agrees with CPU gradient refinement.

## Registered baselines and frames versus precision (WP5/WP8)

Every new comparison adds `average8_registered` beside the unchanged raw
`average8`. `--average-frames 2,4,8` requests raw and registered averages for
each K; the default remains 8. Counts must be unique integers from 2 through 64.
Groups are consecutive and nonoverlapping. Incomplete trailing groups are
omitted from that average series, logged, and counted in `remainder_frames`;
the original single-frame series retains every acquisition. Rebuilds preserve
previously saved average series even when new counts are requested.

Registration uses training's translation-seeded affine ECC at sigma 1, with
the first usable frame in each block as anchor (normally frame 1). Patternless
or failed frames stay in the average at native coordinates, with a recorded
reason. A failed fit never changes the next translation seed. Source rectangles
are warped in bounded batches on `metrology_device`; float64 sums are rounded
once to saved uint8 PNGs. Saved support masks mark the common cubic footprint.
Outside it the PNG retains the ordinary native mean for display, and regions
touching unsupported pixels are excluded from ECD. Matrices, statuses, anchor,
CPU geometry time and device warp/average time are stored per block.

`--average-model NAME` explicitly names the model selected on validation data.
It builds `NAME_average2`, `NAME_average4`, etc. by registering the **saved
denoised** frames and averaging them. No raw-derived transform or brightness
correction is applied to these model averages. This is deployable averaging of
single-frame predictions; it does not implement learned burst fusion.

```bash
REPORT=output/real_models_comparison/comparison.json
python tools/real_sem_compare.py --from-comparison "$REPORT" \
  --output-dir output/260928_real_frames_precision --contours-only \
  --contour-method otsu_refined --refine-estimator gradient_peak \
  --average-frames 2,4,8 --average-model n2n \
  --metrology-device cuda:0 --no-tensorboard
python tools/check_real_sem_averages.py --from-comparison "$REPORT" \
  --output-dir output/260928_average_equivalence --average-frames 8 \
  --source raw --device cuda:0 --cpu-threads 2
```

Use the validation-selected estimator and model name in these examples. The
same averaging flags work on fresh comparisons. Full and contour-only rebuilds
can add missing averages using only saved PNGs. Contour-only mode logs additions
and analyzes those new images while reusing existing brightness/noise/drift.
Render-only mode never creates or measures new averages.

The viewer gives each series a toggle/color and links blocks to their original
acquisitions. `frames_vs_precision.csv` in each site and the curve in the viewer
report K, group count, common holes, observations per hole, median ECD 3σ and
bootstrap 95% intervals. Each family uses the intersection of measurable holes
across its estimable K values. A K with no measurable holes remains an explicit
gap; raw topology failures do not suppress the measurable average points or
fabricate K=1 precision. The contributing K series are exported. Original per-series
coverage remains in `repeatability.csv`. Neither K nor learned fusion is chosen
automatically. GPU throughput and CPU/GPU agreement on the remote acquisitions
remain hardware checks, including the CPU ECC stage and image I/O.

WP5/WP8 verification (2026-09-28): full `python -m pytest -q`:
**1,062 passed, 1 skipped** in 221.99 s. Synthetic drifting bursts become sharper;
blank acquisitions remain included, saved source PNGs stay byte-identical, and
raw-average refined ECD variation decreases over K=1,2,4,8. Rebuild, reuse,
remainder handling and linked registered-block navigation are covered.

## Single-frame template precision diagnostic (WP7)

Add `--template-limit` to a fresh comparison or saved-image rebuild. This is
opt-in and requires a refined detector (`otsu_refined` or `current`). It uses
each matched hole's refined full-average ECD and a five-pixel annulus around
that hole in the **decoded saved uint8 full average**. Raw observations also
come only from decoded saved PNGs. Fits are batched across holes and frames on
the metrology device, with bounded scratch memory and an explicit CPU path.
Patternless inputs skip fitting; failed fits remain visible in coverage counts.

The fixed-orientation model is `I(x) = g*T(c + (x-c-d)/s) + o`, with free
translation `d`, physical diameter scale `s`, gain `g` and offset `o`. Thus
`ECD = s*ECD_template`. The inverse sampling scale corrects an inconsistency
in the handoff equation: sampling `T(s*x)` would instead give diameter divided
by `s`. Gain and offset are nuisance fit parameters, never image corrections.
No diagnostic image is exported or substituted for a model's native output.

The viewer adds a **single-frame template limit — diagnostic, not deployable**
summary row and per-hole 3σ, detrended SD, successive SD and usable fit counts.
The same observations flow through `observations.csv`, `per_hole.csv` and
`repeatability.csv`. `template_fits.csv` adds scale/shift/gain/offset, fit status,
scale/ECD standard errors, and the five-parameter covariance. The report records
the template SHA-256, annulus width, convention, device and elapsed time.

Standard errors use a Gauss–Newton residual sandwich covariance (HC1), allowing
independent heteroscedastic pixel noise. They are conditional on the template.
Template blur, inclusion of the raw frame in the average, correlated SEM noise
and real shape changes can bias this diagnostic; it is not a universal lower
bound or a deployable single-frame result. Compare coverage and the same holes
before interpreting a model's distance from it.

```bash
REFINED_REPORT=output/260928_real_frames_precision/comparison.json
python tools/real_sem_compare.py --from-comparison "$REFINED_REPORT" \
  --output-dir output/260928_real_template_precision --contours-only \
  --template-limit --metrology-device cuda:0 --no-tensorboard
python tools/check_real_sem_template.py \
  --from-comparison output/260928_real_template_precision/comparison.json \
  --frames 32 --device cuda:0 --cpu-threads 2 \
  --output-json output/260928_template_equivalence.json
```

These commands preserve the recorded refinement estimator; use explicit CPU
metrology for `threshold`/`erf`. The check includes image decoding and all holes
in the first 32 saved frames, checks failure-status agreement and parameter/
standard-error differences, and fails if there are no comparable valid fits.
It never changes the source report. A subsequent `--render-only` reuses the
diagnostic. `--no-template-limit` disables it on a remeasurement.

Synthetic calibration uses 320 realizations each of Gaussian and combined
Poisson/Gaussian noise on saved-equivalent uint8 disks. Scale bias is within
three Monte-Carlo standard errors, and mean fitted standard error is within
10% of empirical SD. Physical scale/sign, nuisance parameters, patternless
skips, report integration, export and render-only reuse are covered. H100 timing
and agreement on the real acquisitions remain remote checks.

Final WP7/report audit verification (2026-09-28): full `python -m pytest -q`
passed **1,071 tests**, with **1 optional offline browser test skipped** and
7 existing warnings, in 156.26 s. The audit also covers unavailable raw points
in the K curve, refined contours outside registered common support, and the
remote mean-target checker using the unchanged factory RNG state.

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
