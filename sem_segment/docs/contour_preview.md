# Saved-image contour baseline preview

Gaussian + Otsu is also integrated into the main real SEM comparison workflow.
Use `tools/real_sem_compare.py --contour-method otsu` for normal HTML,
CSV/JSON and TensorBoard reports, including existing mask-area/ECD measurements
on closed regions. See [the main comparison guide](../../edge_denoise/docs/real_sem_comparison.md)
for the remote command. This standalone preview remains a segmentation-only
comparison and still calculates no ECD.

This preview compares saved uint8 images, stored current-method outlines, and
one ordinary segmentation baseline. It reads an existing `comparison.json` and
its image/contour assets. It does not load checkpoints, run inference, rerun the
current detector, or write to the source report. No new ECD is calculated.

From the repository root on the remote server:

```bash
python tools/preview_sem_contours.py \
  --config sem_segment/configs/contour_preview.yml \
  --from-comparison output/260921_real_n2n_comparison/comparison.json \
  --output-dir output/260921_real_n2n_contour_preview \
  --device cuda:0
```

Replace `--from-comparison` with the actual saved report and choose a new,
separate output directory. Open its `index.html` on that machine, keeping its
assets alongside it. There is no need to transfer images or logs to this PC.
Use `--device cpu` locally. The command requires the existing `[analysis]`
dependencies; CUDA detection uses the same optional CuPy installation described
in [the package guide](../README.md). No model weights are required.

## Detector

The reusable [config](../configs/contour_preview.yml) has exactly three settings:

```yaml
polarity: dark
sigma_px: 1.0
min_area_px: 25
```

Every source uses those same settings. The detector decodes the saved image,
applies the report's `segmentation_settings.input.crop`, and requires uint8
grayscale or identical RGB channels in that region. It smooths a float64 working
copy with a Gaussian (reflect boundaries, the library's default four-sigma
truncation). Zero sigma disables smoothing. The ordinary scikit-image Otsu
threshold is computed on that working copy for each image, using its default
256-bin float histogram. Dark foreground is `<= threshold`; bright foreground
is `> threshold`. A constant image has no separable foreground and returns an
empty mask, with its constant value recorded as the threshold.

All foreground components are labeled with four-connectivity before rejecting
those with fewer than `min_area_px` pixels. There is no candidate limit or
maximum-area filter. Marching squares traces the binary mask at 0.5 using
four-connected foreground. Interior loops remain; paths at the measurement
border stay open. The detector never pads, fills holes, splits connected regions,
fits circles, resamples, or refines boundaries. Crop offsets are restored to
full-image coordinates for display.

Gaussian smoothing, Otsu thresholding and four-connected component filtering run
on CUDA when requested. Only integer labels and scalar diagnostics return to CPU;
tracing uses the same scikit-image function as the CPU reference. CUDA events
record stage times and transfers. The standalone preview processes one image at
a time; the main comparison pipeline batches and overlaps these operations.
`cuda:N` refers to the scheduler-visible device N; GPU visibility is left alone.
CUDA errors are reported without silently switching to CPU. Sigma zero disables
smoothing only; CUDA thresholding and components still require CuPy. Metadata
records the actual Gaussian backend.

## Viewer and saved files

Select a site and source, then move the acquisition slider through every saved
image in that source. Slider positions are independent of acquisition IDs:
nonconsecutive IDs remain intact, and averages display their recorded ranges.
Older reports with `full_average` outside the series mapping also expose it as
`average128`.

The three panels show the same full saved image. Current outlines use cyan for
the stored mask (including holes) and orange for stored refinement, with dashed
fallback sections. Their recorded geometry is reused, including its original
border closure. Green shows the new Otsu mask outlines. Missing current contours
are explicitly labeled; a recorded zero-detection result is distinguished from
missing data. Native-resolution links open the original image or standalone SVG
overlays.

Original image files are copied byte-for-byte. SVGs embed those image bytes and
draw lines over them without altering the raster. For TIFF only, a lossless PNG
display copy preserves decoded uint8 pixels because browsers do not display
TIFF; the original TIFF remains linked. No brightness measurements or corrections
are made. Gaussian-smoothed pixels are never displayed or exported as the image.

`metadata.json` contains the source report/image references, one detector config,
crop, thresholds, component/outline counts, actual smoothing backend, and per-image
decode/filter/threshold/component/outline/render timings. `data.js` supplies the
same metadata to the viewer without a web server or network fetch. Keep the
whole directory together. The viewer adds no measurement dashboard or quality
score.

## Reproducible local example and checks

```bash
python tests/sem_segment/build_contour_example.py \
  --output-dir tmp/sem_contour_example
```

Open `tmp/sem_contour_example/preview/index.html` (sigma one), or
`tmp/sem_contour_example/preview_sigma0/index.html` (sigma zero). The example
contains nine saved images: raw acquisitions 9 and 29, two simulated output
variants of each, their two eight-frame block averages, and the uint8 average
of all 128 generated raw images. Every page is labeled synthetic; none of its
outputs comes from a trained model. The generator preserves the original seed,
noise draw order, and geometry. Fixture preparation stores current-method
outlines using the existing classical pipeline primitives on decoded saved
uint8 files; the preview then only reads them. Neither phase computes ECD.

Observed synthetic results with NumPy 2.4.4, SciPy 1.18.1 and scikit-image 0.26.0:

| Raw fixture | Stored current regions | Otsu regions, sigma 0 / 1 | Otsu loops/paths, sigma 0 / 1 |
| --- | ---: | ---: | ---: |
| Frame 9, high noise | 0 | 7 / 7 | 678 / 7 |
| Frame 29, false split | 8 | 7 / 7 | 11 / 7 |

The intended geometry has six complete disks plus one partial at the left edge.
Counts alone conceal the hundreds of internal noise loops in frame 9 at sigma
zero. The tests therefore also check boundaries: complete synthetic disks stay
within 0.65 px of the known radius, and ellipse normalized-radius error stays
below 0.04 at both sigma values. On the noisy fixtures with sigma one, each
complete outline has median radial error below 0.6 px and maximum below 2 px.
These bounds describe these fixtures, not real-data metrology accuracy.

The tests also preserve two counterexamples: sigma one erases a one-pixel-wide
protrusion, and direct Otsu on the smoothed planar step moves the binary outline
inward by one pixel. No alternate threshold or correction is introduced to hide
those effects. Concavity, annuli, connected shapes, thousands of preceding noise
components, large regions, border crossings, crop offsets, constant images,
uint8 enforcement, source-byte preservation, and acquisition indexing are covered.

```bash
python -m pytest tests/sem_segment \
  tests/edge_denoise/test_real_sem_compare.py \
  tests/edge_denoise/test_real_sem_viewer.py \
  tests/sem_noise/test_comparison_metrics.py -q
```

CUDA routing and error paths use mocks locally. Actual CuPy execution and real
SEM behavior still need the remote visual check. Use that comparison to decide
whether the intended interiors are found, real features are missed or merged,
and smoothing erases important geometry before selecting a production method.
