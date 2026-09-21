# Real SEM comparison: audit and focused revision plan

The report should let a reader see what each model did to the same acquisition:
what noise disappeared, whether brightness and geometry were preserved, and
whether a reported diameter follows a plausible boundary. Images come first;
numbers support an inspection the reader can reproduce.

This records the audit and the revision now implemented. Historical findings
below refer to the audited commits, not the replacement. See
[the current report guide](real_sem_comparison.md) for remote commands, saved
uint8 measurement rules, viewer controls, and rebuilding an existing report.

**Scope and evidence.** Audited at `daf14d6`, covering the comparison feature
`4a836c7`, progress/CPU changes `bdd061f` and `bb7e92d`, and CUDA change
`daf14d6`, plus the acquisition-analysis and segmentation code they call.
The remote acquisitions, checkpoints, and reported failing HTML were not
available for this audit. Findings below distinguish code behavior from possible
explanations for those particular segmentation failures.

## What the code actually does

| Finding | Evidence | Consequence |
|---|---|---|
| Model outputs enter the acquisition-correction workflow. | `tools/real_sem_compare.py:348` calls `analyze_dataset` for every series; `AnalysisConfig.registration` defaults to `affine`; `sem_noise/pipeline.py:181` invokes pair and acquisition brightness diagnostics. | The report asks how one output can be corrected to another output/reference, instead of whether that output preserves its corresponding raw input. |
| The diagnostic brightness corrections are real calculations, but do not overwrite the model PNGs. | `sem_noise/pair_diagnostics.py:61` and `:71` create `gain * frame + offset` arrays, plot their means, and save example diagnostic NPZs. The main noise moments use translation-only copies (`sem_noise/pipeline.py:249`); comparison metrology reads saved PNGs (`tools/real_sem_compare.py:283`). | Remove these calculations from the comparison path, not merely their captions. It would be inaccurate to claim that every stored prediction or every metric has been gain-corrected. |
| Predictions are already quantized before comparison measurements. | `edge_denoise/uint8_output.py:18` restores checkpoint intensity units, audits range, then rounds/clips to uint8 at `:30`. `tools/real_sem_compare.py:550` computes brightness from that exported image. | Preserve this required contract: analysis measures delivered uint8 pixels. Intermediate floating-point predictions must never supply metrology. Range flags describe image production only. |
| Segmentation has another contrast transformation. | `sem_segment/config.py:110` defaults to a 1st–99th percentile stretch; `sem_segment/pipeline.py:238` passes that copy to segmentation. Refinement samples the unstretched measurement image. | Saved pixels are preserved, but mask detection sees an adjusted copy. The comparison should disable this automatic stretch under the requested no-correction contract. |
| The report prints internal metrics rather than selecting meaningful questions. | `sem_noise/comparison_report.py:26` collects every numerical mode statistic; `:216` renders the resulting list. Different series independently choose low-gradient masks (`sem_noise/metrics.py:227`). | Large tables are difficult to interpret, and identical settings do not guarantee identical measurement support between models. |
| Most acquisitions cannot be inspected through the comparison figures. | `sem_noise/comparison_report.py:191` shows only the first acquisition of each eight-frame block and its predictions. | The other 112 predictions per model are saved, but lack a usable sequence viewer. |
| Full-image contour evidence is discarded. | `tools/real_sem_compare.py:330` retains outlines only for matched holes 1–4. The renderer combines all their frames on the first image and caps crop radius at 32 pixels (`sem_noise/comparison_report.py:234`). | Large holes are cut off. Unmatched/border detections and failures cannot be inspected. A displayed outline is not necessarily from the displayed acquisition. |
| A noisy correspondence template can disable all later comparisons. | `tools/real_sem_compare.py:474` segments the first eight-frame mean; only non-border regions become template holes. Empty templates produce no observation rows. | Good model segmentations can still have no matched CD results. Missing measurements need an explicit unavailable state, not empty plots or misleading zero counts. |
| Raw segmentation failure can suppress valid model summaries. | `sem_segment/repeatability.py:85` intersects usable holes across raw, average8, and every model. | One unreliable series can empty the common set for all models. Coverage and model-to-model comparisons should remain inspectable independently. |
| Warning prose overstates evidence and lacks a visual location. | `sem_segment/pipeline.py:345` calls a decrease of more than 2% in sampled gradient strength degraded; `:374` asserts those contours are worse. The coordinator saves warning strings, not per-region edge-strength changes. | A weaker sampled gradient is a reason to inspect, not proof of an incorrect physical boundary. Changing numbers make many warning strings unique, defeating the renderer's text deduplication. |
| ECD and refined ECD lack measurement evidence. | `sem_segment/metrology.py:201` measures polygon area; `:219` computes diameter. `RefinedContour.polygon` retains coarse vertices wherever refinement failed (`sem_segment/refine.py:126`). Comparison exports omit those validity flags and interior rings. | The reader cannot see which area produced ECD or which boundary sections were actually refined. The comparison also hardcodes ECD despite the general metrology module supporting other CD definitions. |
| “Complete” describes execution, not usable contours. | `tools/real_sem_compare.py:572` sets overall completion from noise-analysis status alone. | Successful execution can coexist with no usable CD evidence. Both states must be visible without a warning wall. |
| TensorBoard repeats the same presentation problems. | `sem_noise/comparison_report.py:284` exports every scalar; `:294` duplicates shared figures under multiple model tags. | It reproduces the table dump and static panels rather than offering acquisition browsing. |

The default comparison segmenter is global Otsu thresholding with dark polarity,
connected components, and optional watershed splitting
(`sem_segment/backend_classical.py:35`). All detections touching the border can
mean segmentation selected background or merged regions, an unsuitable crop or
polarity, or genuinely partial features. Noise is another possibility. The
messages alone do not establish which happened on the remote data. Likewise,
the absence of warnings on denoised images does not establish correct contours.

The recent performance changes accelerate the same measurements. They do not
resolve these analysis or presentation problems. Preserve the working CPU
implementation and optional CUDA path; further acceleration is not part of this
revision.

## The revised report

**Start with one site and two large, synchronized image panes.** Default to raw
input versus a selected model. Either pane can select raw, any of the six model
outputs, average8, or average128. Show the full field initially. Model selectors
use readable treatment names and show checkpoint step in a compact detail view.
Avoid a default eight-thumbnail grid that makes the relevant edges too small.

| Control | Behavior |
|---|---|
| Frame slider, previous/next, play/pause | Replace successive images in the same viewport without changing zoom or brightness. Raw and each model expose all 128 frames; average8 exposes its 16 blocks. Average128 is a static reference. |
| Link acquisitions, on by default | Raw frame `i` selects model frame `i` and average8 block `floor((i-1)/8)+1`. Show both the acquisition and block's source range. Moving an average8 slider selects the first acquisition in that block in linked raw/model panes; the acquisition slider still exposes all eight individual frames. Unlink for an explicit early/late comparison. |
| Side by side / wipe | Wipe overlays A and B at identical native coordinates; a separate divider slider reveals either side. Each image and its own contour layer move together. This divider is distinct from the frame slider. |
| Contours: off / mask / refined / both | The same frame and wipe controls work with every image's own contours, including raw and both averages. Show all detected regions, including unmatched or excluded regions with different styling. |
| Fit image / 1:1 / linked zoom and pan | Start at full image; zoom deliberately. Clicking a hole fits its complete boundary plus padding, with its location still visible in the full-field view. Crop size comes from the boundary, not the matching gate. |
| Select a hole | Highlight its boundary and measured area, explain its status, and show its diameter across acquisitions. Selecting a trajectory point returns to that exact image and boundary. |

Use one shared fixed intensity scale, initially 0–255 DN, with no per-frame or
per-model autocontrast. Native coordinates remain the default: a model-induced
shift should remain visible. Existing translation estimates are measurements
for correspondence and drift plots, never corrections applied to the images.
Full-resolution files remain accessible from each pane.

Below the viewer, show only three focused comparisons for the selected model:

1. **Brightness preservation.** Plot raw and output means against acquisition
   order/time, then `mean(output_i) - mean(raw_i)` with a zero reference line.
   Use identical pixels and retain charging trends. No gain fitting, offset
   removal, percentile matching, or detrending. For average8, label block ranges
   and block-center times; do not imply it is a single exposure. Exact equality
   of noisy per-frame means is not a prerequisite for preserving the underlying
   trend, and no arbitrary pass/fail tolerance is introduced.
2. **Changes in texture and geometry.** Offer an output-minus-input image with
   one shared, symmetric DN scale and no recentering. Structured edges in this
   difference help reveal smoothing or movement; it is not ground-truth noise.
   Retain the existing output-minus-raw translation tracks, with failed fits
   shown as missing. Native temporal standard-deviation maps can show changing
   texture across all frames, labeled *temporal variation*: motion, charging,
   and specimen changes also contribute. Use common image coordinates or the
   same explicitly shown ROI, not a different automatically selected flat mask
   for each model. Lower variation alone does not identify the best denoiser.
3. **Diameter of the selected hole.** Show its trace and usable observation
   count, linked to the full-image overlay. Make aggregate repeatability a
   secondary expandable view with its contributing holes and counts. Raw,
   average8, and model coverage are reported separately. A model comparison uses
   the common holes of the models being compared and, for paired frame
   comparisons, shared acquisitions. It does not require raw segmentation to
   succeed. Sixteen block averages and 128 single-frame outputs retain their
   different sample counts and exposure lengths.

**Explain the actual diameter.** The current primary number is area-equivalent
circular diameter:

`ECD = 2 * sqrt(A / pi)`

`A` is the area inside the measured polygon minus any interior rings. For nm,
multiply the pixel diameter by the configured nm/pixel. This is not a directly
measured horizontal width or a fitted-circle diameter. Keep this definition and
label it ECD consistently in the comparison; do not imply other configurable CD
definitions were used. Show the shaded area and numerical area alongside the
formula. Refined ECD uses the refined polygon, which currently falls back to the
mask boundary at unsuccessful vertices. Draw those spans differently and show
the refined fraction. Export the exact rings and validity flags used to explain
the number.

**Use concise measurement status.** For example: “Contours unavailable on this
frame: no complete holes detected,” with the attempted segmentation visible.
Border objects remain visible and labeled partial; do not present their ECD as
a complete-hole measurement. A weak-edge warning identifies the affected region
and before/after boundaries as “needs inspection.” It does not prove refinement
failed or improved accuracy. Keep detailed reasons and frame lists expandable;
do not print one paragraph per distinct warning. Distinguish report generation
success from contour availability.

## Small implementation sequence

1. **Remove the wrong analysis path and fix the pixel contract.** Stop invoking
   the complete `sem_noise.pipeline.analyze_dataset` workflow from
   `tools/real_sem_compare.py`. Reuse the existing mean/range/translation data
   and add only the small native-pixel difference and temporal-variation
   calculations the report displays. Place reusable numerical helpers in
   `sem_noise`, with no training imports. Remove correction-report links and
   calculations from this comparison. The standalone acquisition-analysis
   workflow remains available for its original purpose. Disable segmentation
   contrast stretching in comparison settings; derivative/profile operations
   inside edge measurement remain explicit parts of the estimator.

   All analysis must start from saved uint8 RGB images with identical channels,
   including predictions and both averages. This is the user's measurement
   contract, not a display limitation to work around. Never retain or analyze
   intermediate floating-point predictions for metrology, brightness, or noise.
   Keep the existing pre-export range validation solely to identify invalid or
   clipped output. Fixed checkpoint normalization and its inverse belong to
   image production; numerical image analysis begins after uint8 conversion.

2. **Keep enough contour evidence and decouple detection from correspondence.**
   Extend the existing per-frame records to retain all detected outer/interior
   contours, local region IDs, measurement status, and refined-vertex flags.
   Local overlays must work even when registration or hole matching fails.
   Use the existing average128 as the initial hole-location/correspondence
   reference, and use that same reference for the raw/output translation
   tracks so their coordinate systems agree. Segment it once and show the
   complete reference with IDs. It is an aid for locating holes, not a true
   boundary, target diameter, or substitute for segmenting each observation.
   Its own contours and single-image ECD can be inspected, but it has no temporal
   repeatability estimate. If drift has blurred it too much or segmentation is
   wrong, keep local overlays and mark correspondence unavailable. Do not
   fabricate IDs or add automatic backend retries. Check polarity/crop and the
   existing settings on a representative pilot before any algorithm changes.

3. **Replace the report body with the viewer and focused comparisons.** Extend
   `sem_noise/comparison_report.py` with a small static HTML/JavaScript viewer
   and its assets. Use native range inputs and image/canvas/SVG layers. Reuse
   stored measurements and PNGs; no web application framework, server backend,
   database, custom TensorBoard plugin, or automated model score. Load active
   images and nearby frames rather than every full-resolution image at once.
   Avoid embedding every contour twice in a giant HTML document; load the
   selected series' data through simple static assets that also work when the
   downloaded report is opened locally. Keep JSON/CSV behind a Details link and
   format displayed values to meaningful precision.

   Provide a small rebuild entry point using the existing comparison record
   and saved images. Existing runs need contour remeasurement to recover the
   outlines that were never saved; they do not need denoiser inference repeated
   just to obtain a usable viewer. Later HTML-only revisions reuse measurements.
   This is report regeneration, not a general resume/checkpoint system.

4. **Reduce TensorBoard export and verify one site.** Keep the HTML report as
   the full interactive audit. Export a few clearly labeled full-image/contour
   panels and the same selected brightness/diameter summaries to the existing
   separate comparison log directory. For offline acquisition browsing, use
   image sequences whose event step means acquisition number, with checkpoint
   step recorded as metadata; do not combine acquisition and optimizer steps on
   one axis. Do not duplicate the entire multi-model figure under every model.
   The existing opt-in live train/validation panels remain lightweight and
   use optimizer steps. No test-site analysis enters training, and no full
   contour pipeline is added to each validation interval.

## What to drop, and what to preserve

Drop from this comparison: output gain/offset fits, corrected diagnostic images,
two-region/percentile correction graphs, the eight complete acquisition reports,
unfiltered numerical tables, repeated warning paragraphs, sixteen static contact
sheets, four arbitrarily cropped hole stacks, and duplicate TensorBoard exports.
Do not replace them with another comprehensive diagnostic dashboard.

Preserve: acquisition ordering, source/checkpoint hashes, treatment metadata,
held-out-site checks, native model inputs, nonoverlapping raw means, output range
audits, independent drift estimation, numerical contour/metrology primitives,
optional existing CUDA acceleration, and reproducible JSON/CSV details. Do not
change training losses or select a winning model from the reported plateau
values. The task is to make evidence inspectable before deciding what further
training or segmentation changes are warranted.

## Acceptance checks

The revision is complete when a reader can open one site, scrub every frame,
compare any two sources in the same field of view, and trace any displayed ECD
to the exact displayed contour and area without opening a CSV.

Targeted regression checks should cover:

- Identity output, a deliberate constant brightness offset, and a deliberate
  intensity scaling: expected differences remain visible and are never fitted
  away. Guard against calls to acquisition brightness-correction functions.
- Every analytical value comes from the decoded saved uint8 images, including
  a prediction whose fractional values would produce different measurements
  before rounding. Segmentation input conversion cannot change the measurement
  array, and no intermediate prediction array reaches metrology.
- Slider boundaries at acquisitions 8/9 and 120/121, final frame 128, all 16
  average blocks, and static average128. Image, contour, frame label, selected
  hole, and chart marker always refer to the same observation.
- All detected holes remain visible, including more than four, unmatched
  detections, border regions, and an empty correspondence template. Native
  overlay coordinates account for an explicit crop exactly once.
- Polygon area reproduces displayed ECD, including interior rings and mixed
  refined/coarse spans. Failed raw measurements do not erase valid model-only
  comparisons. Missing values are never plotted as zero or reused from another
  frame.
- Browser inspection of full view, 1:1 zoom, a whole-hole close-up, wipe with
  contours, and a failure case. A one-site pilot on the actual remote data must
  show plausible masks before expanding to all sites. Synthetic tests cannot
  establish this.
- Rebuilding from saved results does not load checkpoints, run inference, or
  invoke acquisition-correction analysis. Existing CPU/CUDA parity requirements
  continue to apply if shared numerical code changes.

**Historical audit verification:**
`python -m pytest tests/edge_denoise/test_real_sem_compare.py tests/sem_segment/test_repeatability.py -q`
passed **25 tests**. These establish current contracts, not real-data contour
quality or usable UI. In particular, the six-model workflow test mocks noise,
registration, and segmentation, and explicitly requires affine analysis for
every series (`tests/edge_denoise/test_real_sem_compare.py:170`). That expectation
must be replaced, not preserved, when implementing this plan. No production
code or remote artifacts were changed during the audit.

**Revision verification:** saved-pixel analysis, full-field contours, matching
failure handling, report rebuilding, directory overrides and the segmentation
suite passed locally (267 tests). The complete `sem_noise` suite together with
the comparison tests passed (148 tests; one optional standalone-browser test
skipped). These runs overlap. A synthetic 128-acquisition hole-array report was
rebuilt and its frame links, contour selection, measurement navigation and
difference controls exercised in a browser. No remote artifacts were accessed;
actual contour quality remains a visual judgment on the server's real images.
