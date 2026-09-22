# Establish a simple contour baseline before replacing the algorithm

> Execution update (2026-09-22): algorithmic simplicity does not relax GPU or
> workflow-efficiency requirements. The historical CPU-only remainder below is
> superseded: CUDA runs Gaussian, ordinary per-image Otsu and connected-component
> filtering in bounded batches, overlapping CPU contours. The main workflow also
> supports CUDA native statistics and explicit reuse of unrelated completed
> analysis. See the real SEM comparison guide for execution controls and the
> remote equivalence/throughput benchmark. The detector itself remains unchanged.

> Follow-up scope, 2026-09-22: the user requested integration into the actual
> remote analysis workflow. Gaussian + Otsu now runs through
> `tools/real_sem_compare.py` for both new runs and saved-report rebuilds, and
> is the default in the real comparison recipe. Closed masks feed the existing
> area/ECD calculation; open border paths remain unmeasured. The original
> preview-only scope below is retained as history. See the
> [main comparison guide](../edge_denoise/docs/real_sem_comparison.md).

## 1. Goal and scope

Implement and evaluate this standard pipeline:

**Saved uint8 image → light Gaussian smoothing → Otsu segmentation → region outlines.**

The deliverable is a small visual comparison against the current algorithm. This
revision will not introduce another refinement method, change the main report's
data format, or claim that the baseline is already suitable for real SEM metrology.

Preserve the agreed requirements:

- Analyze decoded, saved uint8 images only.
- Apply identical detector settings to raw images, averages, and model outputs.
- Keep saved/displayed pixels and brightness measurements unchanged.
- Calculate the Otsu threshold independently for each image.
- Start with dark interiors of closed features; support bright foreground through
  one setting.
- Run real-data checks remotely without transferring images or logs here.

## 2. Withdraw the unfinished changes

The committed baseline is `29e3a106f0b2d3e321eadbc2f6d6c391753ce6a3`.

After checking for newer user edits, withdraw the screenshot-driven drafts in:

- `tools/real_sem_compare.py`
- `edge_denoise/configs/sem_real_compare.yml`
- `sem_noise/assets/comparison.html`, `comparison.css`, and `comparison.js`

Preserve the `AGENTS.md` memory update against ad hoc fixes and the unrelated
untracked documentation. Keep the previously committed report improvements.

These reversions had not happened when this plan was written in Plan mode.

## 3. Implement one ordinary baseline

Add one small function in `sem_segment` for the baseline and one preview command
under `tools/`. Keep the estimator independent of training packages.

Use three settings:

```yaml
polarity: dark
sigma_px: 1.0
min_area_px: 25
```

The implementation will:

1. Read the saved uint8 pixels, preserving any recorded measurement crop.
2. Apply Gaussian smoothing to a floating-point working copy. Default sigma is one
   pixel; zero disables it.
3. Use the **ordinary Otsu threshold directly** to create a foreground mask.
4. Remove foreground components smaller than the configured minimum area.
5. Extract outlines using the existing library's contour function.

Use four-connected foreground consistently. Keep interior boundary loops visible
and border-crossing paths open. Restore crop offsets when drawing on the full image.

There will be no watershed, custom midpoint threshold, contour refinement, circle
fitting, hole filling, or invented boundary closure. Process all components before
applying the area filter; impose no arbitrary candidate-count or maximum-area cutoff.

**The preview measures segmentation behavior.** Its mask outlines will not be
presented as subpixel edge measurements, and it will calculate no new ECD values.
That avoids introducing a metrology system before establishing whether the
detected regions are appropriate.

Use the existing optional CuPy dependency for Gaussian filtering when CUDA is
requested; retain a CPU path. Keep the remaining library operations on CPU
initially. Record timings before considering further optimization.

## 4. Produce a small remote visual check

Add `tools/preview_sem_contours.py` to read an existing `comparison.json` and its
saved images. It must not load checkpoints, run inference, or modify the source report.

The preview will contain:

- A source selector: raw, average8, average128, and each model.
- An acquisition slider covering every saved image in the selected source.
- Three full-image panels: **Original**, **Current outlines**, and **Otsu baseline**.
- Clearly labeled outline colors and links to native-resolution images.
- A short caption showing the threshold and detector settings.

Reuse stored contours for the current-method panel. If unavailable, say so; do not
rerun the old algorithm merely to populate that panel.

Write the preview and its images into a separate output directory. Save a small
metadata file containing settings, image references, thresholds, and timings. Do
not add measurement dashboards or tables of speculative quality scores.

This allows inspection of the actual questions:

- Does the outline follow the intended dark interior?
- Does one feature acquire false internal divisions?
- Are genuine features missed or merged?
- Does performance change between noisy inputs, averages, and model outputs?
- Does smoothing erase visibly important geometry?

## 5. Validation and delivery

Turn the existing synthetic example into reproducible tests rather than depending
on generated files. Retain its frame-9 high-noise case and frame-29 false-split case.

Also test:

- Dark and bright objects, ellipses, concave regions, annuli, and connected shapes.
- Many small noise components preceding legitimate features.
- Border intersections, crop offsets, constant images, and empty detections.
- Saved uint8 input enforcement and unchanged source-image bytes.
- Preview source selection and acquisition indexing.

Check boundary placement on known synthetic shapes as well as feature counts.
Compare sigma zero and one in those tests to expose smoothing effects. Report
those results without treating synthetic success as proof of real-data accuracy.

Run the relevant existing tests plus the new baseline and preview tests. Produce
a clearly labeled synthetic preview that can be opened locally.

Provide this remote command after implementation:

```bash
python tools/preview_sem_contours.py \
  --config sem_segment/configs/contour_preview.yml \
  --from-comparison output/260921_real_n2n_comparison/comparison.json \
  --output-dir output/260921_real_n2n_contour_preview \
  --device cuda:0
```

The input directory is an example; the user supplies their actual saved-report
directory through the flag. Use `--device cpu` for local checks.

**Completion of this revision:** the drafts are withdrawn, one standard baseline
is implemented and tested, and the synthetic preview plus remote command are
delivered. Selection of the production method follows evidence from that
comparison. Any additional processing must address a specific observed failure
and demonstrate an improvement.

## Implementation status — 2026-09-22

Completed the revision described above. The five draft files match
`29e3a106f0b2d3e321eadbc2f6d6c391753ce6a3`; the `AGENTS.md` memory update and
unrelated documentation were preserved. The independent estimator is
`sem_segment/otsu_baseline.py`, and the preview command is
`tools/preview_sem_contours.py`.

See [the preview guide](../sem_segment/docs/contour_preview.md) for the remote
command, settings, synthetic findings, and limitations. The relevant suite
passed **316 tests**. Local browser checks over localhost verified image loading,
source selection, and acquisition navigation. Browser screenshot capture timed
out, and browser security policy blocked automated `file://` navigation; direct
file opening was not browser-verified. CUDA routing and error handling were
tested with mocks, not real hardware.

The reproducible generator is `tests/sem_segment/build_contour_example.py`.
Generated examples in this workspace are:

- `tmp/sem_contour_example_20260921/preview/index.html` (sigma one)
- `tmp/sem_contour_example_20260921/preview_sigma0/index.html` (sigma zero)

They are generated artifacts, not files to commit. Production-method selection
and real SEM evaluation remain the next evidence-gathering step.
