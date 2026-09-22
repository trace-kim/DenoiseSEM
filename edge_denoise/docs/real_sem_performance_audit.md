# Real SEM comparison performance audit — 2026-09-22

The original code repeatedly serialized accumulated contour coordinates after
`contours/CD complete`. That confirmed inefficiency is now fixed and benchmarked
locally; a full slow remote rerun is not a prerequisite for the fix.
The last source name, process state, real image sizes and server timings are
unavailable. No H100 speedup has been established.

## What actually runs after that message

The entry point is `tools/real_sem_compare.py`, using the ordinary Gaussian +
Otsu detector selected by `edge_denoise/configs/sem_real_compare.yml`.

| Location of the message | Next operations |
| --- | --- |
| Fresh run, raw or average8 | Close the Otsu iterator; run the next baseline's translation diagnostics, native statistics and contours. |
| Fresh run, average128 | Close the iterator; save the accumulated record; load the first model. |
| Fresh run, model | Close the iterator; save the accumulated record; release/load a model or proceed to finalization. |
| Rebuild, last series | Close the iterator; proceed directly to finalization. Rebuilds do not perform the fresh run's intermediate saves. |
| Finalization | Summarize observations; export observations/per-hole/repeatability CSVs, contour JSON and frame CSV; render HTML assets; encode TensorBoard images and close its writer; save the final record. Fresh runs also save in `finally`. |

Before this audit, the summary and export operations had no messages. The
`Rendering interactive comparison report` message came **after** them.
Consequently, if that message never appeared, viewer rendering and TensorBoard
had not yet started in that invocation. A detector-only benchmark cannot explain
that interval.

## Findings in the original implementation

1. **Intermediate records repeatedly embed all contours.** `save_record()` omits
   a site's contours only once `contours_path` exists. Fresh runs set that path
   during finalization. Each preceding model completion therefore serializes
   earlier models again. With one site and six models, baseline contours are
   serialized seven times in intermediate saves; the first model's contours six
   times. The comment claiming contours were stored once was inaccurate.

2. **JSON export consumes CPU and extra host memory.**
   `sem_noise.pipeline.write_json()` recursively builds a converted copy with
   `_json_value()`, then builds a complete indented JSON string, then writes it.
   Coordinate floats undergo scalar finite checks. The same geometry is later
   converted again for per-frame viewer scripts and formatted for SVG bands.
   These costs scale with vertices, not GPU batch size. The CUDA batch memory
   budget does not bound the growing Python contour record or these copies.

   A local synthetic serialization probe used 128 frames/model, 16 contours/frame
   and 128 floating-point vertices/contour. It exercised the actual pre-change
   `save_record()`; only synthetic coordinates were used:

   | Models accumulated | Vertices | comparison.json | Save wall time |
   | --- | ---: | ---: | ---: |
   | 1 | 262,144 | 25.07 MiB | 1.94 s |
   | 3 | 786,432 | 75.20 MiB | 3.17 s |
   | 6 | 1,572,864 | 150.40 MiB | 6.30 s |

   These are single local measurements, without warm-up normalization, not
   remote predictions. A separate one-model list probe spent 0.52 s converting
   values, 0.23 s encoding JSON, and 0.09 s writing it. Real Otsu half-pixel
   coordinates serialize differently from the synthetic floating-point points;
   image content, storage and CPU speed also affect the result.

3. **Report construction is substantial independent CPU work.** A full site has
   913 images: 128 raw, 16 averages, one reference, and 768 model outputs. The
   viewer exports a contour script per image and an all-acquisition SVG per
   source. It also loads all temporal variation maps together. TensorBoard
   decodes those 913 delivered images and submits two full-size images each:
   pixels and contour overlay, totaling **1,826 image encodes**. Its final writer
   close can wait for pending event writes. Existing mocked-writer tests do not
   establish real encoding throughput.

4. **Earlier stages also limit total speed.** Full-resolution OpenCV ECC remains
   on CPU and prepares the unchanged reference again for every fit. Native
   statistics, difference PNGs, contours and TensorBoard independently decode
   saved images. Native CUDA statistics decode serially; `io_workers` applies
   to the CUDA Otsu decoder. Polygon geometry and contour tracing remain on CPU.
   None of these costs is fixed by increasing `analysis_batch` or using all four
   GPUs. `benchmark_sem_analysis.py` excludes inference, ECC and reporting; its
   native-statistics equivalence check is outside its detector throughput timer.

## Logging implemented in this audit

The actual fresh and rebuild workflows now emit flushed, timestamped start/end
messages and elapsed wall time around long operations. Active operations emit
30-second heartbeats, including blocking exports and writer shutdown. A
heartbeat identifies the active call; it is not proof that call is advancing,
and cannot survive process termination or a native call holding Python's GIL.

Coverage includes validation, input PNG preparation, checkpoint loading,
inference/export, ECC, native statistics, difference PNGs, Otsu worker shutdown,
record saves, summaries, each final CSV/JSON export, viewer contour scripts and
SVGs, temporal/overview PNGs, viewer data, TensorBoard images, and final flush.
Rebuilds also identify record/contour loading and asset copying, with copied MiB.
Loop messages include the first, every sixteenth, and last completed item. JSON
saves identify embedded contour counts and resulting file sizes. Errors identify
the failed stage and elapsed time.

`timings.json` is a small sidecar written after every successful record save. It
includes that save's duration without serializing the large record again.
Top-level and site timings reset for rebuilds. Render-only series analysis
timings describe reused measurements; `report_timings_s` describes the new
render. Repeated save times accumulate. Parent and child timings overlap and
must not be summed; CUDA detector stage totals also overlap with CPU work.
The terminal log remains available during a stalled or failed export.

No detector settings, delivered pixels, correspondence or measurement arithmetic
were changed.

## Performance fixes implemented

`save_record()` now serializes only newly appended contours into compact,
immutable `site/contour_parts/*.jsonl` files. The running `comparison.json`
contains references and counts instead of contour coordinates. Each contour is
converted once, with temporary serialization memory bounded by one contour.
Finalization assembles the existing `site/contours.json` array from those encoded
rows without converting the coordinates again. Completed records retain the
existing v3 layout; the updated reader also accepts new intermediate part
references and old embedded contours. The compact parts remain as recovery
artifacts, so final contour data occupies both the parts and the assembled file.

Parts, the assembled array and comparison metadata are published by replacing
a temporary file only after its write completes. Failure preserves the previous
file and never publishes an unfinished part reference. Running part-based records
require the updated reader; completed reports keep their previous format.

The JSON converter uses ordinary scalar finite checks for Python float
coordinates, preserving nonfinite-to-null handling. CSV conversion streams rows
instead of allocating another converted table. Render-only rebuilds copy/reuse
the existing final contour export rather than re-encoding it.

A second local synthetic benchmark compared the original save implementation
against these actual new storage functions. It used the full workflow's seven
checkpoint boundaries: 145 baseline frames plus six models of 128 frames, with
16 contours/frame and 128 half-pixel vertices/contour (1,869,824 vertices).

| Save/export workload | Original | Updated |
| --- | ---: | ---: |
| Seven accumulated checkpoints plus final contour export | 32.28 s | 2.13 s |
| Final contour export alone | 6.42 s | 0.078 s |
| Last incremental comparison record | 134.74 MiB | 637 bytes |
| Final `contours.json` | 90.78 MiB | 25.27 MiB |

The decoded full contour arrays were equal. This is a single local synthetic
serialization comparison, approximately 15.2× faster for the measured workload.
The synthetic record contains no observation tables/models, so the 637-byte
metadata size is not representative of a real comparison. Detector, inference,
ECC, viewer and TensorBoard work are outside this timer. Total remote speed and
peak host memory remain unverified. Regression tests cover append-only saving,
failed parts/assembly/metadata writes, old formats, unchanged images and
measurements, and render-only reuse.

## Remaining plan

1. **Exercise exports from an existing completed report if needed.** Use
   `--render-only`; do not repeat inference, ECC or contour measurements just to
   obtain timings. The synthetic benchmark already verifies the eliminated
   checkpoint cost. A render-only run isolates remaining report costs.
2. **Address the measured report costs.** Process temporal maps with bounded
   memory. Avoid converting the same unchanged geometry repeatedly, while
   retaining the existing offline viewer and all-acquisition SVGs. If TensorBoard
   dominates, optimize image encoding/overlay work with bounded concurrency and
   verify image pixels, tags and acquisition steps. The existing explicit
   `--no-tensorboard` option is available for iterations that need HTML/CSV/JSON
   only; it is not a claimed speedup of the full default workflow.
3. **Optimize earlier work only where timings justify it.** Reuse prepared ECC
   reference inputs without changing its estimator/settings; share bounded
   decoded saved-image batches where practical. Preserve independent raw/output
   drift diagnostics and brightness comparisons. Retain explicit
   `--contours-only` reuse for contour experiments. Do not change the Otsu
   detector or add distributed execution to address serialization costs.
4. **Validate each further performance change.** Compare decoded PNG bytes, full contour
   coordinates/status, ECD, correspondence, brightness and repeatability against
   the baseline; exercise failed/interrupted export and old-report rebuilds.
   Check native CPU/CUDA equivalence on the server. Collect full-workflow wall
   time and memory during the next comparison that is needed for the study,
   rather than requiring an extra slow comparison solely for profiling.

## Remote commands

Run from the repository root in the existing analysis environment. Preserve
scheduler-provided GPU visibility; the recipe selects logical `cuda:0`. Use a new
output directory. These commands do not resume or modify an already running job.

To inspect remaining report costs, use an existing **complete** v3 comparison
and a fresh sibling directory. This runs no inference, ECC or contour analysis:

```bash
set -o pipefail
mkdir -p output
saved_comparison="output/260921_real_n2n_comparison/comparison.json"
render_dir="output/$(date +%y%m%d_%H%M%S)_real_n2n_render_audit"
/usr/bin/time -v python -u tools/real_sem_compare.py \
  --from-comparison "$saved_comparison" --render-only --tensorboard \
  --output-dir "$render_dir" \
  2>&1 | tee "${render_dir}.log"
```

The next new comparison needed for the study automatically uses the improved
storage path. This command is not required as an extra profiling run:

```bash
audit_dir="output/$(date +%y%m%d_%H%M%S)_real_n2n_comparison"
/usr/bin/time -v python -u tools/real_sem_compare.py \
  --config edge_denoise/configs/sem_real_compare.yml \
  --experiment-prefix 260921_real_n2n \
  --site-dir /data/260904_raw_data/test/260904_0947-13 \
  --output-dir "$audit_dir" \
  2>&1 | tee "${audit_dir}.log"
```

Retain any checkpoint/prepared-manifest overrides from the actual experiment.
Render-only includes copying saved assets; that cost has its own timing. It
reuses the contour export and does not exercise incremental checkpoints.
For contour experiments, use `--contours-only` instead of `--render-only`, with
`--contour-method otsu --metrology-device cuda:0`; add `--no-tensorboard` only
when those image events are unnecessary. Plain `--from-comparison` repeats native
analysis and ECC as well as contours.

CPU/CUDA agreement check on a completed comparison:

```bash
python tools/benchmark_sem_analysis.py \
  --from-comparison "$saved_comparison" --device cuda:0 \
  --frames-per-source 16 --analysis-batch 16 \
  --output-json "${render_dir}_cpu_cuda_agreement.json"
```

The agreement check keeps all images and measurements on the server. It does
not substitute for the complete workflow timings above.
