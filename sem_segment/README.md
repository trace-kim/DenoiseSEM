# sem_segment

Segment meaningful features in an SEM image with SAM 3, extract the contours
that wrap them by two methods, measure those contours, and render the result as
a page you can actually look at.

This package imports nothing else in this repository. It is an *analysis-tier*
package like `sem_noise`: the core pipeline needs no GPU and depends on the
`[analysis]` extra; SAM 3 additionally needs `[segment]`; and the `classical`
backend runs with neither.

```
python -m pip install -e ".[dev,analysis,segment]"

python -m sem_segment backends                       # what can run here, and why not
python -m sem_segment download-weights               # fetch SAM 3 once, explicitly
python -m sem_segment segment --config sem_segment/configs/default.yml \
    --input <image-or-folder> --out output/sem_segment/<run>
```

Each input image produces one self-contained directory: `index.html`,
`metrology.csv`, `contours.json`, `masks.npz`, `summary.json`, `provenance.json`.

## The idea in one paragraph

SAM 3 is very good at the question classical thresholding cannot answer — *which
pixels belong to which feature, and which side is inside* — and not good at the
question metrology actually asks, *where exactly is the edge*. Its vision tower
runs at a fixed 1008 px, so on a larger frame every mask boundary is quantised
to roughly two original pixels. So the pipeline uses the mask for topology and
then measures the edge properly: at every contour vertex it samples the
**original, unmodified** image along the outward normal and locates the
**gradient peak nearest that boundary**.

Nearest, not strongest — that distinction is the whole ballgame on a crowded
field. A useful window usually contains a neighbour's edge too, and if that
neighbour is brighter its edge is steeper, so any strength-based rule measures
the wrong particle. The window is also sized to each feature so it cannot span
the object, and vertices that disagree with their neighbours are dropped.

```
image ──► [segmentation backend] ──► instance masks ──► method 1: mask boundary
  │            SAM 3 / classical                              (region proposal)
  │                                                                  │
  └──────────────── original pixels ────────────────► method 2: edge fit along
                    (never modified)                   the normal (measurement)
                                                                     │
                                                    metrology ◄──────┘
                                                    report, CSV, contours
```

On synthetic edges with known sub-pixel positions, method 2 is unbiased to
better than 0.001 px with all three estimators. On a thresholded disk of known
radius, it removes a +0.15 px bias down to under 0.01 px.

## The invariant that makes this safe

**Segmentation may see a modified image; measurement never does.**

The pipeline holds two arrays. `measure01` is exactly as loaded — never resized,
stretched, filtered or denoised — and is the only thing refinement and metrology
ever read. `model_rgb` is what the backend sees, and may be contrast-stretched
and downsampled to the model's native resolution.

That is why a percentile stretch to help SAM 3 on a low-contrast frame cannot
move a reported edge, and why resizing an oversized frame costs so little: the
mask only has to land within a few pixels for the refinement to recover
full-resolution precision from the original data. A regression test asserts that
changing the stretch leaves refined coordinates bit-identical.

## Backends

| backend | what it is | needs weights |
|---|---|---|
| `sam3_auto` *(default)* | automatic mask generation from a grid of point prompts | yes |
| `sam3_text` | a noun phrase returns every matching instance in one pass | yes |
| `sam3_prompt` | SAM 2-style points/boxes, one object per prompt | yes |
| `classical` | Otsu threshold, connected components, optional watershed | **no** |

`sam3_auto` is the default on purpose. SAM 3's concept head was trained on
natural-image vocabulary, and micrographs are far outside that distribution;
automatic mask generation needs no semantic understanding at all. `sam3_text` is
worth trying — whether "hole" or "particle" means anything on your imagery is an
empirical question — but it is not something to assume.

`classical` is not a toy. It is the CI path, the only backend that works on an
air-gapped machine, and the control arm that shows whether SAM 3 actually bought
anything on a given image.

## Optional CUDA contour refinement

Mask backend and numerical refinement have independent devices. The default
refinement runs on CPU. For a CUDA 12.x server, install
`python -m pip install 'cupy-cuda12x>=13.4,<15'` and add to the segmentation YAML:

```yaml
refine:
  device: cuda:0
  estimator: gradient_peak
  cuda_batch_samples: 1048576
```

Use the matching CuPy wheel for other CUDA versions; see the
[installation guide](https://docs.cupy.dev/en/stable/install.html).
This uses one visible GPU for float64 spline/profile calculations and edge
strength diagnostics. Classical masks, contour tracing and polygon geometry
remain on CPU. `threshold` and `erf` require `refine.device: cpu`.
CUDA is opt-in and fails explicitly if unavailable. Measurements still use
the original native pixels. No neural model or weights are required.

Run `python tools/benchmark_sem_metrology.py --device cuda:0` on the server
to check CPU/GPU parity and complete per-image speed. Add `--image frame.png`
for saved uint8 pixels and `--config your-segmentation.yml` for the same
classical settings/crop used in the comparison. The benchmark returns a nonzero
exit on parity failure and writes warm-up and stage timing details to JSON.
See [the comparison guide](../edge_denoise/docs/real_sem_comparison.md#single-gpu-contour-acceleration)
for single-GPU allocation and remote commands.

A series caller may reuse `sem_segment.cuda.CudaRefiner(config.refine)` as a
context manager and pass it as `refiner=` to `segment_image`. It owns one device,
stream and allocation pool; pixels and measurements are recomputed each call.
Closing it frees only its own cached GPU allocations. Without a supplied refiner,
`segment_image` manages the refiner lifetime for one image.

## Getting the weights

`facebook/sam3` is a gated repository (0.9 B parameters, licence "other").
Request access once at <https://huggingface.co/facebook/sam3>, then:

```bash
hf auth login                            # the flow the model card documents
python -m sem_segment download-weights   # 3.4 GB
python -m sem_segment backends           # should now say ready
```

`hf` ships with `huggingface-hub`, so it is already installed. `HF_TOKEN=hf_...`
works too; the package asks `huggingface_hub` which token it would use, so
either mechanism is detected and `backends` reports which one it found.

The download skips `sam3.pt`, Meta's original-format checkpoint: the repo ships
the model twice and `from_pretrained` only reads `model.safetensors`, so the
default fetch is 3.4 GB rather than 6.9 GB. `--all-files` mirrors the repo.

### If the machine has no internet

`runctl/docs/training_workflow.md` notes the H100 site may be air-gapped. Fetch
on a connected machine as above, then copy the cache across:

```bash
scp -r ~/.cache/huggingface/hub/models--facebook--sam3 <host>:~/.cache/huggingface/hub/
# on that host
export HF_HOME=~/.cache/huggingface HF_HUB_OFFLINE=1
python -m sem_segment backends
```

Or point `segmentation.model_path` at a directory, or use `backend: classical`,
which needs no weights at all.

## Measurements

Everything is computed from the polygon, not by counting pixels, and reported
for **both** methods — `<name>_coarse` from the mask boundary and
`<name>_refined` from the edge fit — so the difference is always visible.

Area, perimeter, centroid, equivalent diameter, circularity, solidity,
convexity, Feret min/max/mean, aspect ratio, equivalent-ellipse axes and
orientation, eccentricity, bounding box and fill, chord widths through the
centroid, critical dimension, line-edge roughness, erf edge width, edge
contrast, and the per-vertex refinement displacement statistics.

Two defaults are deliberate and worth knowing:

- **CD defaults to the equivalent circular diameter, not minimum Feret.** A
  minimum over many caliper angles is an extreme-value statistic — skewed,
  biased low, noisier than the shape it describes. `cd_definition` selects
  another if the narrowest width is genuinely what you want.
- **Roughness is measured from the refined boundary about its own smooth
  trend**, never from the refinement displacement. The displacement says how
  wrong the *segmentation* was, which is a property of the model; roughness is a
  property of the specimen. Conflating them produces a number that improves as
  the segmentation gets worse.

All values are in pixels. Pass `--pixel-size-nm` to add nanometre columns.
Nothing in this package reads a physical scale out of an image file — the value
is operator-supplied, because a TIFF tag is not a calibration.

## Nothing is dropped silently

Every rejected mask and every rejected contour vertex is counted by reason and
carried into `summary.json` and the report. A region count that quietly halved
is otherwise indistinguishable from a specimen that had half as many features.
`valid_fraction` says what proportion of contour vertices produced a usable edge
fit; if it is low, the refined geometry rests on fewer points than it appears to.

## Using it as a library

The CLI is a thin shell. The pipeline is a pure function — it takes an array,
touches no filesystem, and downloads nothing implicitly:

```python
from sem_segment import load_config, load_image, segment_image

config = load_config("sem_segment/configs/default.yml")
image = load_image("frame.tif", crop=(2, 441, 2, 511))   # crop excludes the databar
result = segment_image(image.measure01, config)

for row in result.rows(pixel_size_nm=5.82812):
    print(row["region_id"], row["cd_nm_refined"], row["ler_3sigma_nm"])
```

Folder input writes one independent directory per image and **nothing else** —
no cross-image rollup. Two SEM images in a directory may be entirely unrelated,
so a summary averaging their feature counts would mean nothing. A wrapper that
knows the images *are* related (repeats of one site, a wafer map, a time series)
is the right place for that, and composes on this API.

## A note on `crop`

The crop is applied *before* the grayscale-consistency check. Real instrument
exports carry colored overlays outside the imaging area — the KRISS reference
TIFFs in `data/public_sem_kriss_inspection/` have a green 1-px frame and a
29-row databar, and a strict channel-equality test on the whole file rejects
them outright. `--crop 2,441,2,511` selects the imaging region and they load
cleanly.

## Docs

`docs/sem_segment_method.md` — the maths: normal estimation under staircase
quantisation, the three edge estimators and where each is biased, the roughness
definition, the 1008 px resampling argument, and why tile seams are handled by
selecting whole instances rather than blending masks.
