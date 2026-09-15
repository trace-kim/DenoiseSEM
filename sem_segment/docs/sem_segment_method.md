# SEM Segmentation and Contour Metrology — Method

*2026-09-15 · code: [`sem_segment/`](../) · guide: [`README.md`](../README.md)*

**TL;DR** — SAM 3 answers *which pixels belong to which feature*; it does not
answer *where the edge is*. Its vision tower is fixed at 1008 px, so on a larger
frame every mask boundary is quantised to roughly two original pixels. The
pipeline therefore treats the mask boundary as a region proposal and measures
the edge separately, by fitting a one-dimensional edge model to the original
pixels along each contour normal. On analytic edges that measurement is unbiased
to better than 0.001 px; on a thresholded disk of known radius it removes a
+0.15 px bias to under 0.01 px.

---

## 1. Why the mask boundary is not a measurement

`skimage.measure.find_contours` at level 0.5 produces a polygon with fractional
coordinates, which makes it look sub-pixel. It is not. It interpolates the 0.5
level of a **binary** array, so the only information it has is which pixels the
segmenter set. Its accuracy is bounded by the resolution at which the mask was
produced.

For SAM 3 that resolution is fixed. `Sam3TrackerPromptEncoderConfig` reports
`image_size: 1008, patch_size: 14`, and the upstream documentation warns that
other resolutions degrade accuracy. A 2048² frame is therefore downsampled by
about 2×, segmented, and the mask upsampled back, so boundary positions land on
a grid roughly 2 original pixels coarse.

That is fine for identity and topology and useless for metrology, which is the
whole reason the two steps are separated. It is also why the default policy for
an oversized frame is **resize, not tile** (§6).

## 2. Contour extraction (method 1)

The mask crop is padded by one pixel before tracing. Without the pad a region
running to the image border yields an *open* polyline, and every polygon
measure — shoelace area, closed-ring perimeter, moment centroid — silently
becomes wrong rather than failing.

Rings are classified by enclosed area: the largest is the outer boundary, the
rest are holes. Each is resampled to uniform arc length, **including the
wraparound segment** from the last vertex back to the first; omitting it leaves
a density dip at the seam that shows up as a periodic artefact in any
along-contour statistic.

### Normals

A marching-squares polygon traced from a binary mask is a staircase. An
unsmoothed central-difference tangent oscillates between axis-aligned
directions, and the normals built from it point in the wrong direction at a
large fraction of vertices. Vertex coordinates are therefore Gaussian-smoothed
along the ring (`sigma_normal_px`, default 2.0) before differentiating.

This is a genuine trade-off, not a free fix. Smoothing the vertex sequence also
biases a genuinely curved boundary inward, by approximately `sigma² / (2R)` for
radius `R`. At the default `sigma = 2` and `R = 20 px` that is 0.1 px — which
the refinement then removes anyway, since refinement measures along the normal
rather than trusting the vertex position. What the smoothing has to get right is
only the *direction*, and direction is far more tolerant.

**Orientation is a majority vote over the whole contour, never per vertex.** A
per-vertex inside/outside test flips exactly where the mask is ragged, which is
where the geometry is least reliable, and one flipped normal puts that vertex's
refined position on the wrong side of the edge. The vote probes the **mask**,
never the intensity: SEM contrast polarity varies between detectors and between
feature types, so "inside is brighter" is not something this package may assume.

Hole boundaries need no special case. "Out of the mask" on an annulus means
*toward* the hole centre, which is what the same vote produces — probing outward
in radius from an inner boundary lands inside the ring, so the normal flips
inward. Both boundaries then traverse their profile material-first, which is
what §3 relies on.

## 3. Edge location (method 2)

At each vertex the image is sampled along the outward normal over `±search_px`
at `step_px` spacing, via `scipy.ndimage.map_coordinates`.

**Interpolation order is cubic by default.** Bilinear sampling of a sub-pixel
profile injects a systematic error that repeats once per pixel, of order
0.05–0.1 px — the entire precision budget of this measurement. A test compares
the two across eleven sub-pixel phases and asserts cubic is no worse.

**Polarity is decided once per contour**, by comparing the mean intensity inside
and outside across all vertices, for the same reason orientation is.

### The three estimators

#### `threshold` (default)

The 50 % level between robust plateau estimates, located by linear
interpolation: `t = offsets[i] + step · d[i] / (d[i] − d[i+1])`. Plateaus are the
10th and 90th percentiles of the profile, and the threshold is their midpoint —
numerically the same convention as this repository's existing CD harness
(`burst_diffusion/repeatability.py`, `_PROFILE_PERCENTILES = (10.0, 90.0)`,
`_crossings`), so contour-derived CD stays comparable with everything already
published in `edge_denoise/docs/`.

It is applied in **two passes**, which the existing harness does not need, and
the reason is worth recording. There, the profile is a band average spanning a
whole feature, so the percentiles straddle the edge evenly and do estimate the
plateaus. Here the window is centred on the *segmentation's guess*, which may
sit a couple of pixels off. The edge then divides the window unevenly: with the
edge at +2 in a [−6, +6] window, two-thirds of the samples lie on the low side,
the 90th percentile never reaches the high plateau, and the threshold lands low.

Measured, this is a **constant −0.06 px** for a 2 px starting error — identical
at every sub-pixel phase, which is what identifies it as a plateau-estimation
error rather than an interpolation error. It is a bias, so averaging more
vertices does not remove it. Re-centring the percentile window on the first-pass
estimate restores the balance and the convention together; the window centre
snaps to a sample index, so two further passes drive the residual to zero.

*Bias to be aware of:* a sloped background tilts the profile and moves the two
percentile plateaus unequally, shifting the midpoint.

#### `gradient_peak`

The maximum of |dI/dt| after Gaussian smoothing, refined by a parabola through
the three samples around it:
`δ = ½(y₋₁ − y₊₁) / (y₋₁ − 2y₀ + y₊₁)`. The classical CD-SEM estimator. A peak
found at the window edge is rejected rather than extrapolated.

*Bias to be aware of:* on an asymmetric edge the steepest point is not the
midpoint, so this and `threshold` disagree by a real amount rather than by noise.

#### `erf`

Least squares fit of `I(t) = a + b·erf((t − t₀) / (√2 σ))`, solved by damped
Gauss-Newton with an analytic Jacobian, all vertices batched as 4×4 normal
equations. A generic optimiser per vertex costs a fraction of a millisecond,
which becomes a minute at a hundred thousand vertices; the batched solve is what
makes it usable at full-frame scale. `t₀` is initialised from the `threshold`
estimate, which keeps it to a handful of iterations and stops it walking to a
neighbouring edge.

It uses the whole profile rather than one level crossing or one peak, so it is
the most noise-tolerant of the three, and `σ` falls out as a genuine
measurement: the edge width, combining beam blur with true edge slope, usable as
a focus and astigmatism indicator.

*Bias to be aware of:* it assumes a symmetric edge and exactly one edge inside
the window.

### Measured performance

Across 21 sub-pixel phases × 3 starting offsets on analytic edges (`σ = 1.5`):

| estimator | bias (px) | spread (px) | max abs error (px) |
|---|---|---|---|
| `threshold` | +0.0000 | 0.0003 | 0.0004 |
| `gradient_peak` | −0.0000 | 0.0012 | 0.0017 |
| `erf` | −0.0000 | 0.0002 | 0.0002 |

Under additive Gaussian noise, 3σ of the recovered edge position:

| noise σ | `threshold` | `gradient_peak` | `erf` |
|---|---|---|---|
| 0.01 | 0.042 | 0.164 | 0.028 |
| 0.03 | 0.140 | 0.355 | 0.065 |

`gradient_peak` is 4–6× worse under noise, which is expected: differentiation
amplifies it. `erf` is the best choice on noisy data; `threshold` is the default
for comparability with existing measurements, at a modest cost.

### Rejection, not clipping

A vertex is rejected when contrast is below `min_contrast`, when the profile
window leaves the image, when there is no crossing, when there are several and
`allow_multiple_crossings` is off, when the estimate reaches the search limit, or
(for `erf`) when the normalised residual exceeds `max_residual`.

Clamping a failed fit to the search limit would censor the displacement
distribution toward zero and make the refinement look better than it is. Every
rejection is counted by reason and surfaced as `valid_fraction` and
`refine_rejections`.

Short runs of rejected vertices are interpolated across (`max_gap_px`); longer
runs are left out of the polygon entirely. **Rejected vertices are never
zero-filled** — that would plant an artificial notch in the contour an order of
magnitude larger than the roughness being measured.

### Curvature

A normal-profile fit assumes the edge is locally straight. On a small feature it
is not, and the measured radius is biased. Measured on analytic disks
(`σ = 1.5`, `threshold`): under 0.25 px at R = 8, under 0.10 px at R = 32, and
monotone in between. No correction is applied. Silently applying one would make
the numbers non-comparable with commercial CD-SEM tools, which do not.

## 4. Metrology

All measures are computed from the polygon, not by counting pixels — the point
of refinement is a boundary located between pixel centres, and a pixel count
discards exactly that. Both methods are always computed and reported side by
side (`*_coarse`, `*_refined`), so the difference is always visible.

Polygon moments are analytic:
`A = ½Σ(x_i y_{i+1} − x_{i+1} y_i)`, with centroid and second moments from the
same cross products; the equivalent ellipse takes `4√λ` of the covariance
eigenvalues. Feret minimum uses the rotating-calipers theorem — the minimum
width of a convex body is attained perpendicular to a hull edge, so checking
each hull-edge normal is exact rather than sampled. Feret mean follows Cauchy's
formula (`hull perimeter / π`), which doubles as an internal consistency check
and is asserted as such.

**CD defaults to the equivalent circular diameter**, `2√(A/π)`, not minimum
Feret. A minimum over many caliper angles is an extreme-value statistic: skewed,
biased low, and with larger variance than the underlying shape warrants. That is
a poor headline number for a repeatability table. `feret_min`, `feret_mean` and
`chord_mean` remain selectable when the narrowest width is genuinely the
quantity of interest.

### Line-edge roughness

LER is the deviation of the **refined boundary** about its own low-pass trend,
projected onto the local normal, high-passed at `ler_highpass_cutoff_px`.

It is emphatically **not** computed from the refinement displacement. The
displacement measures how far the segmentation's boundary was from the truth,
which is a property of the model; roughness is a property of the specimen.
Conflating them yields a number that *improves as the segmentation gets worse*.
A test asserts that a boundary displaced by a large constant reads as smooth,
and that a boundary with a known sinusoidal wobble recovers `3A/√2`.

The filter response is part of the measurement. A Gaussian high-pass at cutoff
`c` passes a component of wavelength `λ` with gain `1 − exp(−2π²c²/λ²)`; at
`λ ≈ 3c` that is only about 85 %. The config rejects a cutoff below
`2·sigma_normal_px`, because below that the band has already been smoothed away
by the normal estimation and LER would be reading its own filter.

### Aggregation

Border-touching regions are **measured and reported individually but excluded
from image-level aggregates**. Their visible geometry is well defined; their true
extent is not, and averaging truncated features into a size distribution is how a
population quietly acquires a low-side tail. They remain nearest-neighbour
*candidates* while being excluded as nearest-neighbour *queries* — an asymmetry
that is deliberate, since a truncated feature is still a real neighbour.

## 5. Mask post-processing

**Containment is checked separately from IoU.** A mask nested inside another has
a *low* IoU with it, so IoU-only deduplication keeps both and every feature is
measured twice. `sam3_auto` produces this "the hole and the hole's dark core"
pattern routinely. The default `containment_keep: larger` keeps the outer
boundary rather than the higher-scoring one — the opposite of the usual
detection convention, and correct here, because the tight inner core often
scores higher and is not the feature.

**Holes are filled only when small.** Always-filling silently turns a genuine
annulus — a ring, a via seen through a dielectric — into a disk. Speckle holes
below `max_fill_area_fraction` of the region are filled; larger ones are kept,
counted, and subtracted from area.

**Region ids are positional, not score-ordered**, so two runs on the same image
produce identical CSV row order.

### Polarity selection in the classical backend

The obvious heuristic — "features are whichever side of the threshold covers
less of the frame" — inverts as soon as features cover more than half the image,
and then segments the gaps *between* them, which looks plausible and is entirely
wrong. Real micrographs do this routinely; a field of packed spheres is mostly
sphere. This was observed on `data/public_sem_kriss_inspection/30us_001.tif`,
where it produced 49 background fragments instead of 16 spheres, and passed
every synthetic test because those had features on a majority-background field.

The durable distinction is topological: features are many compact regions, often
wholly inside the frame; substrate is one large region running off every edge.
Each side is scored by the fraction of its area in components touching the
border, and the lower score wins. This handles bright spheres on dark substrate
and dark holes in a bright matrix without assuming either. `polarity` can be
forced to `bright` or `dark` when the heuristic is wrong.

## 6. Oversized frames

`resize` is the default, and `tile` is opt-in. Refinement only needs the mask
boundary within a few pixels of the truth, which survives a 2× downsample
easily, and it then recovers full-resolution precision from the original pixels.
That costs one forward pass and has **no seams at all**.

Tiling keeps the model at native scale but introduces seams, requires an overlap
wider than the largest feature, and can lose a feature entirely. It is worth it
only when features would fall below roughly ten pixels across in the resized
frame — which the pipeline warns about rather than deciding silently.

### Seams

**Label maps cannot be blended.** Averaging two binary masks and re-thresholding
at 0.5 produces a boundary that is a mixture of two different model opinions —
fabricated geometry, which a metrology package must not emit. This is worth
stating explicitly because the obvious thing to reach for is the Hann-window
overlap-add already in this repository (`edge_denoise/distill.py`), which is
correct for *continuous* images and wrong for this data type.

Instead, whole instances are selected. Each tile's instances are mapped to full
frame coordinates, and anything touching a tile's **interior** border — a tile
edge that is not also an image edge — is discarded, because it is truncated by
the tile and strictly worse than the neighbouring tile's view of the same
feature. Given an overlap wider than the largest feature, every feature is
wholly inside at least one tile, so nothing is lost. That single rule eliminates
the seam problem, and it is why the overlap constraint is a requirement rather
than a nicety.

A feature wider than the overlap is truncated in *every* tile and lost
completely. Those are counted as `tiles_truncated_instances` and surfaced; a
non-zero count means the overlap was too small.

The last tile origin is **clamped, not padded**. Padding fabricates a hard
border that the model segments as a feature; clamping merely gives the last tile
more overlap, which is harmless.

## 7. The measurement invariant

The pipeline holds two arrays. `measure01` is exactly as loaded — never resized,
stretched, filtered or denoised — and is the only array refinement and metrology
read. `model_rgb` is what the backend sees and may be contrast-stretched and
downsampled.

This is what makes it safe to stretch a low-contrast frame for SAM 3's benefit:
the transformation cannot reach a reported edge position. A regression test
asserts that changing the stretch percentiles leaves refined coordinates
bit-identical.

It is also why a learned denoiser is deliberately *not* wired into this package.
Denoising for segmentation would be harmless under this invariant, but this
repository's own harness exists to detect that learned denoisers shift edges, and
a package whose job is to report edge positions should not make that call on the
operator's behalf.

## 8. Known limitations

- **No physical scale is read from any file.** `pixel_size_nm` is
  operator-supplied. A TIFF resolution tag is print metadata, not a calibration;
  the FEI `PixelWidth` tag in the reference data is real but vendor-specific, and
  trusting one vendor's tag silently would be worse than requiring the value.
- **SAM 3 on SEM imagery is unvalidated here.** The adapters are tested for
  output-shape correctness against a fake `transformers`, and the numerical core
  is tested against ground truth, but whether SAM 3 segments a given micrograph
  well is an empirical question this package cannot answer in advance. That is
  what the `classical` control arm is for.
- **The erf estimator assumes one symmetric edge per window.** Dense features
  closer together than `2·search_px` will produce `multiple_crossings`
  rejections rather than wrong answers, which is the intended failure mode but
  does reduce `valid_fraction`.
- **No repeatability harness.** This package measures one image at a time.
  Quantifying measurement precision needs repeated acquisitions, which is what
  `burst_diffusion/repeatability.py` does for the denoising work; joining that
  comparison table would be a natural next step and is not done here.
