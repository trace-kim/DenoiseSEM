"""Method 2: locating the edge to sub-pixel precision on the original pixels.

The segmentation gives a boundary that is correct in *topology* - which pixels
belong to which feature, and which side is inside - but only approximate in
*position*.  For SAM 3 on a frame larger than its native 1008 px the mask
boundary is quantised to roughly two original pixels.  This module takes that
boundary as a starting guess and measures where the edge actually is, by
sampling the unmodified image along the contour normal and fitting a
one-dimensional edge model to the resulting profile.

Three estimators are offered because they fail differently, and the method
documentation states the bias of each rather than presenting them as
interchangeable:

``gradient_peak`` (the default)
    Gradient-based edge detection: the |dI/dt| peak nearest the mask boundary,
    refined by a parabola through the three samples around it.  The classical
    CD-SEM estimator, and direction-free - it never asks whether intensity
    rises or falls outward, so it cannot be misled by the polarity ambiguity a
    level-crossing estimator suffers on a feature no wider than its own search
    window.  It is biased on an asymmetric edge, where the steepest point is not
    the midpoint.

``threshold``
    The 50 % level between robust plateau estimates (the 10th and 90th
    percentiles of the profile), located by linear interpolation.  Matches the
    convention the rest of this repository's CD measurements use, so numbers
    stay comparable.  It is biased by a sloped background, and it takes its
    threshold from the whole window, so a strong edge elsewhere in the window
    moves the level itself - which is why the adaptive window matters more for
    this estimator than for the gradient one.

``erf``
    Least squares fit of ``a + b erf((t - t0) / (sqrt(2) sigma))``.  Uses the
    whole profile rather than a level crossing or a single peak, so it is the
    most noise-tolerant, and ``sigma`` comes out as a by-product - a physically
    meaningful edge width (beam blur combined with true edge slope) that doubles
    as a focus indicator.  It is biased when the edge is genuinely not
    symmetric, and it assumes a single edge inside the window.

Whichever estimator is used, the candidate chosen is the one **nearest the
mask boundary**, never the strongest.  A window wide enough to be useful usually
holds more than one edge - the feature's own, and a neighbour's a few pixels
further out - and choosing by strength measures the neighbour whenever the
neighbour is brighter.  The segmentation already answered where this feature's
boundary is; strength only rejects noise.  The window is also sized to the
feature (``adaptive_search``) so it cannot span the whole object, and a
coherence filter drops vertices that disagree with their immediate neighbours,
since a boundary cannot genuinely jump several pixels between adjacent vertices.

Two further decisions are worth knowing about:

*Polarity is decided once per contour, not per vertex.*  Whether material is
brighter or darker than its surroundings is a property of the detector and the
feature, not of one vertex.  Deciding per vertex lets noise flip the sense
exactly where contrast is weakest, putting that vertex's edge on the wrong side.

*Rejected vertices are reported, never clipped.*  Clamping a failed fit to the
search limit would censor the distribution toward zero and make the refinement
look better than it is.  Short runs of rejects are interpolated across; long
runs are left out of the polygon entirely and counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RefineConfig
from .contours import Contour

#: Robust plateau percentiles, matching this repository's existing CD harness.
PROFILE_PERCENTILES = (10.0, 90.0)

#: Why a vertex produced no usable edge position.
REJECT_OK = 0
REJECT_OUT_OF_BOUNDS = 1
REJECT_LOW_CONTRAST = 2
REJECT_NO_CROSSING = 3
REJECT_MULTIPLE_CROSSINGS = 4
REJECT_AT_SEARCH_LIMIT = 5
REJECT_POOR_FIT = 6
REJECT_INCOHERENT = 7

REJECT_NAMES = {
    REJECT_OK: "ok",
    REJECT_OUT_OF_BOUNDS: "out_of_bounds",
    REJECT_LOW_CONTRAST: "low_contrast",
    REJECT_NO_CROSSING: "no_crossing",
    REJECT_MULTIPLE_CROSSINGS: "multiple_crossings",
    REJECT_AT_SEARCH_LIMIT: "at_search_limit",
    REJECT_POOR_FIT: "poor_fit",
    REJECT_INCOHERENT: "incoherent_with_neighbours",
}


@dataclass
class RefinedContour:
    """A refined boundary, kept on the coarse contour's own parametrisation.

    Holding the displacement against the *same* vertices the coarse contour used
    is what makes the two directly comparable; the polygon actually measured is
    exposed separately as :attr:`polygon`, which contains only vertices whose
    edge fit succeeded.
    """

    base_points: np.ndarray  # (N, 2) coarse vertices, (y, x)
    normals: np.ndarray  # (N, 2) outward unit normals
    displacement: np.ndarray  # (N,) signed px along the normal; nan where invalid
    valid: np.ndarray  # (N,) bool
    reasons: np.ndarray  # (N,) int reject codes
    edge_width: np.ndarray  # (N,) erf sigma in px; nan for other estimators
    contrast: np.ndarray  # (N,) profile amplitude in [0, 1] intensity
    is_hole: bool = False
    region_id: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def refined_points(self) -> np.ndarray:
        """(N, 2) refined positions; rows for unrefined vertices hold nan."""
        shift = np.where(self.valid, self.displacement, np.nan)
        return self.base_points + shift[:, None] * self.normals

    @property
    def polygon(self) -> np.ndarray:
        """(N, 2) the complete boundary: refined where it could be, coarse elsewhere.

        Every vertex is present.  Dropping the unrefined ones and letting the
        ring close across the gap would draw - and, worse, *measure* - a straight
        line where no boundary was ever observed; on real data that produced
        chords up to 193 px long feeding straight into area and perimeter.

        Falling back to the segmentation's own vertex invents nothing.  It is a
        real estimate of the boundary, simply not a refined one, and
        :attr:`refined_fraction` says how much of the ring it accounts for.
        """
        shift = np.where(self.valid, self.displacement, 0.0)
        return self.base_points + shift[:, None] * self.normals

    @property
    def valid_fraction(self) -> float:
        return float(np.mean(self.valid)) if self.valid.size else 0.0

    @property
    def refined_fraction(self) -> float:
        """Share of the boundary that refinement actually moved."""
        return self.valid_fraction

    def reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for code, name in REJECT_NAMES.items():
            if code == REJECT_OK:
                continue
            n = int(np.count_nonzero(self.reasons == code))
            if n:
                counts[name] = n
        return counts

    def as_contour(self) -> Contour:
        """The refined ring as a plain :class:`Contour` for downstream geometry."""
        return Contour(points=self.polygon, is_hole=self.is_hole, region_id=self.region_id)


def _prepare_profile_image(image01: np.ndarray, interp_order: int) -> tuple[np.ndarray, int]:
    """Share interpolation coefficients within one image, never across images."""
    from scipy.ndimage import spline_filter

    image = np.asarray(image01, dtype=np.float64)
    if interp_order <= 1:
        return image, 0
    # Match scipy.ndimage.map_coordinates(prefilter=True, mode="nearest"):
    # its spline prefilter extends the edge by 12 pixels before filtering.
    # Sampling must use the same offset; changing the boundary mode would move
    # edges near the image border. Regression tests compare to SciPy directly.
    padding = 12
    padded = np.pad(image, padding, mode="edge")
    return spline_filter(padded, order=interp_order, mode="nearest"), padding


def sample_profiles(
    image01: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    *,
    search_px: float,
    step_px: float,
    interp_order: int = 3,
    _prepared: tuple[np.ndarray, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the image along each vertex normal.

    Returns ``(profiles, offsets, in_bounds)`` where ``profiles`` is (N, M),
    ``offsets`` is the (M,) signed distance along the normal, and ``in_bounds``
    is (N,) - False for a vertex whose window leaves the image, because an edge
    cannot be measured from samples that were never acquired.

    Cubic interpolation is the default: bilinear sampling of a sub-pixel profile
    injects a systematic error that repeats once per pixel and is of order
    0.05-0.1 px, which is the entire precision budget of this measurement.
    """
    from scipy.ndimage import map_coordinates

    count = int(round(2.0 * search_px / step_px)) + 1
    offsets = np.linspace(-search_px, search_px, count)
    # (N, M) sample coordinates: vertex + offset along its own normal.
    rows = points[:, 0:1] + offsets[None, :] * normals[:, 0:1]
    cols = points[:, 1:2] + offsets[None, :] * normals[:, 1:2]

    height, width = image01.shape
    in_bounds = (
        (rows >= 0.0) & (rows <= height - 1.0) & (cols >= 0.0) & (cols <= width - 1.0)
    ).all(axis=1)

    coordinates = np.vstack([rows.ravel(), cols.ravel()])
    sampling_image = np.asarray(image01, dtype=np.float64)
    if _prepared is not None:
        sampling_image, padding = _prepared
        coordinates = coordinates + padding
    profiles = map_coordinates(
        sampling_image,
        coordinates,
        order=interp_order,
        mode="nearest",
        prefilter=_prepared is None,
    ).reshape(rows.shape)
    return profiles, offsets, in_bounds


def _nearest_candidate(
    strength: np.ndarray,
    offsets: np.ndarray,
    *,
    min_height: float,
    min_prominence: float,
) -> int | None:
    """Index of the edge candidate nearest the mask boundary, or None.

    This is the rule the whole module turns on.  A profile window wide enough to
    be useful usually contains more than one edge: the feature's own, and a
    neighbour's a few pixels further out.  Choosing by *strength* - the steepest
    gradient, or the only crossing in the polarity the window happens to imply -
    picks the neighbour whenever the neighbour is brighter, which is exactly the
    failure that pushed small dim particles onto their bright neighbours.

    The segmentation already answered "where is this feature's boundary": it is
    ``t = 0``.  So the right question is which candidate lies nearest that, and
    strength is only used to reject noise ripples.
    """
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(strength, height=min_height, prominence=min_prominence)
    if peaks.size == 0:
        return None
    return int(peaks[int(np.argmin(np.abs(offsets[peaks])))])


def _parabolic_offset(values: np.ndarray, index: int) -> float:
    """Sub-pixel correction from a parabola through three samples."""
    if index <= 0 or index >= values.size - 1:
        return 0.0
    y0, y1, y2 = values[index - 1], values[index], values[index + 1]
    denominator = y0 - 2.0 * y1 + y2
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / denominator, -1.0, 1.0))


def _contour_polarity(profiles: np.ndarray, offsets: np.ndarray) -> float:
    """+1 when intensity rises outward, -1 when it falls, decided once per contour."""
    inner = profiles[:, offsets < 0.0]
    outer = profiles[:, offsets > 0.0]
    if inner.size == 0 or outer.size == 0:
        return 1.0
    return 1.0 if float(np.nanmean(outer) - np.nanmean(inner)) >= 0.0 else -1.0


def _crossings_at(
    profiles: np.ndarray,
    offsets: np.ndarray,
    polarity: float,
    low: np.ndarray,
    high: np.ndarray,
    config: RefineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest same-direction 50 % crossing, given plateau estimates."""
    contrast = high - low
    threshold = 0.5 * (low + high)

    # Orient so every profile rises with increasing offset; a valid edge is then
    # always an upward crossing and the search is one rule rather than two.
    oriented = profiles if polarity > 0 else -profiles
    oriented_threshold = threshold if polarity > 0 else -threshold
    delta = oriented - oriented_threshold[:, None]

    # Crossings in EITHER direction are candidates. Filtering by the polarity the
    # window implies is what broke small features: when the window spans a whole
    # particle, the inner and outer halves both straddle it, the implied polarity
    # can invert, and the only crossing in the "correct" direction then belongs to
    # a neighbour. Nearest-to-the-mask-boundary is the reliable rule; polarity is
    # kept only to break ties between two equidistant crossings.
    rising = (delta[:, :-1] <= 0.0) & (delta[:, 1:] > 0.0)
    falling = (delta[:, :-1] >= 0.0) & (delta[:, 1:] < 0.0)
    crossing = rising | falling
    step = float(offsets[1] - offsets[0])

    n = profiles.shape[0]
    positions = np.full(n, np.nan)
    reasons = np.full(n, REJECT_OK, dtype=np.int16)
    counts = crossing.sum(axis=1)

    for index in range(n):
        if contrast[index] < config.min_contrast:
            reasons[index] = REJECT_LOW_CONTRAST
            continue
        if counts[index] == 0:
            reasons[index] = REJECT_NO_CROSSING
            continue
        d = delta[index]
        candidates = np.flatnonzero(crossing[index])
        crossings = np.array([offsets[i] + step * (-d[i] / (d[i + 1] - d[i])) for i in candidates])
        order = np.argsort(np.abs(crossings))
        best = int(order[0])
        if order.size > 1 and abs(abs(crossings[order[0]]) - abs(crossings[order[1]])) < step:
            preferred = rising[index] if polarity > 0 else falling[index]
            for k in order[:2]:
                if preferred[candidates[k]]:
                    best = int(k)
                    break
        positions[index] = crossings[best]
    return positions, reasons


def _recentred_plateaus(
    profiles: np.ndarray,
    offsets: np.ndarray,
    centres: np.ndarray,
    half_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Plateau percentiles from a fixed-width window centred on each estimate."""
    step = float(offsets[1] - offsets[0])
    half_idx = max(2, int(round(half_px / step)))
    n, m = profiles.shape
    centre_idx = np.where(np.isfinite(centres), (centres - offsets[0]) / step, (m - 1) / 2.0)
    centre_idx = np.clip(np.round(centre_idx).astype(int), half_idx, m - 1 - half_idx)
    columns = centre_idx[:, None] + np.arange(-half_idx, half_idx + 1)[None, :]
    window = profiles[np.arange(n)[:, None], columns]
    low, high = np.percentile(window, PROFILE_PERCENTILES, axis=1)
    return low, high


def _estimate_threshold(
    profiles: np.ndarray,
    offsets: np.ndarray,
    polarity: float,
    config: RefineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """50 % crossing between robust plateaus, by linear interpolation.

    The level convention matches this repository's existing CD harness exactly:
    the threshold is the midpoint of the 10th and 90th percentiles of the
    profile, and the crossing is interpolated as ``i + d[i] / (d[i] - d[i+1])``.

    It is applied in **two passes**, which the existing harness does not need.
    There, the profile is a band average spanning a whole feature, so the
    percentiles straddle the edge evenly and estimate the true plateaus.  Here
    the window is centred on the *segmentation's guess*, which may sit a couple
    of pixels off; the edge then divides the window unevenly, the 90th
    percentile never reaches the high plateau, and the threshold lands low.  The
    resulting bias is a constant fraction of a pixel - measured at -0.06 px for
    a 2 px starting error - and it is a bias, not noise, so averaging more
    vertices does not remove it.  Re-centring the percentile window on the first
    pass restores the balance and the convention together.
    """
    low, high = np.percentile(profiles, PROFILE_PERCENTILES, axis=1)
    first, first_reasons = _crossings_at(profiles, offsets, polarity, low, high, config)

    half = max(2.0 * float(offsets[1] - offsets[0]), 0.5 * config.search_px)
    positions, reasons = first, first_reasons
    # The window centre snaps to a sample index, so one pass can still leave the
    # edge up to half a step off centre. Two more passes drive that to zero; it
    # converges immediately because each pass starts nearer than the last.
    for _ in range(3):
        low, high = _recentred_plateaus(profiles, offsets, positions, half)
        updated, updated_reasons = _crossings_at(profiles, offsets, polarity, low, high, config)
        keep = np.isfinite(updated)
        positions = np.where(keep, updated, positions)
        reasons = np.where(keep, updated_reasons, reasons)

    # Fall back to the first pass where re-centring found no crossing at all.
    fallback = ~np.isfinite(positions) & np.isfinite(first)
    positions[fallback] = first[fallback]
    reasons[fallback] = REJECT_OK
    return positions, reasons.astype(np.int16), high - low


def _estimate_gradient_peak(
    profiles: np.ndarray,
    offsets: np.ndarray,
    polarity: float,
    config: RefineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient-based edge detection: the |dI/dt| peak nearest the boundary.

    Direction-free by construction - it never asks whether intensity rises or
    falls outward - so it cannot be misled by the polarity ambiguity that
    afflicts a level-crossing estimator on a small feature.

    Candidates are local maxima of |dI/dt| passing a height and prominence test,
    and the one nearest ``t = 0`` wins.  Taking the global maximum instead is
    wrong: on a dim particle beside a bright one the neighbour's edge is at
    least as steep, so a global argmax silently measures the neighbour.
    """
    from scipy.ndimage import gaussian_filter1d

    step = float(offsets[1] - offsets[0])
    low, high = np.percentile(profiles, PROFILE_PERCENTILES, axis=1)
    contrast = high - low

    sigma_samples = max(config.deriv_sigma_px / step, 1e-6)
    smoothed = gaussian_filter1d(profiles, sigma=sigma_samples, axis=1, mode="nearest")
    magnitude = np.abs(np.gradient(smoothed, step, axis=1))

    n = profiles.shape[0]
    positions = np.full(n, np.nan)
    reasons = np.full(n, REJECT_OK, dtype=np.int16)

    for index in range(n):
        if contrast[index] < config.min_contrast:
            reasons[index] = REJECT_LOW_CONTRAST
            continue
        row = magnitude[index]
        peak_value = float(row.max())
        if peak_value <= 0.0:
            reasons[index] = REJECT_NO_CROSSING
            continue
        # Height and prominence are fractions of the strongest edge in this
        # window, so the test adapts to local contrast instead of assuming one.
        candidate = _nearest_candidate(
            row,
            offsets,
            min_height=config.peak_height_fraction * peak_value,
            min_prominence=config.peak_prominence_fraction * peak_value,
        )
        if candidate is None:
            reasons[index] = REJECT_NO_CROSSING
            continue
        positions[index] = offsets[candidate] + _parabolic_offset(row, candidate) * step
    return positions, reasons, contrast


def _erf_model(offsets: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``a + b erf(u)`` and its Jacobian for a batch of profiles.

    ``params`` is (N, 4) holding (a, b, t0, sigma).  Returns (N, M) predictions
    and the (N, M, 4) Jacobian.
    """
    from scipy.special import erf

    a = params[:, 0:1]
    b = params[:, 1:2]
    t0 = params[:, 2:3]
    sigma = np.maximum(params[:, 3:4], 1e-3)

    u = (offsets[None, :] - t0) / (np.sqrt(2.0) * sigma)
    erf_u = erf(u)
    prediction = a + b * erf_u

    # d/du erf(u) = 2/sqrt(pi) exp(-u^2)
    gaussian = (2.0 / np.sqrt(np.pi)) * np.exp(-(u**2))
    jacobian = np.empty(u.shape + (4,))
    jacobian[..., 0] = 1.0
    jacobian[..., 1] = erf_u
    jacobian[..., 2] = b * gaussian * (-1.0 / (np.sqrt(2.0) * sigma))
    jacobian[..., 3] = b * gaussian * (-u / sigma)
    return prediction, jacobian


def _estimate_erf(
    profiles: np.ndarray,
    offsets: np.ndarray,
    polarity: float,
    config: RefineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched damped Gauss-Newton fit of an erf edge to every profile.

    Fitting each vertex with a generic optimiser would take a fraction of a
    millisecond each, which becomes a minute on a frame with a hundred thousand
    vertices.  The model has an analytic Jacobian, so all vertices are solved
    together as a batch of 4x4 normal equations instead.
    """
    low, high = np.percentile(profiles, PROFILE_PERCENTILES, axis=1)
    contrast = high - low

    # Initialise from the threshold estimate; a good t0 is what keeps the fit to
    # a handful of iterations and stops it walking to a neighbouring edge.
    initial_t0, threshold_reasons, _ = _estimate_threshold(profiles, offsets, polarity, config)
    t0 = np.where(np.isfinite(initial_t0), initial_t0, 0.0)

    params = np.column_stack(
        [
            0.5 * (low + high),
            polarity * 0.5 * np.maximum(contrast, 1e-6),
            t0,
            np.full(profiles.shape[0], 1.0),
        ]
    )

    damping = 1e-3
    for _ in range(12):
        prediction, jacobian = _erf_model(offsets, params)
        residual = profiles - prediction
        jt_j = np.einsum("nmi,nmj->nij", jacobian, jacobian)
        jt_r = np.einsum("nmi,nm->ni", jacobian, residual)
        jt_j[:, np.arange(4), np.arange(4)] *= 1.0 + damping
        jt_j[:, np.arange(4), np.arange(4)] += 1e-12
        try:
            delta = np.linalg.solve(jt_j, jt_r[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:  # pragma: no cover - singular batch
            break
        params = params + delta
        params[:, 3] = np.clip(np.abs(params[:, 3]), 1e-2, 10.0)
        if np.nanmax(np.abs(delta)) < 1e-6:
            break

    prediction, _ = _erf_model(offsets, params)
    scale = np.maximum(contrast, 1e-6)
    residual_rms = np.sqrt(np.mean((profiles - prediction) ** 2, axis=1)) / scale

    positions = params[:, 2]
    widths = params[:, 3]
    reasons = np.full(profiles.shape[0], REJECT_OK, dtype=np.int16)
    reasons[contrast < config.min_contrast] = REJECT_LOW_CONTRAST
    # A vertex whose initialisation already failed has no trustworthy fit.
    reasons[(reasons == REJECT_OK) & (threshold_reasons != REJECT_OK)] = REJECT_NO_CROSSING
    reasons[(reasons == REJECT_OK) & (residual_rms > config.max_residual)] = REJECT_POOR_FIT
    reasons[(reasons == REJECT_OK) & ~np.isfinite(positions)] = REJECT_POOR_FIT
    return positions, reasons, contrast, widths


def _coherence_filter(
    displacement: np.ndarray,
    valid: np.ndarray,
    *,
    window: int = 7,
    threshold_mad: float = 3.0,
) -> np.ndarray:
    """Invalidate vertices whose shift disagrees with their neighbours'.

    A boundary is a continuous thing: adjacent vertices, one pixel apart, cannot
    genuinely disagree about where the edge is by several pixels.  A vertex that
    does has almost certainly locked onto a different feature, and one such
    vertex drags a visible spike across the contour.

    Comparison is against a local median with a MAD scale, so a genuinely
    curved or rough boundary is not penalised - only departures from what the
    immediate neighbourhood agrees on.
    """
    if threshold_mad <= 0 or valid.sum() < max(8, window):
        return valid

    n = displacement.size
    filled = np.where(valid, displacement, np.nan)
    # Circular neighbourhood medians; the ring has no ends.
    half = max(1, window // 2)
    padded = np.concatenate([filled[-half:], filled, filled[:half]])
    local = np.full(n, np.nan)
    for i in range(n):
        neighbourhood = np.concatenate([padded[i : i + half], padded[i + half + 1 : i + window]])
        finite = neighbourhood[np.isfinite(neighbourhood)]
        if finite.size:
            local[i] = np.median(finite)

    residual = np.abs(filled - local)
    finite = residual[np.isfinite(residual)]
    if finite.size < 4:
        return valid
    scale = 1.4826 * np.median(np.abs(finite - np.median(finite)))
    limit = max(threshold_mad * scale, 0.5)
    keep = ~(np.isfinite(residual) & (residual > limit))

    # The local test cannot see a run of outliers longer than its own window:
    # such a run dominates its neighbourhood, the median follows it, and the
    # residual stays small. A region-level test does see it, because the run
    # still departs from what the whole contour agrees on. Using the median
    # keeps a *uniformly* displaced boundary - where the mask really was off by
    # a constant - entirely intact; only a subset disagreeing with its own
    # region is removed.
    observed = filled[np.isfinite(filled)]
    if observed.size >= 8:
        centre = np.median(observed)
        spread = 1.4826 * np.median(np.abs(observed - centre))
        region_limit = max(threshold_mad * spread, 1.0)
        far = np.isfinite(filled) & (np.abs(filled - centre) > region_limit)
        keep &= ~far
    return valid & keep


def _interpolate_short_gaps(
    displacement: np.ndarray,
    valid: np.ndarray,
    *,
    spacing_px: float,
    max_gap_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bridge short runs of rejected vertices around the closed ring.

    Filling a rejected vertex with zero would plant an artificial notch in the
    contour an order of magnitude larger than the roughness being measured, so a
    gap is either interpolated from its neighbours - when it is short enough
    that the boundary cannot have done anything interesting inside it - or left
    out of the polygon entirely.
    """
    n = valid.size
    if n == 0 or valid.all() or not valid.any() or max_gap_px <= 0:
        return displacement, valid

    filled = displacement.copy()
    repaired = valid.copy()
    max_run = max(1, int(np.floor(max_gap_px / max(spacing_px, 1e-9))))

    # Rotate so index 0 is valid, making the runs contiguous rather than wrapped.
    start = int(np.flatnonzero(valid)[0])
    order = (np.arange(n) + start) % n
    d = filled[order]
    v = repaired[order]

    index = 0
    while index < n:
        if v[index]:
            index += 1
            continue
        end = index
        while end < n and not v[end]:
            end += 1
        run = end - index
        left = index - 1  # valid by construction
        right = end % n
        if run <= max_run and v[right]:
            weights = np.arange(1, run + 1) / (run + 1)
            d[index:end] = d[left] + weights * (d[right] - d[left])
            v[index:end] = True
        index = end

    inverse = np.empty(n, dtype=int)
    inverse[order] = np.arange(n)
    return d[inverse], v[inverse]


def adaptive_search_px(mask: np.ndarray, config: RefineConfig) -> float:
    """Search radius that cannot span the feature it is measuring.

    The window has to reach past the edge into background on one side and into
    material on the other.  On a feature whose half-width is comparable to the
    configured radius it does neither: it swallows the object and reaches the
    next one, and every selection rule then has several edges to choose between.
    The inscribed radius - the distance transform's maximum - is the right scale
    because it tracks the *narrowest* part of an elongated shape, which the
    equivalent circular radius does not.
    """
    if not config.adaptive_search:
        return config.search_px
    from scipy import ndimage

    inscribed = float(ndimage.distance_transform_edt(mask).max()) if mask.any() else 0.0
    if inscribed <= 0:
        return config.search_px
    return float(
        np.clip(config.search_fraction * inscribed, config.min_search_px, config.search_px)
    )


def refine_contour(
    contour: Contour,
    image01: np.ndarray,
    config: RefineConfig,
    *,
    spacing_px: float = 1.0,
    search_px: float | None = None,
    _prepared: tuple[np.ndarray, int] | None = None,
) -> RefinedContour:
    """Measure the true edge position at every vertex of a coarse contour."""
    points = np.asarray(contour.points, dtype=np.float64)
    normals = contour.normals
    if normals is None:
        raise ValueError("contour has no normals; call contours.attach_normals first")
    n = points.shape[0]
    if n == 0:
        empty = np.zeros(0)
        return RefinedContour(
            base_points=points,
            normals=np.zeros((0, 2)),
            displacement=empty,
            valid=np.zeros(0, dtype=bool),
            reasons=np.zeros(0, dtype=np.int16),
            edge_width=empty,
            contrast=empty,
            is_hole=contour.is_hole,
            region_id=contour.region_id,
        )

    radius = float(search_px) if search_px is not None else config.search_px
    profiles, offsets, in_bounds = sample_profiles(
        image01,
        points,
        normals,
        search_px=radius,
        step_px=config.step_px,
        interp_order=config.interp_order,
        _prepared=_prepared,
    )
    polarity = _contour_polarity(profiles[in_bounds] if in_bounds.any() else profiles, offsets)

    widths = np.full(n, np.nan)
    if config.estimator == "threshold":
        positions, reasons, contrast = _estimate_threshold(profiles, offsets, polarity, config)
    elif config.estimator == "gradient_peak":
        positions, reasons, contrast = _estimate_gradient_peak(profiles, offsets, polarity, config)
    elif config.estimator == "erf":
        positions, reasons, contrast, widths = _estimate_erf(profiles, offsets, polarity, config)
    else:  # pragma: no cover - the config Literal prevents this
        raise ValueError(f"unknown estimator {config.estimator!r}")

    # A vertex whose sampling window left the image was never measurable.
    reasons[~in_bounds] = REJECT_OUT_OF_BOUNDS
    positions[~in_bounds] = np.nan

    # Reject rather than clip at the search limit: clamping would censor the
    # distribution toward zero and flatter the refinement.
    limit = config.max_shift_px if config.max_shift_px is not None else radius
    finite = np.isfinite(positions)
    at_limit = finite & (np.abs(positions) >= limit - 0.5 * config.step_px)
    reasons[(reasons == REJECT_OK) & at_limit] = REJECT_AT_SEARCH_LIMIT
    reasons[(reasons == REJECT_OK) & ~finite] = REJECT_NO_CROSSING

    valid = reasons == REJECT_OK
    displacement = np.where(valid, positions, np.nan)
    # A neighbour-locked vertex can pass every per-vertex test and still be
    # obviously wrong relative to the rest of the contour.
    coherent = _coherence_filter(displacement, valid, threshold_mad=config.coherence_mad)
    reasons[valid & ~coherent] = REJECT_INCOHERENT
    valid = coherent
    displacement = np.where(valid, displacement, np.nan)
    displacement, valid = _interpolate_short_gaps(
        displacement, valid, spacing_px=spacing_px, max_gap_px=config.max_gap_px
    )

    return RefinedContour(
        base_points=points,
        normals=normals,
        displacement=displacement,
        valid=valid,
        reasons=reasons,
        edge_width=widths,
        contrast=contrast,
        is_hole=contour.is_hole,
        region_id=contour.region_id,
        meta={
            "estimator": config.estimator,
            "search_px": radius,
            "polarity": "rising_outward" if polarity > 0 else "falling_outward",
            "profiles_shape": tuple(profiles.shape),
        },
    )


def edge_strength_along(points: np.ndarray, image01: np.ndarray, *, sigma: float = 1.0,
                        magnitude: np.ndarray | None = None) -> float:
    """Mean |grad I| sampled on a contour - does it sit on an edge?

    The only way to ask "did refinement improve this boundary" without ground
    truth.  A boundary that tracks a real edge samples a higher gradient than one
    that does not, so comparing this between the coarse and refined rings says
    directly whether the move was an improvement, a wash, or a regression.

    It is not a proxy for accuracy: a contour can sit on a strong edge and still
    be the wrong edge.  It is a regression detector, which is what was missing.
    ``magnitude`` may contain the gradient already computed on this same image
    with this sigma, so multiple contours need not repeat the image filtering.
    """
    from scipy.ndimage import gaussian_gradient_magnitude, map_coordinates

    if points.shape[0] < 3 or not np.isfinite(points).all():
        return float("nan")
    if magnitude is None:
        magnitude = gaussian_gradient_magnitude(np.asarray(image01, dtype=np.float64), sigma)
    sampled = map_coordinates(magnitude, [points[:, 0], points[:, 1]], order=1, mode="nearest")
    return float(np.mean(sampled))


def refine_all(
    contours: list[Contour],
    image01: np.ndarray,
    config: RefineConfig,
    *,
    spacing_px: float = 1.0,
    search_px: list[float] | None = None,
) -> list[RefinedContour]:
    """Refine every contour against the same unmodified image."""
    if not contours:
        return []
    prepared = _prepare_profile_image(image01, config.interp_order)
    radii = search_px or [None] * len(contours)
    return [
        refine_contour(c, image01, config, spacing_px=spacing_px, search_px=r, _prepared=prepared)
        for c, r in zip(contours, radii)
    ]
