"""Scene-level paired comparisons over a ``repeatability.json``.

The repeatability harness (:mod:`burst_diffusion.repeatability`) measures every
method on identical sources, seeds, crops, and CD sites, so two methods can be
compared *paired* per scene.  This module is the one place that comparison is
done, so that two mistakes made once in a hand analysis cannot recur:

- **The scene (source) is the independent unit.**  Sites inside one scene
  share the same noisy frames, registration, and model output; treating the
  37 sites of a 10-scene split as 37 independent observations pseudoreplicates
  and overstates significance.  Every metric here is reduced to one value per
  scene before the paired test.
- **``scene_sigmas_px`` stores 1-sigma values.**  Only the harness summary
  multiplies by three.  This module reports CD as 3-sigma everywhere, and says
  so in its output.

Per-scene metrics (``None`` when a scene lacks the data for either arm, in
which case that scene is dropped from the pair):

- ``cd_3sigma_px``: 3 x the scene's pooled c4-debiased CD sigma;
- ``cd_abs_bias_px``: mean over the scene's sites of |mean measured CD - clean CD|;
- ``cd_bias_px``: the same, signed;
- ``pixel_sigma``: mean per-pixel repeatability sigma over the scene;
- ``psnr_db``: mean PSNR of the scene's realizations.

The test is a two-sided paired Student t on the per-scene differences
``arm - control``.  With ~10 scenes its power is low, so a non-significant
result is *not* evidence of equality; and with several arms against one
control the Bonferroni threshold ``0.05 / arms`` is printed alongside.  The
t-distribution tail is evaluated from the regularized incomplete beta function
(no SciPy dependency).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

METRICS: tuple[str, ...] = (
    "cd_3sigma_px",
    "cd_abs_bias_px",
    "cd_bias_px",
    "pixel_sigma",
    "psnr_db",
)
# Which direction of ``arm - control`` is an improvement; signed bias has none.
LOWER_IS_BETTER: dict[str, bool | None] = {
    "cd_3sigma_px": True,
    "cd_abs_bias_px": True,
    "cd_bias_px": None,
    "pixel_sigma": True,
    "psnr_db": False,
}
METRIC_LABELS: dict[str, str] = {
    "cd_3sigma_px": "CD 3-sigma per scene, px (lower is better)",
    "cd_abs_bias_px": "|CD bias| per scene, px (lower is better)",
    "cd_bias_px": "signed CD bias per scene, px (closer to 0 is better; no test direction)",
    "pixel_sigma": "pixel sigma per scene, intensity in [0, 1] (lower is better)",
    "psnr_db": "PSNR per scene, dB (higher is better)",
}


# ---------------------------------------------------------------------------
# Student t tail without SciPy


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    max_iterations = 300
    epsilon = 3.0e-14
    tiny = 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b) for a, b > 0 and 0 <= x <= 1."""
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if x == 0.0 or x == 1.0:
        return x
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t: float, dof: int) -> float:
    """P(|T_dof| >= |t|) for a Student t variable with ``dof`` degrees of freedom."""
    if dof < 1:
        raise ValueError(f"dof must be >= 1, got {dof}")
    if math.isnan(t):
        return math.nan
    if math.isinf(t):
        return 0.0
    x = dof / (dof + t * t)
    return regularized_incomplete_beta(0.5 * dof, 0.5, x)


def sign_test_two_sided_p(negative: int, positive: int) -> float | None:
    """Exact two-sided binomial sign test on the non-zero differences.

    Robust companion to the paired t: with ten scenes one scene with a large
    opposite-sign delta can drive the t-test to p ~ .3 while nine of ten
    scenes agree in direction (which this test scores at p ~ .02).  Neither
    replaces the other -- the t weighs magnitudes, the sign test only
    directions -- so both are reported.  ``None`` when there are no
    non-zero differences.
    """
    n = negative + positive
    if n == 0:
        return None
    k = min(negative, positive)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0**n
    return float(min(1.0, 2.0 * tail))


def paired_t(deltas: Sequence[float]) -> dict:
    """Mean, t statistic, two-sided p, sign-test p, and n for paired differences.

    ``t``/``p`` are ``None`` with fewer than two pairs or a zero-variance
    difference (then the mean is the whole story).
    """
    values = np.asarray(list(deltas), dtype=np.float64)
    n = int(values.size)
    if n == 0:
        return {"n": 0, "mean": None, "t": None, "p": None, "negative": 0, "sign_p": None}
    mean = float(values.mean())
    negative = int((values < 0.0).sum())
    positive = int((values > 0.0).sum())
    sign_p = sign_test_two_sided_p(negative, positive)
    if n < 2:
        return {"n": n, "mean": mean, "t": None, "p": None, "negative": negative, "sign_p": sign_p}
    spread = float(values.std(ddof=1))
    # Identical differences give a spread of ~1e-17, not exactly zero.
    if spread <= 1e-12 * max(1.0, abs(mean)):
        return {"n": n, "mean": mean, "t": None, "p": None, "negative": negative, "sign_p": sign_p}
    t = mean / (spread / math.sqrt(n))
    return {
        "n": n,
        "mean": mean,
        "t": float(t),
        "p": float(student_t_two_sided_p(t, n - 1)),
        "negative": negative,
        "sign_p": sign_p,
    }


# ---------------------------------------------------------------------------
# per-scene metrics from the harness output


def _scene_bias(sites: Sequence[Mapping], detail: Sequence[Mapping]) -> tuple[float, float] | None:
    """(mean |bias|, mean signed bias) over the scene's sites with any valid CD."""
    absolute: list[float] = []
    signed: list[float] = []
    for site, measured in zip(sites, detail):
        values = [v for v in measured["cd_values"] if v is not None]
        if not values:
            continue
        bias = float(np.mean(values)) - float(site["clean_cd"])
        absolute.append(abs(bias))
        signed.append(bias)
    if not absolute:
        return None
    return float(np.mean(absolute)), float(np.mean(signed))


def per_scene_metrics(results: Mapping, method: str) -> dict[str, list[float | None]]:
    """One value per source (in ``per_source`` order) for every metric in
    :data:`METRICS`; ``None`` where the scene has no value for that metric."""
    if method not in results["methods"]:
        raise KeyError(f"method {method!r} not in results (have {sorted(results['methods'])})")
    out: dict[str, list[float | None]] = {name: [] for name in METRICS}
    for record in results["per_source"]:
        entry = record["methods"][method]
        sigma = entry.get("cd_scene_sigma_px")
        out["cd_3sigma_px"].append(None if sigma is None else 3.0 * float(sigma))
        bias = _scene_bias(record.get("sites", []), entry.get("cd_sites", []))
        out["cd_abs_bias_px"].append(None if bias is None else bias[0])
        out["cd_bias_px"].append(None if bias is None else bias[1])
        pixel = entry.get("pixel_sigma_mean")
        out["pixel_sigma"].append(None if pixel is None else float(pixel))
        psnr = entry.get("psnr") or []
        out["psnr_db"].append(float(np.mean(psnr)) if psnr else None)
    return out


def resolve_method(results: Mapping, name: str) -> str:
    """Accept an exact method name or a bare arm name (``ft_noisy`` ->
    ``one_shot@ft_noisy`` when that is the unique match)."""
    methods = results["methods"]
    if name in methods:
        return name
    candidates = [m for m in methods if m.endswith(f"@{name}")]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise KeyError(f"no method named {name!r} (have {sorted(methods)})")
    raise KeyError(f"arm {name!r} is ambiguous: {sorted(candidates)}; give the full method name")


def paired_report(
    results: Mapping,
    *,
    control: str,
    arms: Sequence[str] | None = None,
    metrics: Sequence[str] = METRICS,
) -> dict:
    """Paired scene-level statistics of every arm against ``control``.

    Returns ``{"control", "arms": {arm: {metric: paired_t(...) + "scenes"}},
    "bonferroni_alpha", "metrics"}``.  ``arms`` defaults to every method except
    the control.
    """
    control_name = resolve_method(results, control)
    if arms is None:
        arm_names = [m for m in results["methods"] if m != control_name]
    else:
        arm_names = [resolve_method(results, a) for a in arms]
        if control_name in arm_names:
            raise ValueError(f"control {control_name!r} cannot also be an arm")
    for metric in metrics:
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")
    control_values = per_scene_metrics(results, control_name)
    sources = list(results.get("source_indices", range(len(results["per_source"]))))
    report: dict = {
        "control": control_name,
        "metrics": list(metrics),
        "unit": "scene (source); CD values are 3-sigma",
        "bonferroni_alpha": 0.05 / max(1, len(arm_names)),
        "arms": {},
    }
    for arm in arm_names:
        arm_values = per_scene_metrics(results, arm)
        per_metric: dict = {}
        for metric in metrics:
            deltas: list[float] = []
            scenes: list[int] = []
            for source, a, c in zip(sources, arm_values[metric], control_values[metric]):
                if a is None or c is None:
                    continue
                deltas.append(a - c)
                scenes.append(int(source))
            stats = paired_t(deltas)
            stats["scenes"] = scenes
            stats["deltas"] = deltas
            per_metric[metric] = stats
        report["arms"][arm] = per_metric
    return report


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def format_markdown(report: Mapping) -> str:
    """One table per metric: arm | mean delta | delta < 0 | t | p."""
    lines = [
        f"# Paired scene-level comparison vs `{report['control']}`",
        "",
        "Unit: one value per scene (source); sites within a scene are not "
        "independent. CD figures are **3-sigma** (the JSON's `scene_sigmas_px` "
        "is 1-sigma). Two-sided paired t on `arm - control` (weighs magnitudes) "
        "and an exact sign test on the direction counts (robust to one outlier "
        "scene); with ~10 scenes both are weak, so p > .05 is not evidence of "
        f"equality. Bonferroni threshold across the {len(report['arms'])} arm(s): "
        f"alpha = {report['bonferroni_alpha']:.4f}.",
        "",
    ]
    for metric in report["metrics"]:
        lines.append(f"## {METRIC_LABELS.get(metric, metric)}")
        lines.append("")
        lines.append("| arm | n scenes | mean delta | delta < 0 | t | p (t) | p (sign) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for arm, per_metric in report["arms"].items():
            stats = per_metric[metric]
            lines.append(
                f"| {arm} | {stats['n']} | {_fmt(stats['mean'], '+.4f')} | "
                f"{stats['negative']}/{stats['n']} | {_fmt(stats['t'], '+.2f')} | "
                f"{_fmt(stats['p'], '.4f')} | {_fmt(stats['sign_p'], '.4f')} |"
            )
        lines.append("")
    return "\n".join(lines)


def load_results(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
