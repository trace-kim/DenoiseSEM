from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from burst_diffusion.cli import app
from burst_diffusion.paired import (
    METRICS,
    format_markdown,
    paired_report,
    paired_t,
    per_scene_metrics,
    regularized_incomplete_beta,
    resolve_method,
    student_t_two_sided_p,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Student t without SciPy


@pytest.mark.parametrize(
    ("t", "dof", "expected"),
    [
        (0.0, 9, 1.0),
        (1.0, 1, 0.5),  # Cauchy: P(|T| > 1) = 1/2
        (2.262157, 9, 0.05),  # textbook two-sided 5% critical value
        (3.59, 9, 0.00587),  # the ft_consist-vs-ft_noisy figure from the audit
        (-3.04, 9, 0.01402),
        (2.0, 60, 0.0500),
    ],
)
def test_student_t_two_sided_tail_matches_reference_values(
    t: float, dof: int, expected: float
) -> None:
    assert student_t_two_sided_p(t, dof) == pytest.approx(expected, rel=2e-3, abs=2e-4)


def test_incomplete_beta_endpoints_and_symmetry() -> None:
    assert regularized_incomplete_beta(2.0, 3.0, 0.0) == 0.0
    assert regularized_incomplete_beta(2.0, 3.0, 1.0) == 1.0
    # I_x(a, b) = 1 - I_{1-x}(b, a)
    assert regularized_incomplete_beta(2.5, 0.5, 0.3) == pytest.approx(
        1.0 - regularized_incomplete_beta(0.5, 2.5, 0.7), abs=1e-12
    )
    assert student_t_two_sided_p(math.inf, 5) == 0.0
    with pytest.raises(ValueError):
        student_t_two_sided_p(1.0, 0)


def test_paired_t_degenerate_inputs() -> None:
    assert paired_t([]) == {"n": 0, "mean": None, "t": None, "p": None, "negative": 0}
    single = paired_t([-0.3])
    assert single["n"] == 1 and single["mean"] == pytest.approx(-0.3) and single["t"] is None
    flat = paired_t([0.1, 0.1, 0.1])
    assert flat["t"] is None and flat["negative"] == 0
    stats = paired_t([-1.0, -2.0, -3.0, -4.0])
    assert stats["negative"] == 4
    assert stats["t"] == pytest.approx(-2.5 / (math.sqrt(5.0 / 3.0) / 2.0))


# ---------------------------------------------------------------------------
# per-scene reduction and pairing


def _results() -> dict:
    """Three scenes; the control lacks a CD sigma on scene 2 (dropped from the
    CD pair but kept for pixel sigma / PSNR)."""

    def method(sigmas, cds, pixel, psnr):
        return [
            {
                "cd_scene_sigma_px": sigma,
                "cd_sites": [{"cd_values": values} for values in cd],
                "pixel_sigma_mean": px,
                "psnr": [p, p + 0.2],
            }
            for sigma, cd, px, p in zip(sigmas, cds, pixel, psnr)
        ]

    control = method(
        sigmas=[0.10, 0.20, None],
        cds=[[[10.2, 10.4]], [[20.5, 20.7], [20.1, None]], [[30.0]]],
        pixel=[0.010, 0.020, 0.030],
        psnr=[30.0, 31.0, 32.0],
    )
    arm = method(
        sigmas=[0.05, 0.10, 0.15],
        cds=[[[10.1, 10.1]], [[20.2, 20.2], [20.0, 20.0]], [[30.5]]],
        pixel=[0.005, 0.010, 0.015],
        psnr=[29.0, 30.0, 31.0],
    )
    sites = [
        [{"clean_cd": 10.0}],
        [{"clean_cd": 20.0}, {"clean_cd": 20.0}],
        [{"clean_cd": 30.0}],
    ]
    per_source = [
        {
            "source_index": index,
            "sites": sites[index],
            "methods": {"one_shot@control": control[index], "one_shot@arm": arm[index]},
        }
        for index in range(3)
    ]
    return {
        "source_indices": [5, 21, 87],
        "methods": {"one_shot@control": {}, "one_shot@arm": {}},
        "per_source": per_source,
    }


def test_per_scene_metrics_reduce_sites_to_one_value_per_scene() -> None:
    values = per_scene_metrics(_results(), "one_shot@control")
    assert set(values) == set(METRICS)
    # CD is reported as 3-sigma; a missing scene sigma stays None.
    assert values["cd_3sigma_px"] == [pytest.approx(0.30), pytest.approx(0.60), None]
    # scene 2: sites biased +0.6 and +0.1 (the None measurement is skipped) -> mean 0.35
    assert values["cd_abs_bias_px"][1] == pytest.approx(0.35)
    assert values["cd_bias_px"][1] == pytest.approx(0.35)
    assert values["psnr_db"] == [pytest.approx(30.1), pytest.approx(31.1), pytest.approx(32.1)]
    with pytest.raises(KeyError):
        per_scene_metrics(_results(), "one_shot@missing")


def test_paired_report_pairs_only_scenes_both_arms_measured() -> None:
    report = paired_report(_results(), control="control")
    assert report["control"] == "one_shot@control"
    assert list(report["arms"]) == ["one_shot@arm"]
    cd = report["arms"]["one_shot@arm"]["cd_3sigma_px"]
    assert cd["n"] == 2 and cd["scenes"] == [5, 21]
    assert cd["deltas"] == [pytest.approx(-0.15), pytest.approx(-0.30)]
    assert cd["negative"] == 2
    pixel = report["arms"]["one_shot@arm"]["pixel_sigma"]
    assert pixel["n"] == 3 and pixel["negative"] == 3
    psnr = report["arms"]["one_shot@arm"]["psnr_db"]
    assert psnr["mean"] == pytest.approx(-1.0)
    assert psnr["t"] is None  # zero-variance difference -> no test
    assert report["bonferroni_alpha"] == pytest.approx(0.05)


def test_resolve_method_accepts_bare_arm_names_and_rejects_ambiguity() -> None:
    results = _results()
    assert resolve_method(results, "arm") == "one_shot@arm"
    assert resolve_method(results, "one_shot@arm") == "one_shot@arm"
    with pytest.raises(KeyError, match="no method"):
        resolve_method(results, "nope")
    results["methods"]["iter_prediction@arm"] = {}
    with pytest.raises(KeyError, match="ambiguous"):
        resolve_method(results, "arm")
    with pytest.raises(ValueError, match="cannot also be an arm"):
        paired_report(_results(), control="control", arms=["control"])


def test_markdown_report_states_the_unit_and_three_sigma() -> None:
    text = format_markdown(paired_report(_results(), control="control"))
    assert "3-sigma" in text and "scene" in text
    assert "| one_shot@arm | 2 | -0.2250 | 2/2 |" in text


def test_paired_cli_writes_the_markdown_report(tmp_path: Path) -> None:
    results_path = tmp_path / "repeatability.json"
    results_path.write_text(json.dumps(_results()), encoding="utf-8")
    out = tmp_path / "paired.md"
    result = runner.invoke(
        app,
        ["paired", "--results", str(results_path), "--control", "control", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "Paired scene-level comparison vs `one_shot@control`" in out.read_text(encoding="utf-8")
