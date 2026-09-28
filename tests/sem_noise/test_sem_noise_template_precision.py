from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from scipy.special import erf
import torch

from sem_noise.template_precision import fit_template_batch


def disk():
    yy, xx = np.mgrid[:64, :64]
    radius = np.hypot(yy - 31.5, xx - 31.5)
    pixels = np.rint(120 + 80 * erf((radius - 15.3) / 1.3 / np.sqrt(2))).astype(np.uint8)
    return pixels, abs(radius - 15.3) < 5


@pytest.mark.parametrize("poisson", [False, True])
def test_template_scale_is_unbiased_and_standard_errors_match_monte_carlo(poisson):
    template, annulus = disk()
    rng = np.random.default_rng(89)
    truth = np.repeat(template[None], 320, axis=0).astype(float)
    if poisson:
        truth = rng.poisson(truth * 8) / 8
    images = np.rint(truth + rng.normal(0, 3, truth.shape)).clip(0, 255).astype(np.uint8)
    results = fit_template_batch(np.repeat(template[None], len(images), 0), images,
                                 np.repeat(annulus[None], len(images), 0))
    assert Counter(r["status"] for r in results) == {"valid": 320}
    scales = np.array([r["scale"] for r in results])
    errors = np.array([r["scale_se"] for r in results])
    empirical = scales.std(ddof=1)
    assert abs(scales.mean() - 1) < 3 * empirical / np.sqrt(len(images))
    assert errors.mean() == pytest.approx(empirical, rel=.1)


def test_template_physical_scale_and_nuisance_parameters():
    template, annulus = disk()
    yy, xx = np.mgrid[:64, :64]
    scale, dy, dx = 1.015, .3, -.4
    radius = np.hypot((yy - 31.5 - dy) / scale, (xx - 31.5 - dx) / scale)
    image = np.rint(1.04 * (120 + 80 * erf((radius - 15.3) / 1.3 / np.sqrt(2))) + 3).astype(np.uint8)
    fit = fit_template_batch(template[None], image[None], annulus[None])[0]
    assert fit["status"] == "valid"
    assert fit["scale"] == pytest.approx(scale, abs=.002)
    assert fit["dy_px"] == pytest.approx(dy, abs=.04)
    assert fit["dx_px"] == pytest.approx(dx, abs=.04)
    assert fit["gain"] == pytest.approx(1.04, abs=.02)
    assert fit["offset_dn"] == pytest.approx(3, abs=2)


def test_template_patternless_frames_skip_before_any_sampling(monkeypatch):
    template, annulus = disk()
    rng = np.random.default_rng(21)
    images = np.rint(128 + rng.normal(0, 4, (2, 64, 64))).astype(np.uint8)
    images[0] = 128
    monkeypatch.setattr(torch.nn.functional, "grid_sample", lambda *a, **k: pytest.fail("blank images must skip fitting"))
    fits = fit_template_batch(np.repeat(template[None], 2, 0), images, np.repeat(annulus[None], 2, 0))
    assert all(f["status"] == "skipped_low_contrast" and f["scale_se"] is None for f in fits)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        fit_template_batch(template[None], images[:1], annulus[None], device="cuda:0")
    with pytest.raises(ValueError, match="uint8"):
        fit_template_batch(template[None].astype(float), images[:1], annulus[None])


def test_template_failed_fits_do_not_report_precision():
    template, annulus = disk()
    no_support = fit_template_batch(template[None], template[None], np.zeros_like(annulus[None]))[0]
    assert no_support["status"] == "insufficient_support" and no_support["scale_se"] is None
    unfinished = fit_template_batch(template[None], np.roll(template, 2, axis=0)[None],
                                    annulus[None], max_iterations=1)[0]
    assert unfinished["status"] == "not_converged" and unfinished["scale_se"] is None
    with pytest.raises(ValueError, match="finite"):
        fit_template_batch(template[None], template[None], annulus[None], initial_shifts_yx=[[np.nan, 0]])
