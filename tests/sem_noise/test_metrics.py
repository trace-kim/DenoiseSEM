from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")

from sem_noise.config import AnalysisConfig
from sem_noise.metrics import analyze_mode, mean_variance, spatial_diagnostics, temporal_diagnostics


def test_white_noise_variance_and_averaging_match_known_truth() -> None:
    rng = np.random.default_rng(3)
    samples = rng.normal(0, 7, (128, 4096)) + np.linspace(100, 900, 4096)
    result = temporal_diagnostics(samples, np.arange(128), samples.mean(axis=1), AnalysisConfig(), 2.0)
    assert abs(result["acf"][0]["pixel_acf"]) < 0.03
    for row in result["averaging"]:
        assert row["pixel_allan_deviation_dn"] == pytest.approx(7 / np.sqrt(row["block_frames"]), rel=0.06)
    assert result["averaging"][-1]["block_frames"] == 32
    assert result["averaging"][-1]["tau_s"] == 64
    assert result["averaging"][-1]["disjoint_pairs"] == 2
    assert result["frequency_unit"] == "Hz"


def test_missing_frames_are_not_treated_as_adjacent() -> None:
    samples = np.array([[0, 0], [0, 0], [100, 100], [100, 100]], dtype=float)
    result = temporal_diagnostics(samples, np.array([0, 1, 4, 5]), samples.mean(axis=1), AnalysisConfig(min_frames=4), None)
    assert result["acf"][0]["pairs"] == 2
    assert result["averaging"] == []
    assert not result["periodogram_available"]


def test_irregular_time_disables_temporal_periodogram() -> None:
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(32, 50))
    result = temporal_diagnostics(samples, np.arange(32), samples.mean(axis=1), AnalysisConfig(), None, True)
    assert not result["periodogram_available"]
    assert result["averaging"][0]["tau_s"] is None


def test_temporal_correlation_and_drift_break_white_noise_averaging() -> None:
    rng = np.random.default_rng(42)
    samples = np.zeros((128, 2000))
    for i in range(1, 128):
        samples[i] = 0.8 * samples[i - 1] + rng.normal(size=2000)
    samples += np.arange(128)[:, None] * 0.1
    result = temporal_diagnostics(samples, np.arange(128), samples.mean(axis=1), AnalysisConfig(), None)
    assert result["acf"][0]["pixel_acf"] > 0.7
    assert result["averaging"][-1]["pixel_allan_deviation_dn"] > result["averaging"][-1]["white_noise_reference_dn"] * 4


def test_signal_dependent_variance_fit_recovers_slope_and_offset() -> None:
    rng = np.random.default_rng(5)
    means = np.linspace(50, 1000, 20000)
    samples = means + rng.normal(size=(128, len(means))) * np.sqrt(0.35 * means + 25)
    bins, fit = mean_variance(samples, 12)
    assert len(bins) == 12
    assert fit["available"]
    assert fit["slope_dn"] == pytest.approx(0.35, abs=0.012)
    assert fit["intercept_dn2"] == pytest.approx(25, abs=5)
    assert fit["r_squared"] > 0.99


def test_flat_signal_does_not_produce_a_noise_slope_claim() -> None:
    samples = np.random.default_rng(8).normal(500, 10, (128, 8000))
    _, fit = mean_variance(samples, 12)
    assert not fit["available"]
    assert "repeat-mean noise" in fit["reason"]


def test_native_moments_clipping_masks_and_finite_reference_correction() -> None:
    rng = np.random.default_rng(31)
    stack = rng.normal(500, 6, (64, 48, 48)).astype(np.float32)
    stack[:, 0, 0] = 0
    config = AnalysisConfig(registration="none", sample_pixels=1600, spatial_pairs=4)
    summary, maps, _ = analyze_mode(stack, np.zeros((64, 2)), np.ones(64, dtype=bool), np.arange(64),
                                     (slice(0, 48), slice(0, 48)), True, (0, 65535), config, None, False)
    assert not maps["valid_mask"][0, 0]
    assert summary["flat_temporal_sigma_dn"] == pytest.approx(6, rel=0.025)
    assert summary["residual_distribution"]["std_dn"] == pytest.approx(6, rel=0.025)
    assert summary["adjacent_difference_distribution"]["std_dn"] == pytest.approx(6, rel=0.03)
    assert summary["spatial"]["row_banding_ratio"] == pytest.approx(1, abs=0.35)


def test_spatial_spectrum_detects_scan_row_noise() -> None:
    rng = np.random.default_rng(20)
    stack = rng.normal(0, 1, (32, 64, 64)) + rng.normal(0, 4, (32, 64, 1))
    summary, psd = spatial_diagnostics(stack, np.zeros((32, 2)), np.arange(32), np.arange(32),
                                       (slice(0, 64), slice(0, 64)), True, AnalysisConfig())
    assert summary["row_banding_ratio"] > 30
    assert summary["column_banding_ratio"] < 2
    assert summary["acf"][0]["x_acf"] > 0.85
    assert np.mean(psd) == pytest.approx(17, rel=0.3)


def test_float64_samples_preserve_small_noise_on_large_offset() -> None:
    rng = np.random.default_rng(33)
    stack = rng.normal(1e9, 1, (32, 48, 48))
    result, _, _ = analyze_mode(stack, np.zeros((32, 2)), np.ones(32, dtype=bool), np.arange(32),
                                (slice(0, 48), slice(0, 48)), True, (None, None),
                                AnalysisConfig(registration="none", spatial_pairs=2), None, False)
    assert result["residual_distribution"]["std_dn"] == pytest.approx(1, rel=0.04)
