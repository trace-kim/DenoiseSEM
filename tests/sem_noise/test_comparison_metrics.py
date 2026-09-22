import numpy as np
import pytest

from sem_noise.comparison_metrics import difference_rgb, native_series_statistics


def test_native_brightness_and_variation_preserve_offsets_scales_and_repeats():
    raw = np.array([[[10, 30], [20, 40]], [[12, 32], [22, 42]], [[12, 32], [22, 42]]], dtype=np.uint8)
    original = raw.copy()
    for changed in (raw, raw + 7, raw * 2):
        rows, std = native_series_statistics(iter(changed))
        np.testing.assert_allclose([r["mean_dn"] for r in rows], changed.mean(axis=(1, 2)))
        np.testing.assert_allclose(std, changed.astype(float).std(axis=0, ddof=1))
        assert len(rows) == 3  # Duplicate acquisitions are not removed.
    np.testing.assert_array_equal(raw, original)
    assert native_series_statistics([raw[0]])[1] is None


def test_float_predictions_cannot_enter_native_analysis():
    with pytest.raises(ValueError, match="uint8"):
        native_series_statistics([np.ones((8, 8)) * 120.49])
    with pytest.raises(ValueError, match="identical dimensions"):
        native_series_statistics([np.zeros((8, 8), dtype=np.uint8), np.zeros((9, 8), dtype=np.uint8)])


def test_difference_map_is_signed_fixed_scale_and_does_not_correct_pixels():
    raw = np.array([[20, 20, 20]], dtype=np.uint8)
    output = np.array([[12, 20, 28]], dtype=np.uint8)
    rgb = difference_rgb(output, raw, 8)
    np.testing.assert_array_equal(rgb, [[[0, 0, 255], [255, 255, 255], [255, 0, 0]]])
    np.testing.assert_array_equal(output, [[12, 20, 28]])
    with pytest.raises(ValueError):
        difference_rgb(output.astype(float), raw, 8)


def test_cuda_native_statistics_match_cpu_without_full_float_downloads(fake_cupy):
    rng = np.random.default_rng(921)
    frames = rng.integers(0, 256, (17, 48, 64), dtype=np.uint8)
    before = frames.copy()
    expected_rows, expected_std = native_series_statistics(iter(frames))
    rows, std = native_series_statistics(iter(frames), device="cuda:2")
    assert rows == expected_rows
    np.testing.assert_array_equal(std, expected_std)
    np.testing.assert_array_equal(frames, before)
    assert fake_cupy.devices == [2]
    assert len(fake_cupy.downloads) == 2  # Frame summaries and the final SD map.
    assert native_series_statistics([frames[0]], device="cuda")[1] is None
    for bad in ([], [frames[0].astype(float)], [frames[0], frames[0][:-1]]):
        with pytest.raises(ValueError):
            native_series_statistics(bad, device="cuda:0")
