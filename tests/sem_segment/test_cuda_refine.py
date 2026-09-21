"""Exercise GPU orchestration with NumPy/SciPy stand-ins; never access a GPU.

These tests verify grouping, transfers, estimator rules and integration. Actual
CuPy kernel parity and wall-clock speed are checked by the server benchmark.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage, signal
from synthetic_images import blurred_disk, gaussian_blurred_step, make_config

from sem_segment import cuda
from sem_segment.config import RefineConfig
from sem_segment.contours import Contour
from sem_segment.pipeline import segment_image
from sem_segment.refine import _nearest_candidates, edge_strength_along, refine_all


class FakeCuPy:
    __version__ = "mock"

    def __init__(self):
        self.devices, self.uploads, self.downloads = [], [], []
        self.syncs = self.freed = 0
        self.cuda = SimpleNamespace(Device=self.device, Stream=self.stream,
                                    MemoryPool=self.pool, using_allocator=lambda _: nullcontext(),
                                    runtime=SimpleNamespace(getDeviceProperties=lambda _: {"name": b"mock GPU"}))

    def __getattr__(self, name):
        return getattr(np, name)

    def device(self, index):
        self.devices.append(index)
        return nullcontext()

    def stream(self, **kwargs):
        owner = self

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def synchronize(self):
                owner.syncs += 1

        return Stream()

    def pool(self):
        def free():
            self.freed += 1
        return SimpleNamespace(malloc=None, free_all_blocks=free)

    def asarray(self, array, dtype=None):
        self.uploads.append(np.shape(array))
        return np.array(array, dtype=dtype, copy=True)

    def asnumpy(self, array):
        self.downloads.append(array.shape)
        return np.array(array, copy=True)


@pytest.fixture
def fake_cuda(monkeypatch):
    cp = FakeCuPy()
    monkeypatch.setattr(cuda, "_load_cuda", lambda: (cp, ndimage, signal))
    return cp


@pytest.mark.parametrize("device", ["auto", "cuda:0,1", "cuda:-1", "mps", "0"])
def test_refine_rejects_unsupported_device(device):
    with pytest.raises(ValueError):
        RefineConfig(device=device)


@pytest.mark.parametrize("kwargs", [{"estimator": "erf"}, {"estimator": "threshold"}, {"enabled": False}])
def test_cuda_requires_supported_estimator(kwargs):
    with pytest.raises(ValueError, match="CUDA refinement requires"):
        RefineConfig(device="cuda:0", **kwargs)


def test_cpu_never_loads_cupy(monkeypatch):
    def forbidden():
        raise AssertionError("CPU path imported CuPy")
    monkeypatch.setattr(cuda, "_load_cuda", forbidden)
    result = segment_image(blurred_disk(), make_config())
    assert result.diagnostics.refinement["device"] == "cpu"
    assert result.regions


@pytest.mark.parametrize("order", [1, 3])
def test_batched_gpu_matches_cpu_with_mixed_radii_empty_contours_and_borders(fake_cuda, order):
    image = np.rint(gaussian_blurred_step() * 255) / 255
    original = image.copy()
    contours = []
    for x in (39.1, 41.2, 0.1, 39.7, 60.0):
        points = np.column_stack((np.linspace(0, 63, 91), np.full(91, x)))
        normals = np.tile([[0., 1.]], (len(points), 1))
        contours.append(Contour(points=points, normals=normals))
    contours.append(Contour(points=np.empty((0, 2)), normals=np.empty((0, 2))))
    radii = [6., 4.125, 6., 4.125, 6., 6.]
    config = RefineConfig(device="cuda:2", interp_order=order, cuda_batch_samples=1024)
    expected = refine_all(contours, image, config, search_px=radii)
    with cuda.CudaRefiner(config) as refiner:
        actual, strengths, timings = refiner.measure(contours, image, spacing_px=1., search_px=radii)
        assert refiner.describe()["device"] == "cuda:2"
        assert timings["refine"] >= 0 and timings["edge_strength"] >= 0
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a.polygon, b.polygon, rtol=0, atol=1e-12)
        np.testing.assert_allclose(a.contrast, b.contrast, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(a.valid, b.valid)
        np.testing.assert_array_equal(a.reasons, b.reasons)
        assert a.meta == b.meta
    expected_strengths = [[edge_strength_along(c.points, image), edge_strength_along(r.polygon, image)]
                         for c, r in zip(contours, expected)]
    np.testing.assert_allclose(strengths, expected_strengths, atol=1e-14, equal_nan=True)
    np.testing.assert_array_equal(image, original)
    assert fake_cuda.uploads.count(image.shape) == 1
    assert len([s for s in fake_cuda.downloads if len(s) == 2 and s[1] == 6]) > 5  # chunked across holes
    assert fake_cuda.devices == [2] and fake_cuda.freed == 1 and fake_cuda.syncs > 0
    with pytest.raises(RuntimeError, match="closed"):
        refiner.measure(contours, image, spacing_px=1., search_px=radii)


def test_gpu_peak_selection_keeps_prominence_plateaus_endpoints_and_ties(fake_cuda):
    rng = np.random.default_rng(452)
    strength = np.round(rng.random((60, 17)) * 4)
    strength[0] = 0  # no peaks
    strength[1] = [9, 0, 0, 0, 3, 3, 3, 0, 0, 0, 3, 3, 3, 0, 0, 0, 9]
    offsets = np.linspace(-4, 4, 17)
    low, prominence = np.full(60, .5), np.full(60, 1.)
    actual = _nearest_candidates(strength, offsets, min_height=low, min_prominence=prominence,
                                 xp=fake_cuda, find_peaks=signal.find_peaks)
    expected = []
    for row in strength:
        peaks, _ = signal.find_peaks(row, height=.5, prominence=1.)
        expected.append(peaks[np.argmin(np.abs(offsets[peaks]))] if len(peaks) else -1)
    np.testing.assert_array_equal(actual, expected)
    assert actual[1] == 5


def test_gpu_pipeline_reuses_one_backend_without_reusing_image_pixels(fake_cuda):
    config = make_config(refine={"device": "cuda:0"})
    cpu_config = make_config()
    images = [np.rint(blurred_disk(radius=r) * 255) / 255 for r in (24., 26.)]
    with cuda.CudaRefiner(config.refine) as refiner:
        for image in images:
            expected = segment_image(image, cpu_config)
            actual = segment_image(image, config, refiner=refiner)
            np.testing.assert_array_equal(actual.label_map(), expected.label_map())
            for a, b in zip(actual.refined, expected.refined):
                np.testing.assert_allclose(a.polygon, b.polygon, atol=1e-12)
                np.testing.assert_array_equal(a.reasons, b.reasons)
            for a, b in zip(actual.regions, expected.regions):
                assert a.refined.equivalent_diameter_px == pytest.approx(b.refined.equivalent_diameter_px, abs=1e-12)
                assert a.refined.major_axis_px == pytest.approx(b.refined.major_axis_px, abs=1e-12)
            assert actual.diagnostics.edge_strength_change == pytest.approx(expected.diagnostics.edge_strength_change)
            assert actual.provenance["refinement"]["backend"] == "cupy"
    assert fake_cuda.devices == [0]
    assert fake_cuda.uploads.count(images[0].shape) == 2


def test_unavailable_device_fails_explicitly(fake_cuda):
    def unavailable(_):
        raise RuntimeError("invalid device ordinal")
    fake_cuda.cuda.runtime.getDeviceProperties = unavailable
    with pytest.raises(RuntimeError, match="Cannot initialize.*cuda:3.*No CPU fallback"):
        cuda.CudaRefiner(RefineConfig(device="cuda:3"))


def test_missing_cupy_error_explains_installation(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(RuntimeError, match="requires CuPy.*cupy-cuda12x"):
        cuda.CudaRefiner(RefineConfig(device="cuda:0"))


def test_gpu_pool_released_on_failure_and_config_mismatch_rejected(fake_cuda):
    image = blurred_disk()
    with pytest.raises(ValueError, match="settings differ"):
        with cuda.CudaRefiner(RefineConfig(device="cuda")) as refiner:
            segment_image(image, make_config(), refiner=refiner)
    assert fake_cuda.freed == 1
    with cuda.CudaRefiner(RefineConfig(device="cuda")) as refiner:
        refined, strength, _ = refiner.measure([], image, spacing_px=1., search_px=[])
        assert not refined and strength.shape == (0, 2)
        with pytest.raises(ValueError, match="search radius"):
            refiner.measure([], image, spacing_px=1., search_px=[6.])


def test_server_benchmark_measures_pipeline_and_detects_changed_results(fake_cuda, tmp_path):
    from copy import deepcopy
    from tools.benchmark_sem_metrology import benchmark, check_parity
    from sem_segment.image_io import load_image
    from PIL import Image

    pixels = np.rint(255 * blurred_disk()).astype(np.uint8)
    path = tmp_path / "frame.png"
    Image.fromarray(np.repeat(pixels[..., None], 3, axis=2)).save(path)
    native = load_image(path).native
    record = benchmark([(path.name, native)], make_config(), repeats=2)
    assert record["status"] == "passed"
    row = record["images"][0]
    assert row["parity"]["regions"] == 1
    assert len(row["times_s"]["gpu"]) == 2
    assert row["median_stages_s"]["gpu"]["refine"] >= 0
    assert row["speedup"] == row["median_cpu_s"] / row["median_gpu_s"]
    expected = segment_image(native.astype(float) / 255, make_config())
    different = deepcopy(expected)
    different.refined[0].reasons[0] += 1
    different.refined[0].displacement[3] += .1
    different.regions[0].refined.equivalent_diameter_px += .01
    parity = check_parity(expected, different)
    assert not parity["passed"]
    assert any("rejection" in f for f in parity["failures"])
    assert parity["max_coordinate_difference_px"] > .001
    assert parity["max_diameter_difference_px"] == pytest.approx(.01)


def test_comparison_series_reuses_gpu_on_saved_uint8_files(fake_cuda, tmp_path, monkeypatch):
    from tools import real_sem_compare as compare
    from sem_segment import pipeline
    from sem_segment.image_io import read_native

    config = make_config(refine={"device": "cuda:0"})
    pixels = np.rint(255 * blurred_disk()).astype(np.uint8)
    monkeypatch.setattr(compare, "read_uint8", lambda path: read_native(path)[0])
    frames = []
    for index in range(2):
        path = f"site/raw/frame_{index + 1:03d}.png"
        compare.save_rgb(tmp_path / path, pixels)
        frames.append({"path": path, "index": index + 1, "order": index + 1, "timestamp_s": None,
                       "dy_px": 0., "dx_px": 0., "registration_status": "registered", "clipped": index == 1})
    expected = segment_image(pixels.astype(float) / 255, make_config())
    template = np.array([[expected.regions[0].coarse.centroid_y, expected.regions[0].coarse.centroid_x]])
    used = []
    original = pipeline.segment_image

    def measured(image, cfg, **kwargs):
        np.testing.assert_allclose(image * 255, pixels, rtol=0, atol=1e-12)
        used.append(kwargs["refiner"])
        return original(image, cfg, **kwargs)

    monkeypatch.setattr(pipeline, "segment_image", measured)
    series = {"frames": frames}
    observations, _ = compare.measure_series(tmp_path, "raw", series, template, 5., config)
    assert used[0] is used[1] and used[0].closed
    assert len(observations) == 4
    assert {r["status"] for r in observations} == {"valid"}
    assert sum(r["clipped"] for r in observations) == 2
    assert series["refinement_backend"]["device"] == "cuda:0"
    assert all("segmentation_timings_s" in frame for frame in frames)
    assert fake_cuda.devices == [0] and fake_cuda.freed == 1

    from sem_noise.comparison_report import comparison_metrics, write_tensorboard
    series.update(step=0, noise={})
    record = {"models": {}, "arms": [], "artifacts": [], "prediction_ranges": [],
              "sites": [{"name": "site", "series": {"raw": series}, "repeatability": [],
                         "full_average": frames[0]["path"]}]}
    record["metrics"] = comparison_metrics(record)
    scalars, texts = {}, {}
    writer = SimpleNamespace(add_scalar=lambda name, value, step: scalars.update({name: value}),
                             add_text=lambda name, value, step: texts.update({name: value}), close=lambda: None)
    write_tensorboard(tmp_path, record, writer_factory=lambda **kwargs: writer)
    metric = record["metrics"][0]
    assert metric["refinement_device"] == texts["site/raw/refinement_device"] == "cuda:0"
    assert metric["values"]["time/refine_s_per_image"] == scalars["site/raw/time/refine_s_per_image"]
    assert metric["values"]["time/total_s_per_image"] >= 0


def test_comparison_resolves_one_gpu_and_respects_segmentation_config(tmp_path):
    from tools.real_sem_compare import ComparisonSettings, resolve_segmentation_settings

    yaml_path = tmp_path / "segment.yml"
    yaml_path.write_text("segmentation:\n  backend: classical\nrefine:\n  device: cuda:2\n")
    settings = ComparisonSettings(checkpoints={}, sites={}, output_dir=tmp_path / "out", segmentation_config=yaml_path)
    segment = resolve_segmentation_settings(settings)
    assert segment.refine.device == "cuda:2" and settings.device == "cuda:2"
    assert (segment.input.black_level, segment.input.white_level) == (0., 255.)
    settings.device = "cuda:0"
    with pytest.raises(ValueError, match="Single-GPU comparison"):
        resolve_segmentation_settings(settings)
    settings.metrology_device = "cpu"
    assert resolve_segmentation_settings(settings).refine.device == "cpu"
    settings.metrology_device = "cuda:0"
    assert resolve_segmentation_settings(settings).refine.device == "cuda:0"
    yaml_path.write_text("segmentation:\n  backend: sam3_auto\n  device: cuda:1\n")
    with pytest.raises(ValueError, match="mask model"):
        resolve_segmentation_settings(settings)
    yaml_path.write_text("segmentation:\n  backend: sam3_auto\n")
    assert resolve_segmentation_settings(settings).segmentation.device == "cuda:0"


def test_preflight_detects_kernel_failure_and_benchmark_rejects_no_measurements(fake_cuda, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("NVRTC compilation failed")
    monkeypatch.setattr(signal, "find_peaks", broken)
    with pytest.raises(RuntimeError, match="Cannot initialize.*NVRTC"):
        cuda.CudaRefiner(RefineConfig(device="cuda:0"))
    from tools.benchmark_sem_metrology import check_parity
    blank = segment_image(np.zeros((32, 32)), make_config())
    assert not check_parity(blank, blank)["passed"]
