from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from typer.testing import CliRunner

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from burst_diffusion.data import BurstCache, BurstSource
from burst_diffusion.real_data import normalize_native
from edge_denoise.cli import app
from edge_denoise.config import RealMatchingConfig
from edge_denoise.real_data import RealPairFactory
from edge_denoise.real_matching import (MatchedRealPairFactory, compose_pair, crop_matrix,
                                        match_prediction, percentile_mapping, sample_target)
from edge_denoise.train import Trainer, _masked_mean, load_checkpoint
from sem_noise.pair_matching import measure_quantile_brightness
from test_edge_denoise_real import prepared, real_config


def matching_config(dataset, run, registration="none", brightness="percentile", **kwargs):
    cfg = real_config(dataset, run, **kwargs)
    cfg.data.real_matching = RealMatchingConfig(registration=registration, brightness=brightness)
    return cfg


def test_cached_percentiles_are_the_unchanged_full_image_estimator() -> None:
    rng = np.random.default_rng(41)
    a = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    b = rng.integers(20, 230, (64, 64), dtype=np.uint8)
    a[0], b[-1] = 255, 0
    result = measure_quantile_brightness(a, b)
    gain, offset = percentile_mapping(np.percentile(a, np.arange(10, 91, 5)),
                                       np.percentile(b, np.arange(10, 91, 5)))
    assert gain == result["gain"] and offset == result["offset_dn"]
    with pytest.raises(ValueError, match="equal"):
        percentile_mapping(np.arange(17), np.ones(17))


def test_inline_percentile_corrects_only_B_and_keeps_full_image_support(prepared, tmp_path) -> None:
    cache = BurstCache(prepared)
    before = {s.source_index: s.frames.copy() for s in cache.all_sources}
    cfg = matching_config(prepared, tmp_path / "run")
    factory = MatchedRealPairFactory(cache, cfg, seed=5)
    batch, infos = factory.sample_batch(count=6, return_info=True)
    for i, info in enumerate(infos):
        source = next(s for s in cache.train_sources if s.source_index == info.source_index)
        a, b = source.frames[info.input_replica], source.frames[info.target_replica]
        y, x = info.crop_yx
        measured = measure_quantile_brightness(a, b)
        expected = (measured["gain"] * b[y:y + 16, x:x + 16].astype(float) + measured["offset_dn"]) / 4095 * 2 - 1
        np.testing.assert_allclose(batch.targets[i, 0], expected, atol=2e-7)
        np.testing.assert_array_equal(batch.inputs[i, 0], normalize_native(a[y:y + 16, x:x + 16], 0, 4095) * 2 - 1)
        assert batch.target_valid[i].all()
    for source in cache.all_sources:
        np.testing.assert_array_equal(source.frames, before[source.source_index])
    # Coefficients are small cached records, not fits repeated for each crop.
    assert all(np.asarray(r["percentiles_dn"]).shape == (8, 17) for r in factory.measurements.values())
    state = factory.state_dict()
    expected = factory.sample_batch()
    factory.load_state_dict(state)
    actual = factory.sample_batch()
    assert torch.equal(expected.inputs, actual.inputs) and torch.equal(expected.targets, actual.targets)


def test_normalization_offset_and_corrected_values_are_not_clipped(prepared, tmp_path) -> None:
    cache = BurstCache(prepared)
    factory = MatchedRealPairFactory(cache, matching_config(prepared, tmp_path / "run"), seed=1)
    factory.black, factory.white = 100, 4095
    index = cache.train_sources[0].source_index
    q = np.arange(17, dtype=float) * 10 + 300
    factory.quantiles[index][0] = 10 * q + 6000
    factory.quantiles[index][1] = q
    gain, offset_unit = factory._brightness(index, 0, 1)
    assert gain == pytest.approx(10)
    assert offset_unit == pytest.approx((6000 + 9 * 100) / 3995)
    sample = factory._pair(cache.train_sources[0], (8, 8), 0, 1, 2)
    assert sample[1].min() > 1  # Targets can exceed the model's nominal [-1,1] range.


def test_translation_retains_existing_estimator_sampling_and_bounds(prepared, tmp_path, monkeypatch) -> None:
    from edge_denoise import real_matching

    calls = []
    shifts = np.array([[0, 0], [.25, -.5], [-.25, .5], [1, -1], [0, 0], [0, 0], [0, 0], [0, 0]])
    def estimate(frames, **kwargs):
        calls.append(kwargs)
        return shifts.copy(), np.zeros_like(shifts)
    monkeypatch.setattr(real_matching, "estimate_translations", estimate)
    cache = BurstCache(prepared)
    cfg = matching_config(prepared, tmp_path / "run", registration="translation", brightness="none")
    factory = MatchedRealPairFactory(cache, cfg, seed=10)
    assert len(calls) == len(cache.train_sources + cache.val_sources)
    settings = cache.real_metadata["registration"]
    assert calls[0]["sigma"] == settings["sigma"] and calls[0]["radius"] == settings["radius"]
    original = RealPairFactory(cache, real_config(prepared, tmp_path / "old"), seed=10)
    original.sites = {key: {**site, "shifts": shifts.tolist(), "bounds": factory.sites[key]["bounds"]}
                      for key, site in original.sites.items()}
    new_batch, new_infos = factory.sample_batch(count=6, return_info=True)
    old_batch, old_infos = original.sample_batch(count=6, return_info=True)
    assert new_infos == old_infos
    assert torch.equal(new_batch.inputs, old_batch.inputs)
    assert torch.equal(new_batch.targets, old_batch.targets)
    assert len(calls) == len(cache.train_sources + cache.val_sources)


def test_affine_patch_is_a_single_warp_of_original_B_with_valid_footprint() -> None:
    import cv2

    rng = np.random.default_rng(9)
    raw = rng.integers(0, 65536, (70, 80), dtype=np.uint16)
    matrices = np.array([[[1.02, .01, 3], [-.02, .99, 2], [0, 0, 1]],
                         [[.98, -.03, -1], [.01, 1.01, 4], [0, 0, 1]]])
    pair = compose_pair(matrices, 0, 1)
    np.testing.assert_allclose(pair, (matrices[1] @ np.linalg.inv(matrices[0]))[:2])
    local = crop_matrix(pair, np.array([10, 12]))
    actual, valid = sample_target(raw, local, 24, 0, 65535)
    expected = cv2.warpAffine(normalize_native(raw, 0, 65535), local, (24, 24),
                              flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    np.testing.assert_allclose(actual[valid], expected[valid], atol=1e-7)
    edge = np.array([[1.01, .01, -3], [0, 1, -2]])
    _, valid = sample_target(raw, edge, 24, 0, 65535)
    assert not valid[:4].any() and valid[-1, -1]


def test_affine_startup_uses_report_ECC_and_reports_failure(prepared, tmp_path, monkeypatch) -> None:
    from sem_noise import pair_matching

    calls = []
    def estimate(a, b, **kwargs):
        calls.append(kwargs)
        return np.eye(2, 3), 1.0
    monkeypatch.setattr(pair_matching, "estimate_geometry", estimate)
    cache = BurstCache(prepared)
    factory = MatchedRealPairFactory(cache, matching_config(prepared, tmp_path / "run", registration="affine"), seed=1)
    assert [c["motion"] for c in calls] == ["translation", "affine"] * (7 * 3)
    assert all(c["sigma"] == 1 and c["input_invalid"].dtype == bool for c in calls)
    count = len(calls)
    factory.sample_batch(count=3)
    assert len(calls) == count
    def fail(*args, **kwargs):
        raise ValueError("test ECC failure")
    monkeypatch.setattr(pair_matching, "estimate_geometry", fail)
    with pytest.raises(ValueError, match="site .*frame 1: test ECC failure"):
        MatchedRealPairFactory(cache, matching_config(prepared, tmp_path / "bad", registration="affine"), seed=1)


def test_prediction_mapping_is_differentiable_and_masks_exclude_padding() -> None:
    values = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16).requires_grad_()
    matrix = torch.tensor([[[0., -1, 15], [1, 0, 0]]])
    mapped = match_prediction(values, matrix, torch.tensor([[2., -3.]]))
    yy, xx = np.mgrid[:16, :16]
    expected = 2 * (xx * 16 + 15 - yy) - 3
    np.testing.assert_allclose(mapped.detach()[0, 0], expected, atol=0.001)  # float32 cubic grid interpolation
    mapped.sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0
    errors = torch.zeros((2, 1, 16, 16))
    errors[0, 0, 5, 5] = 100
    valid = torch.ones_like(errors, dtype=torch.bool)
    valid[0, 0, 5, 5] = False
    assert _masked_mean(errors, valid, 3) == 0
    valid[1] = False
    with pytest.raises(ValueError, match="no valid"):
        _masked_mean(errors, valid, 3)


def test_leave_one_out_corrects_each_other_frame_to_A_before_averaging(prepared, tmp_path) -> None:
    cache = BurstCache(prepared)
    cfg = matching_config(prepared, tmp_path / "run", target="noisy_mean", consistency=1, gradient=4)
    factory = MatchedRealPairFactory(cache, cfg, seed=0)
    source = cache.train_sources[0]
    a, second = 3, 4
    sample = factory._pair(source, (8, 8), a, 0, second)
    expected = []
    for b in range(len(source.frames)):
        if b == a:
            continue
        fit = measure_quantile_brightness(source.frames[a], source.frames[b])
        expected.append((fit["gain"] * source.frames[b, 8:24, 8:24].astype(float) + fit["offset_dn"]) / 4095 * 2 - 1)
    np.testing.assert_allclose(sample[1][0], np.mean(expected, axis=0), atol=2e-7)
    fit = measure_quantile_brightness(source.frames[a], source.frames[second])
    np.testing.assert_allclose(sample[5], [fit["gain"], 2 * fit["offset_dn"] / 4095 + fit["gain"] - 1])
    assert sample[-1].target_replica is None and sample[-1].input_replica != sample[-1].second_replica


@pytest.mark.parametrize("registration,brightness,target,consistency", [
    ("translation", "percentile", "noisy", 0), ("affine", "percentile", "noisy", 0),
    ("none", "percentile", "noisy", 0), ("affine", "none", "noisy", 0),
    ("affine", "percentile", "noisy_mean", 1),
])
def test_requested_arms_train_validate_and_resume(prepared, tmp_path, monkeypatch,
                                                  registration, brightness, target, consistency) -> None:
    from edge_denoise import real_matching
    from sem_noise import pair_matching

    monkeypatch.setattr(real_matching, "estimate_translations",
                        lambda frames, **kwargs: (np.zeros((len(frames), 2)), np.zeros((len(frames), 2))))
    monkeypatch.setattr(pair_matching, "estimate_geometry", lambda *a, **kw: (np.eye(2, 3), 1.0))
    cfg = matching_config(prepared, tmp_path / "run", registration, brightness,
                          target=target, consistency=consistency, gradient=4 if consistency else 0)
    trainer = Trainer(cfg)
    checkpoint = trainer.run()
    saved = load_checkpoint(checkpoint)
    assert saved["factory"]["matching_settings"] == cfg.data.real_matching.model_dump()
    record = json.loads((cfg.training.run_dir / "real_matching.json").read_text(encoding="utf-8"))
    assert record["dataset_fingerprint"] == trainer.cache.real_fingerprint
    resumed = Trainer(cfg, resume_from=checkpoint)
    expected, actual = trainer.factory.sample_batch(), resumed.factory.sample_batch()
    assert torch.equal(expected.inputs, actual.inputs) and torch.equal(expected.targets, actual.targets)
    changed = cfg.model_copy(deep=True)
    changed.data.real_matching.brightness = "none" if brightness == "percentile" else "percentile"
    with pytest.raises(ValueError, match="resume real pair matching differs"):
        Trainer(changed, resume_from=checkpoint)


def test_cli_overrides_support_the_four_terminal_recipe(prepared, tmp_path) -> None:
    cfg = real_config(prepared, tmp_path / "unused")
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    result = CliRunner().invoke(app, ["train", "--config", str(path), "--dataset-dir", str(prepared),
                                       "--run-dir", str(tmp_path / "run"), "--image-size", "16",
                                       "--batch-size", "1", "--accumulation-steps", "4", "--max-steps", "1",
                                       "--lr", "0.0002", "--real-registration", "none",
                                       "--real-brightness", "percentile", "--device", "cpu"])
    assert result.exit_code == 0, (result.output, result.exception)
    saved = load_checkpoint(tmp_path / "run" / "ckpt_latest.pt")["config"]
    assert saved["data"]["real_matching"] == {"registration": "none", "brightness": "percentile"}
    assert saved["training"]["accumulation_steps"] == 4 and saved["training"]["lr"] == 0.0002


def test_inline_matching_rejects_already_registered_cache(prepared, tmp_path) -> None:
    cache = BurstCache(prepared)
    cache.real_metadata["registration"]["mode"] = "translation"
    with pytest.raises(ValueError, match="align none"):
        MatchedRealPairFactory(cache, matching_config(prepared, tmp_path / "run"), seed=0)


def test_unknown_matching_mode_fails_before_training() -> None:
    with pytest.raises(ValueError):
        RealMatchingConfig(registration="automatic")


@pytest.mark.parametrize("target", ["noisy", "noisy_mean"])
def test_rotated_bright_acquisitions_match_native_A_in_target_and_consistency(tmp_path, monkeypatch, target) -> None:
    # Exact quarter-turns preserve the full-image distribution. Camera gains
    # and offsets therefore have a known inverse, independent of the code.
    yy, xx = np.mgrid[:64, :64]
    base = (1000 + 2 * xx + 3 * yy).astype(np.uint16)
    frames = np.stack([base, 2 * np.rot90(base, -1) + 30, 3 * np.rot90(base, 2) + 50])
    matrices = np.array([np.eye(3), [[0, -1, 63], [1, 0, 0], [0, 0, 1]],
                         [[-1, 0, 63], [0, -1, 63], [0, 0, 1]]], dtype=float)
    source = BurstSource(source_index=0, clean=None, frames=frames)
    cache = SimpleNamespace(train_sources=[source], val_sources=[], real_metadata={
        "registration": {"mode": "none"}, "normalization": {"black": 0, "white": 65535},
        "sites": [{"source_index": 0, "name": "rotated", "bounds": [4, 4, 60, 60]}]}, burst_dir=tmp_path)
    cfg = matching_config(tmp_path, tmp_path / "run", registration="affine", target=target, consistency=1)
    # Parent opens existing mean arrays for mean targets; use simple temporary
    # placeholders to establish that the inline target never reads their values.
    if target == "noisy_mean":
        np.save(tmp_path / "aligned.npy", np.zeros_like(frames, dtype=np.float32))
        np.save(tmp_path / "sum.npy", np.zeros((64, 64), dtype=np.float32))
        cache.real_metadata["sites"][0].update(aligned={"path": "aligned.npy"}, sum={"path": "sum.npy"})
    monkeypatch.setattr(MatchedRealPairFactory, "_measure", lambda self, raw, prepared: {
        "matrices": matrices.tolist(), "percentiles_dn": [np.percentile(frame, np.arange(10, 91, 5)).tolist()
                                                            for frame in raw]})
    factory = MatchedRealPairFactory(cache, cfg, seed=0)
    sample = factory._pair(source, (12, 18), 1, 2, 0)  # A is NOT the registration anchor.
    batch, infos = factory._batch([sample])
    y, x = infos[0].crop_yx
    expected_A = normalize_native(frames[1, y:y + 16, x:x + 16], 0, 65535) * 2 - 1
    np.testing.assert_array_equal(batch.inputs[0, 0], expected_A)
    np.testing.assert_allclose(batch.targets[0, 0], expected_A, atol=2e-7)
    matched = match_prediction(batch.second, batch.second_matrices, batch.second_brightness)
    delta = matched - batch.inputs
    # Allow float32 bicubic interpolation roundoff in the normalized coordinates.
    assert _masked_mean(delta ** 2, batch.second_valid, batch.loss_margin) < 1e-10
