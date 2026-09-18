from __future__ import annotations

import csv
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
from edge_denoise.real_data import RealPairFactory, estimate_translations
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
    assert saved["data"]["real_matching"] == {"registration": "none", "brightness": "percentile",
                                               "registration_failure": None}
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


def mixed_geometry(frames, *, registration="translation") -> dict:
    matrices = np.repeat(np.eye(3)[None], len(frames), axis=0)
    matrices[1, :2, 2], matrices[3, :2, 2] = [3, -2], [-1, 2]
    if registration == "affine":
        matrices[1, :2, :2] = [[1.01, .02], [-.01, .99]]
    diagnostics = [{"status": "registered"} for _ in frames]
    diagnostics[0] = {"status": "reference"}
    diagnostics[2] = {"status": "skipped_failed_registration", "reason": "synthetic failure"}
    diagnostics[4] = {"status": "skipped_low_contrast", "contrast": 0.001}
    return {"matrices": matrices.tolist(), "diagnostics": diagnostics,
            "percentiles_dn": [np.percentile(frame, np.arange(10, 91, 5)).tolist() for frame in frames]}


@pytest.mark.parametrize("registration", ["translation", "affine"])
@pytest.mark.parametrize("a,b,second", [(1, 2, 4), (2, 1, 3), (1, 3, 2)])
@pytest.mark.parametrize("brightness", ["none", "percentile"])
def test_unavailable_geometry_disables_only_affected_pairs(prepared, tmp_path, monkeypatch,
                                                         registration, a, b, second, brightness) -> None:
    monkeypatch.setattr(MatchedRealPairFactory, "_measure", lambda self, frames, _: mixed_geometry(frames, registration=registration))
    cfg = matching_config(prepared, tmp_path / "run", registration, brightness, consistency=1)
    factory = MatchedRealPairFactory(BurstCache(prepared), cfg, seed=0)
    source = factory.cache.train_sources[0]
    sample = factory._pair(source, (8, 8), a, b, second)
    y, x = sample[-1].crop_yx
    native = lambda index: normalize_native(source.frames[index, y:y + 16, x:x + 16], 0, 4095)
    np.testing.assert_array_equal(sample[0][0], native(a) * 2 - 1)
    if a in (2, 4) or b in (2, 4):
        fit = measure_quantile_brightness(source.frames[a], source.frames[b]) if brightness == "percentile" else {"gain": 1, "offset_dn": 0}
        expected = (fit["gain"] * native(b) + fit["offset_dn"] / 4095) * 2 - 1
        np.testing.assert_allclose(sample[1][0], expected, atol=3e-7)
    else:
        assert not np.array_equal(factory._pair_matrix(source.source_index, a, b), np.eye(2, 3))
    if a in (2, 4) or second in (2, 4):
        np.testing.assert_array_equal(sample[2][0], native(second) * 2 - 1)
        np.testing.assert_array_equal(sample[4], np.eye(2, 3))
        if brightness == "none":
            batch, _ = factory._batch([sample])
            assert torch.equal(match_prediction(batch.second, batch.second_matrices, batch.second_brightness), batch.second)


@pytest.mark.parametrize("a", [1, 2])
def test_mean_target_keeps_failed_frames_and_uses_native_coordinates(prepared, tmp_path, monkeypatch, a) -> None:
    monkeypatch.setattr(MatchedRealPairFactory, "_measure", lambda self, frames, _: mixed_geometry(frames))
    cfg = matching_config(prepared, tmp_path / "run", "translation", "none", target="noisy_mean", consistency=1)
    factory = MatchedRealPairFactory(BurstCache(prepared), cfg, seed=0)
    source = factory.cache.train_sources[0]
    sample = factory._pair(source, (8, 8), a, 0, 4)
    y, x = sample[-1].crop_yx
    # Independent integer-coordinate oracle, including ALL other acquisitions.
    shifts = np.array([[0, 0], [-2, 3], [0, 0], [2, -1], [0, 0], [0, 0], [0, 0], [0, 0]])
    images = []
    for j in range(len(source.frames)):
        if j == a:
            continue
        dy, dx = (0, 0) if a in (2, 4) or j in (2, 4) else shifts[j] - shifts[a]
        images.append(normalize_native(source.frames[j, y + dy:y + dy + 16, x + dx:x + dx + 16], 0, 4095))
    np.testing.assert_allclose(sample[1][0], np.mean(images, axis=0) * 2 - 1, atol=3e-7)


def test_failed_frames_remain_sampled_and_reported_on_resume(prepared, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(MatchedRealPairFactory, "_measure", lambda self, frames, _: mixed_geometry(frames))
    cfg = matching_config(prepared, tmp_path / "run", "translation", "percentile", consistency=1)
    trainer = Trainer(cfg)
    _, info = trainer.factory.sample_batch(count=200, return_info=True)
    assert {2, 4}.issubset({row.input_replica for row in info})
    assert {2, 4}.issubset({row.target_replica for row in info})
    assert {2, 4}.issubset({row.second_replica for row in info})
    checkpoint = trainer.run()
    with (cfg.training.run_dir / "registration_frames.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 * 8
    failures = [row for row in rows if row["status"].startswith("skipped")]
    assert len(failures) == 6 and all(row["retained_in_split"] == "True" for row in failures)
    assert all(row["filename"].endswith(f"frame_{row['frame_index']}.tif") for row in failures)
    assert {row["reason"] for row in failures} == {"synthetic failure", "contrast below the configured minimum; geometry is unmeasured"}
    assert (cfg.training.run_dir / "registration_report.html").is_file()
    resumed = Trainer(cfg, resume_from=checkpoint)
    for key in trainer.factory.geometry_available:
        np.testing.assert_array_equal(resumed.factory.geometry_available[key], trainer.factory.geometry_available[key])
    expected, actual = trainer.factory.sample_batch(), resumed.factory.sample_batch()
    assert torch.equal(expected.inputs, actual.inputs) and torch.equal(expected.targets, actual.targets)
    assert torch.equal(expected.second_matrices, actual.second_matrices)


@pytest.mark.parametrize("reason", ["shift", "covariance"])
def test_train_cli_registration_failure_override_retains_rejected_frames(prepared, tmp_path, monkeypatch, reason) -> None:
    from edge_denoise import real_data

    monkeypatch.setattr(real_data, "coarse_shift", lambda *args, **kwargs: (0, 0))
    monkeypatch.setattr(real_data, "refine_shift", lambda *args, **kwargs:
                        (np.array([1000., 0.]) if reason == "shift" else np.zeros(2),
                         np.full((2, 2), np.nan) if reason == "covariance" else np.eye(2)))
    cfg = real_config(prepared, tmp_path / "run")
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    args = ["train", "--config", str(path), "--real-registration", "translation", "--real-brightness", "percentile"]
    strict = CliRunner().invoke(app, args)
    assert strict.exit_code != 0 and "--registration-failure skip" in str(strict.exception)
    skipped = CliRunner().invoke(app, args + ["--registration-failure", "skip"])
    assert skipped.exit_code == 0, (skipped.output, skipped.exception)
    payload = json.loads((cfg.training.run_dir / "real_matching.json").read_text(encoding="utf-8"))
    assert payload["settings"]["registration_failure"] == "skip"
    for site in payload["sites"].values():
        assert site["failure_policy"] == "skip"
        assert any(row["status"] == "skipped_failed_registration" for row in site["diagnostics"])


def test_affine_failure_policy_and_old_checkpoint_settings(prepared, tmp_path, monkeypatch) -> None:
    from sem_noise import pair_matching

    def failed(*args, **kwargs):
        raise ValueError("synthetic ECC failure")
    monkeypatch.setattr(pair_matching, "estimate_geometry", failed)
    cfg = matching_config(prepared, tmp_path / "run", "affine")
    cache = BurstCache(prepared)
    with pytest.raises(ValueError, match="registration-failure skip"):
        MatchedRealPairFactory(cache, cfg, seed=0)
    cfg.data.real_matching.registration_failure = "skip"
    factory = MatchedRealPairFactory(cache, cfg, seed=0)
    assert all(np.count_nonzero(valid) == 1 for valid in factory.geometry_available.values())
    assert torch.isfinite(factory.sample_batch().targets).all()
    legacy = MatchedRealPairFactory(cache, matching_config(prepared, tmp_path / "legacy", "none"), seed=0)
    state = legacy.state_dict()
    del state["matching_settings"]["registration_failure"]  # b091b23 checkpoint compatibility.
    legacy.load_state_dict(state)


def test_translation_signs_on_analytic_images_with_nonanchor_input() -> None:
    # Generate continuous shifted specimens directly, without either warp implementation.
    yy, xx = np.mgrid[:96, :96]
    truth = np.array([[0, 0], [1.25, -.5], [-1.5, 2.0]])
    def scene(y, x):
        return .12 + .45 * np.exp(-((y - 28) ** 2 + (x - 35) ** 2) / 200) + .3 * np.exp(-((y - 65) ** 2 + (x - 60) ** 2) / 98)
    frames = np.stack([scene(yy - dy, xx - dx) for dy, dx in truth])
    kwargs = dict(black=0, white=1, sigma=1.5, radius=6, max_shift=10, device="cpu")
    measured, _ = estimate_translations(frames, **kwargs)
    np.testing.assert_allclose(measured, truth, atol=.06)
    matrices = np.repeat(np.eye(3)[None], 3, axis=0)
    matrices[:, :2, 2] = measured[:, ::-1]
    for a, b in ((1, 2), (2, 1)):
        direct, _ = estimate_translations(frames[[a, b]], **kwargs)
        pair = compose_pair(matrices, a, b)
        np.testing.assert_allclose(pair[:, 2], (truth[b] - truth[a])[::-1], atol=.09)
        np.testing.assert_allclose(direct[1], truth[b] - truth[a], atol=.06)
        # Independent fits have interpolation bias: opposite 0.05px errors can
        # disagree by 0.10px even on a noiseless analytic specimen.
        np.testing.assert_allclose(pair[:, 2], direct[1, ::-1], atol=.12)
        sampled, valid = sample_target(frames[b], crop_matrix(pair, np.array([16, 16])), 48, 0, 1)
        expected = frames[a, 16:64, 16:64]
        assert np.sqrt(np.mean((sampled[valid] - expected[valid]) ** 2)) < .002


def test_general_affine_composition_uses_correct_order_and_crop_coordinates() -> None:
    matrices = np.array([[[1.02, -.03, 2], [.01, .99, -3], [0, 0, 1]],
                         [[.98, .02, -4], [-.02, 1.01, 1], [0, 0, 1]]])
    points = np.column_stack((np.random.default_rng(84).uniform(0, 512, (100, 2)), np.ones(100)))
    for a, b in ((0, 1), (1, 0)):
        points_a = points @ matrices[a].T
        points_b = points @ matrices[b].T
        pair = compose_pair(matrices, a, b)
        np.testing.assert_allclose(points_a @ pair.T, points_b[:, :2], atol=1e-12)
        origin_yx = np.array([23, 47])
        local_a = points_a.copy()
        local_a[:, :2] -= origin_yx[::-1]
        np.testing.assert_allclose(local_a @ crop_matrix(pair, origin_yx).T, points_b[:, :2], atol=1e-12)
