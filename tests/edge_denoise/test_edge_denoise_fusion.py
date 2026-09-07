"""Drifting bursts, registration, and burst fusion (drift.py, register.py, fusion.py)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from conftest import write_burst

from burst_diffusion.data import BurstCache, BurstSource
from burst_diffusion.repeatability import RealizationProvider
from edge_denoise.config import Config, FusionConfig
from edge_denoise.drift import (
    DriftParams,
    burst_truths,
    generate_drift_dataset,
    load_drift_truth,
    render_frame,
    sample_burst_truth,
)
from edge_denoise.fusion import (
    AlignmentSource,
    BurstFusionArms,
    FusionBatch,
    FusionDenoiser,
    FusionFactory,
    crop_aligned,
    warp_prediction,
)
from edge_denoise.register import (
    RegistrationTable,
    Trajectory,
    align_burst,
    coarse_shift,
    fuse_mean,
    gaussian_smooth,
    refine_shift,
    register_burst,
    smooth_trajectory,
    warp_frame,
    warp_scene,
)


def _smooth_scene(height: int = 96, width: int = 112, seed: int = 0) -> np.ndarray:
    """A band-limited random scene with structure in both directions."""
    rng = np.random.default_rng(seed)
    noise = torch.from_numpy(rng.random((1, 1, height, width)).astype(np.float32))
    smooth = gaussian_smooth(noise, 2.5)[0, 0].numpy().astype(np.float64)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    return 0.2 + 0.6 * smooth


# ---------------------------------------------------------------------------
# drift model


def test_burst_truth_is_anchored_at_frame_zero_and_velocity_is_the_frame_step() -> None:
    rng = np.random.default_rng(1)
    truth = sample_burst_truth(rng, 8, DriftParams(walk_sigma=0.0, velocity_sigma=0.5), list(range(8)))
    np.testing.assert_allclose(truth.position[0], 0.0)
    # Without a random walk the trajectory is a straight line: every frame's
    # velocity equals the constant velocity and positions step by it.
    np.testing.assert_allclose(np.diff(truth.position, axis=0), truth.velocity[:-1], atol=1e-12)
    np.testing.assert_allclose(truth.velocity, np.broadcast_to(truth.velocity[0], (8, 2)), atol=1e-12)
    assert truth.gain.shape == (8,) and truth.offset.shape == (8,)


def test_render_frame_is_the_clipped_poisson_model_of_the_warped_scene() -> None:
    scene = _smooth_scene()
    rng = np.random.default_rng(0)
    frame = render_frame(scene, np.zeros(2), np.zeros(2), 1.0, 0.0, peak=10.0, rng=rng)
    assert frame.dtype == np.uint8 and frame.shape == scene.shape
    levels = np.unique(frame)
    assert set(levels.tolist()) <= set(np.rint(np.arange(11) / 10.0 * 255.0).astype(int).tolist())
    # Many draws average to the scene (unbiased where nothing clips).
    frames = [
        render_frame(scene, np.zeros(2), np.zeros(2), 1.0, 0.0, peak=10.0, rng=rng).astype(float) / 255.0
        for _ in range(200)
    ]
    assert abs(float(np.mean(frames) - scene.mean())) < 0.01


# ---------------------------------------------------------------------------
# registration primitives


def test_warp_scene_and_warp_frame_invert_each_other() -> None:
    scene = _smooth_scene()
    moved = warp_scene(scene, (3.0, -2.0))
    np.testing.assert_allclose(moved[10:-10, 10:-10], scene[7:-13, 12:-8], atol=1e-6)  # integer shift is exact
    back, valid = warp_frame(moved, (3.0, -2.0))
    np.testing.assert_allclose(back[valid][:], scene[valid], atol=1e-6)
    # Sub-pixel: the round trip is a slight blur but stays close.
    moved = warp_scene(scene, (0.4, -0.3), (0.2, 0.1))
    back, valid = warp_frame(moved, (0.4, -0.3), (0.2, 0.1))
    inner = np.zeros_like(valid)
    inner[4:-4, 4:-4] = True
    assert float(np.abs(back - scene)[inner & valid].max()) < 0.01


def test_coarse_and_refined_shift_recover_a_known_translation() -> None:
    scene = _smooth_scene()
    moved = warp_scene(scene, (2.0, -3.0))
    ref = torch.from_numpy(scene)[None, None].float()
    mov = torch.from_numpy(moved)[None, None].float()
    assert coarse_shift(ref, mov, radius=5) == (2, -3)
    # The bounded search ignores a far-away peak.
    assert coarse_shift(ref, mov, radius=1, guess=(2, -3)) == (2, -3)
    moved = warp_scene(scene, (0.37, -0.61))
    mov = torch.from_numpy(moved)[None, None].float()
    estimate, covariance = refine_shift(ref, mov, (0.0, 0.0))
    np.testing.assert_allclose(estimate, [0.37, -0.61], atol=0.01)
    assert covariance.shape == (2, 2) and np.all(np.diag(covariance) >= 0.0)


def test_smooth_trajectory_anchors_frame_zero_and_follows_the_measurements() -> None:
    truth = np.stack([np.array([0.3 * k, -0.2 * k]) for k in range(10)])
    measured = truth + np.random.default_rng(0).normal(0.0, 0.02, truth.shape)
    measured[0] = 0.0
    cov = np.tile(np.eye(2) * 0.02**2, (10, 1, 1))
    # One frame with no information along x: a huge covariance -> the prior fills in.
    measured[5, 1] = 4.0
    cov[5] = np.diag([0.02**2, 1e4])
    position, velocity = smooth_trajectory(measured, cov)
    np.testing.assert_allclose(position[0], 0.0, atol=1e-6)
    assert float(np.abs(position - truth).max()) < 0.08
    np.testing.assert_allclose(velocity.mean(axis=0), [0.3, -0.2], atol=0.05)


def test_register_burst_recovers_a_rendered_drift_from_noisy_frames() -> None:
    scene = _smooth_scene(height=128, width=144)
    rng = np.random.default_rng(3)
    truth = sample_burst_truth(rng, 6, DriftParams(velocity_sigma=0.5, walk_sigma=0.2, gain_sigma=0.0, offset_sigma=0.0), list(range(6)))
    frames = [
        render_frame(scene, truth.position[k], truth.velocity[k], 1.0, 0.0, peak=40.0, rng=rng).astype(float) / 255.0
        for k in range(6)
    ]
    trajectory = register_burst(frames, sigma=1.5, radius=4)
    assert len(trajectory) == 6
    np.testing.assert_allclose(trajectory.position[0], 0.0)
    assert float(np.abs(trajectory.position - truth.position).max()) < 0.15
    aligned, valid = align_burst(frames, trajectory)
    fused = fuse_mean(aligned, valid, frames[0])
    inner = np.s_[12:-12, 12:-12]
    assert float(np.mean((fused - scene)[inner] ** 2)) < float(np.mean((np.mean(frames, axis=0) - scene)[inner] ** 2))


def test_trajectory_round_trips_through_json() -> None:
    trajectory = Trajectory(
        position=np.arange(6.0).reshape(3, 2),
        velocity=np.ones((3, 2)),
        raw_position=np.zeros((3, 2)),
        covariance=np.tile(np.eye(2), (3, 1, 1)),
    )
    table = RegistrationTable(method="raw", sigma=2.0, radius=6, frames_per_burst=3, bursts={4: [trajectory]})
    path = Path(__file__).parent / "_registration_roundtrip.json"
    try:
        table.save(path)
        loaded = RegistrationTable.load(path)
    finally:
        path.unlink(missing_ok=True)
    assert loaded.method == "raw" and loaded.frames_per_burst == 3
    np.testing.assert_allclose(loaded.bursts[4][0].position, trajectory.position)
    np.testing.assert_allclose(loaded.bursts[4][0].covariance, trajectory.covariance)
    found, offset = loaded.trajectory(4, 2)
    assert offset == 2 and found is loaded.bursts[4][0]


# ---------------------------------------------------------------------------
# drifting dataset


def _drift_dataset(tmp_path: Path) -> tuple[Path, Path]:
    source = write_burst(tmp_path / "source", num_sources=4, replicas=4, height=40, width=48)
    out = tmp_path / "drift"
    generate_drift_dataset(
        source,
        out,
        frames_per_burst=4,
        holdout_retakes=2,
        params=DriftParams(velocity_sigma=0.3, walk_sigma=0.1),
        peak=10.0,
        seed=0,
        val_fraction=0.34,
        test_fraction=0.0,
        split_seed=7,
    )
    return source, out


def test_generate_drift_dataset_layout_split_and_truth(tmp_path: Path) -> None:
    source, out = _drift_dataset(tmp_path)
    truth = load_drift_truth(out)
    assert truth is not None and truth["frames_per_burst"] == 4
    source_cache = BurstCache(source, val_fraction=0.34, split_seed=7)
    drift_cache = BurstCache(out, val_fraction=0.34, split_seed=7)
    assert [s.source_index for s in drift_cache.val_sources] == [s.source_index for s in source_cache.val_sources]
    assert truth["split"]["val_source_indices"] == [s.source_index for s in source_cache.val_sources]
    for item in drift_cache.train_sources:
        assert len(item.frames) == 4
        assert len(burst_truths(truth, item.source_index)) == 1
    for item in drift_cache.val_sources:
        assert len(item.frames) == 8  # 2 retakes x 4 frames
        bursts = burst_truths(truth, item.source_index)
        assert [b.replicas for b in bursts] == [[0, 1, 2, 3], [4, 5, 6, 7]]
        np.testing.assert_allclose(bursts[1].position[0], 0.0)
    rows = [json.loads(line) for line in (out / "burst" / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 4 * len(drift_cache.train_sources) + 8 * len(drift_cache.val_sources)
    with pytest.raises(FileExistsError):
        generate_drift_dataset(source, out, frames_per_burst=4, holdout_retakes=2, val_fraction=0.34, split_seed=7)


# ---------------------------------------------------------------------------
# fusion batches and the warped loss


def test_crop_aligned_integer_shift_matches_slicing_and_subpixel_interpolates() -> None:
    frame = (np.random.default_rng(0).random((40, 48)) * 255).astype(np.uint8)
    crop = crop_aligned(frame, 5, 7, 16, 2.0, -3.0)
    np.testing.assert_allclose(crop, frame[7:23, 4:20].astype(np.float32) / 255.0, atol=1e-6)
    half = crop_aligned(frame, 5, 7, 16, 0.5, 0.0)
    assert half.shape == (16, 16) and 0.0 <= half.min() and half.max() <= 1.0
    edge = crop_aligned(frame, 0, 0, 16, -1.5, -2.5)  # samples outside: reflect padding, no crash
    assert edge.shape == (16, 16)


def test_warp_prediction_is_identity_at_zero_and_a_roll_at_integer_shifts() -> None:
    prediction = torch.from_numpy(_smooth_scene(32, 32)).float()[None, None]
    assert warp_prediction(prediction, torch.zeros(1, 2)) is prediction
    shifted = warp_prediction(prediction, torch.tensor([[1.0, -2.0]]))
    # warped(r, c) = prediction(r - 1, c + 2)
    np.testing.assert_allclose(shifted[0, 0, 4:-4, 4:-4].numpy(), prediction[0, 0, 3:-5, 6:-2].numpy(), atol=1e-5)
    prediction.requires_grad_(True)
    warp_prediction(prediction, torch.tensor([[0.3, 0.2]])).sum().backward()
    assert prediction.grad is not None and float(prediction.grad.abs().sum()) > 0.0


def _fusion_config(dataset_dir: Path, run_dir: Path, *, align: str, registration: Path | None, max_steps: int = 2) -> Config:
    return Config.model_validate(
        {
            "data": {"dataset_dir": str(dataset_dir), "image_size": 16, "channels": 1, "val_fraction": 0.34, "split_seed": 7},
            "objective": {
                "representation": "image",
                "target": "noisy",
                "lambda_image": 1.0,
                "lambda_gradient": 4.0,
                "fusion": {
                    "frames_per_burst": 4,
                    "levels": [1, 2, 3],
                    "align": align,
                    "registration": None if registration is None else str(registration),
                    "warp_margin": 1,
                },
            },
            "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": []},
            "training": {
                "run_dir": str(run_dir),
                "batch_size": 2,
                "max_steps": max_steps,
                "log_every": 1,
                "val_every": max(1, max_steps),
                "val_images": 1,
                "checkpoint_every": max(1, max_steps),
                "device": "cpu",
                "seed": 3,
            },
        }
    )


def _register_dataset(dataset_dir: Path, out: Path) -> RegistrationTable:
    cache = BurstCache(dataset_dir, val_fraction=0.0)
    bursts = {}
    for source in cache.all_sources:
        trajectories = []
        for burst in range(len(source.frames) // 4):
            frames = [f.astype(float) / 255.0 for f in source.frames[burst * 4 : burst * 4 + 4]]
            trajectories.append(register_burst(frames, sigma=1.5, radius=3))
        bursts[source.source_index] = trajectories
    table = RegistrationTable(method="raw", sigma=1.5, radius=3, frames_per_burst=4, bursts=bursts)
    table.save(out)
    return table


def test_fusion_factory_batches_respect_levels_subsets_and_alignment(tmp_path: Path) -> None:
    _, dataset = _drift_dataset(tmp_path)
    table = _register_dataset(dataset, tmp_path / "registration.json")
    cache = BurstCache(dataset, val_fraction=0.34, split_seed=7)
    fusion = FusionConfig(frames_per_burst=4, levels=[1, 2, 3], align="registered", registration=tmp_path / "registration.json")
    factory = FusionFactory(
        cache,
        image_size=16,
        batch_size=6,
        fusion=fusion,
        alignment=AlignmentSource("registered", frames_per_burst=4, table=table),
        seed=0,
    )
    batch, info = factory.sample_batch(return_info=True)
    assert isinstance(batch, FusionBatch)
    assert batch.inputs.shape == (6, 1, 16, 16) and batch.targets.shape == (6, 1, 16, 16)
    assert batch.levels.shape == (6,) and batch.shifts.shape == (6, 2)
    assert float(batch.shifts.abs().max()) <= 0.5 + 1e-6
    for sample, level in zip(info, batch.levels.tolist()):
        assert sample.level == int(level) and sample.level in (1, 2, 3)
        assert len(sample.subset) == sample.level and sample.target not in sample.subset
    assert float(batch.inputs.min()) >= -1.0 and float(batch.inputs.max()) <= 1.0
    val = factory.val_batch(count=2, level=3)
    assert val.clean.shape == (2, 1, 16, 16)
    np.testing.assert_allclose(val.shifts.numpy(), 0.0, atol=1e-6)  # frame 0 is the burst origin

    # align 'none': plain crops, no residual shift.
    plain = FusionFactory(
        cache, image_size=16, batch_size=3, fusion=FusionConfig(frames_per_burst=4, levels=[2], align="none"),
        alignment=AlignmentSource("none", frames_per_burst=4), seed=0,
    )
    batch, info = plain.sample_batch(return_info=True)
    np.testing.assert_allclose(batch.shifts.numpy(), 0.0)
    source = next(s for s in cache.train_sources if s.source_index == info[0].source_index)
    top, left = info[0].crop_yx
    expected = np.mean([source.frames[i][top : top + 16, left : left + 16] for i in info[0].subset], axis=0) / 255.0 * 2.0 - 1.0
    np.testing.assert_allclose(batch.inputs[0, 0].numpy(), expected, atol=1e-6)

    # target 'complement_mean': the registered mean of every frame outside the
    # subset, in frame-0 coordinates, so no residual warp.
    cmean = FusionFactory(
        cache, image_size=16, batch_size=3,
        fusion=FusionConfig(frames_per_burst=4, levels=[1, 2], align="registered", registration=tmp_path / "registration.json", target="complement_mean", level_cap=2),
        alignment=AlignmentSource("registered", frames_per_burst=4, table=table), seed=5,
    )
    batch, info = cmean.sample_batch(return_info=True)
    np.testing.assert_allclose(batch.shifts.numpy(), 0.0)
    sample = info[0]
    source = next(s for s in cache.train_sources if s.source_index == sample.source_index)
    trajectory = table.bursts[sample.source_index][sample.burst]
    top, left = sample.crop_yx
    complement = [i for i in range(4) if i not in sample.subset]
    assert sample.target in complement and len(complement) == 4 - sample.level
    from edge_denoise.fusion import crop_aligned_batch, _window_shift

    center_row = top + 7.5
    shifts = [_window_shift(trajectory, i, center_row, source.clean.shape[0]) for i in complement]
    expected = crop_aligned_batch([source.frames[i] for i in complement], top, left, 16, shifts).mean(axis=0) * 2.0 - 1.0
    np.testing.assert_allclose(batch.targets[0, 0].numpy(), expected, atol=1e-6)
    val = cmean.val_batch(count=1, level=2)  # validation keeps the raw frame-0 target
    first = cache.val_sources[0]
    height, width = first.clean.shape[:2]
    t0, l0 = (height - 16) // 2, (width - 16) // 2
    np.testing.assert_allclose(val.shifts.numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(
        val.targets[0, 0].numpy(), first.frames[0][t0 : t0 + 16, l0 : l0 + 16] / 255.0 * 2.0 - 1.0, atol=1e-6
    )

    # target 'multi_frame': extra raw targets with their own residual shifts,
    # masked when the complement is smaller than the requested count.
    multi = FusionFactory(
        cache, image_size=16, batch_size=4,
        fusion=FusionConfig(frames_per_burst=4, levels=[1, 3], align="registered", registration=tmp_path / "registration.json", target="multi_frame", targets_per_sample=3),
        alignment=AlignmentSource("registered", frames_per_burst=4, table=table), seed=9,
    )
    batch, info = multi.sample_batch(return_info=True)
    assert batch.extra_targets is not None and batch.extra_targets.shape == (4, 2, 1, 16, 16)
    assert batch.extra_shifts is not None and batch.extra_shifts.shape == (4, 2, 2)
    assert batch.extra_mask is not None and batch.extra_mask.shape == (4, 2)
    for sample, mask in zip(info, batch.extra_mask):
        others = 4 - 1 - sample.level  # frames outside the subset besides the main target
        assert int(mask.sum()) == min(others, 2)
    assert float(batch.extra_shifts.abs().max()) <= 0.5 + 1e-6

    # align 'truth' reads the generator's record.
    truth = load_drift_truth(dataset)
    oracle = FusionFactory(
        cache, image_size=16, batch_size=2, fusion=FusionConfig(frames_per_burst=4, levels=[1], align="truth"),
        alignment=AlignmentSource("truth", frames_per_burst=4, truth=truth), seed=1,
    )
    assert oracle.sample_batch().inputs.shape == (2, 1, 16, 16)


def test_fusion_config_rejects_incompatible_objectives(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="registration"):
        FusionConfig(frames_per_burst=4, levels=[1], align="registered")
    with pytest.raises(ValueError, match="levels"):
        FusionConfig(frames_per_burst=4, levels=[4], align="none")
    raw = _fusion_config(tmp_path, tmp_path / "run", align="none", registration=None).model_dump(mode="json")
    raw["objective"]["target"] = "clean"
    with pytest.raises(ValueError, match="target must be 'noisy'"):
        Config.model_validate(raw)


def test_multi_frame_target_trains(tmp_path: Path) -> None:
    from edge_denoise.train import Trainer

    _, dataset = _drift_dataset(tmp_path)
    registration = tmp_path / "registration.json"
    _register_dataset(dataset, registration)
    config = _fusion_config(dataset, tmp_path / "run", align="registered", registration=registration, max_steps=2)
    raw = config.model_dump(mode="json")
    raw["objective"]["fusion"]["target"] = "multi_frame"
    raw["objective"]["fusion"]["targets_per_sample"] = 2
    trainer = Trainer(Config.model_validate(raw))
    trainer.run()
    assert trainer.step == 2


def test_fusion_training_runs_checkpoints_and_fuses_a_retake(tmp_path: Path) -> None:
    from edge_denoise.train import LATEST_CHECKPOINT_NAME, Trainer, load_checkpoint

    _, dataset = _drift_dataset(tmp_path)
    registration = tmp_path / "registration.json"
    _register_dataset(dataset, registration)
    config = _fusion_config(dataset, tmp_path / "run", align="registered", registration=registration, max_steps=2)
    trainer = Trainer(config)
    assert trainer.is_fusion
    checkpoint = trainer.run()
    assert trainer.step == 2 and checkpoint == tmp_path / "run" / LATEST_CHECKPOINT_NAME
    payload = load_checkpoint(checkpoint)
    assert Config.model_validate(payload["config"]) == config

    denoiser = FusionDenoiser.from_checkpoint(checkpoint, device="cpu")
    cache = BurstCache(dataset, val_fraction=0.34, split_seed=7)
    source = cache.val_sources[0]
    arms = BurstFusionArms(denoiser, frame_counts=[1, 3], frames_per_retake=4, align="registered")
    outputs = arms.outputs(source, 1)
    assert set(outputs) == {"fuse1", "fuse3", "regavg1", "regavg3"}
    assert outputs["fuse3"].shape == source.clean.shape and outputs["regavg3"].shape == source.clean.shape
    # A one-frame arm cannot infer scan motion from the other retake frames.
    trajectory = arms._trajectories[(source.source_index, 1, 1)]
    raw = source.frames[4].astype(float) / 255.0
    np.testing.assert_array_equal(trajectory.velocity, np.zeros((1, 2)))
    np.testing.assert_array_equal(outputs["regavg1"], raw)
    assert denoiser.level_for(3) == 3.0
    burst = arms.burst_arms()
    np.testing.assert_allclose(burst["fuse3"](source, 1), outputs["fuse3"])

    # The provider joins the harness through generate_source with the retake stride.
    from burst_diffusion.config import Config as BurstConfig
    from burst_diffusion.repeatability import repeatability

    bridge = BurstConfig.model_validate(
        {
            "data": {"dataset_dir": str(dataset), "image_size": 16, "channels": 1, "val_fraction": 0.34, "split_seed": 7},
            "schedule": {"num_steps": 1},
            "model": {"ch": 8, "ch_mult": [1], "num_res_blocks": 1, "attn_resolutions": []},
            "training": {"run_dir": str(tmp_path / "rep"), "device": "cpu"},
            "sampling": {},
        }
    )
    results = repeatability(
        bridge, {}, out_dir=tmp_path / "rep", num_seeds=2, avg_counts=(4,), extra_providers={"fuse": arms.provider()}, seed_stride=4,
    )
    assert results["seed_stride"] == 4
    assert results["methods"]["fuse3@fuse"]["realizations_per_source"] == [2] * len(cache.val_sources)
    assert results["methods"]["avg_of_4"]["realizations_per_source"] == [2] * len(cache.val_sources)
    assert results["methods"]["single_frame"]["realizations_per_source"] == [2] * len(cache.val_sources)


@pytest.mark.parametrize("align", ["registered", "none", "truth"])
def test_one_frame_fusion_arm_never_registers_or_resamples(align: str) -> None:
    frames = [np.arange(16, dtype=np.uint8).reshape(4, 4), np.full((4, 4), 200, dtype=np.uint8)]
    source = BurstSource(source_index=7, clean=np.zeros((4, 4), dtype=np.uint8), frames=frames)
    denoiser = Mock(spec=FusionDenoiser)
    denoiser.register.side_effect = AssertionError("one frame must not be registered")
    denoiser.registered_mean.side_effect = AssertionError("the anchor must not be resampled")
    denoiser.denoise_mean.side_effect = lambda image, count, **kwargs: image.copy()
    # An empty truth mapping must never be read for the one-frame arm.
    arms = BurstFusionArms(
        denoiser, frame_counts=[1], frames_per_retake=2, align=align, truth={} if align == "truth" else None
    )
    before = arms.outputs(source, 0)
    source.frames[1][:] = 0
    after = arms.outputs(source, 0)
    for name in ("fuse1", "regavg1"):
        np.testing.assert_array_equal(before[name], source.frames[0].astype(np.float64) / 255.0)
        np.testing.assert_array_equal(after[name], before[name])
    trajectory = arms._trajectories[(7, 0, 1)]
    np.testing.assert_array_equal(trajectory.position, np.zeros((1, 2)))
    np.testing.assert_array_equal(trajectory.velocity, np.zeros((1, 2)))


def test_fusion_registration_and_outputs_use_only_each_arms_frame_prefix() -> None:
    def evaluate(last_value: int) -> tuple[BurstFusionArms, dict[str, np.ndarray], Mock]:
        frames = [np.full((4, 4), value, dtype=np.uint8) for value in (10, 20, 30, last_value)]
        source = BurstSource(source_index=7, clean=np.zeros((4, 4), dtype=np.uint8), frames=frames)
        denoiser = Mock(spec=FusionDenoiser)

        def register(prefix, **kwargs):
            trajectory = Trajectory.identity(len(prefix))
            trajectory.velocity[:, 0] = prefix[-1].mean()
            return trajectory

        def registered_mean(prefix, count, trajectory):
            assert len(prefix) == count == len(trajectory)
            # Make any motion leaked from the later frames observable in the output.
            return np.mean(prefix, axis=0) + trajectory.velocity[0, 0]

        denoiser.register.side_effect = register
        denoiser.registered_mean.side_effect = registered_mean
        denoiser.denoise_mean.side_effect = lambda image, count, **kwargs: image.copy()
        arms = BurstFusionArms(denoiser, frame_counts=[1, 2, 4], frames_per_retake=4)
        outputs = arms.outputs(source, 0)
        repeated = arms.outputs(source, 0)
        for name in outputs:
            np.testing.assert_array_equal(repeated[name], outputs[name])
        return arms, outputs, denoiser

    arms, before, denoiser = evaluate(40)
    changed_arms, after, changed_denoiser = evaluate(240)
    for name in ("fuse1", "regavg1", "fuse2", "regavg2"):
        np.testing.assert_array_equal(after[name], before[name])
    assert not np.array_equal(after["fuse4"], before["fuse4"])
    for count in (1, 2):
        np.testing.assert_array_equal(
            arms._trajectories[(7, 0, count)].velocity,
            changed_arms._trajectories[(7, 0, count)].velocity,
        )
    for model in (denoiser, changed_denoiser):
        assert [len(call.args[0]) for call in model.register.call_args_list] == [2, 4]
    assert set(arms._trajectories) == {(7, 0, 1), (7, 0, 2), (7, 0, 4)}


def test_truth_fusion_arm_limits_privileged_motion_to_selected_frames() -> None:
    truth = sample_burst_truth(np.random.default_rng(5), 4, DriftParams(), list(range(4)))
    source = BurstSource(
        source_index=7, clean=np.zeros((4, 4), dtype=np.uint8),
        frames=[np.full((4, 4), value, dtype=np.uint8) for value in (10, 20, 30, 40)],
    )
    denoiser = Mock(spec=FusionDenoiser)
    denoiser.denoise_mean.side_effect = lambda image, count, **kwargs: image.copy()

    def registered_mean(prefix, count, trajectory):
        assert len(prefix) == count == len(trajectory) == 2
        np.testing.assert_array_equal(trajectory.position, truth.position[:2])
        np.testing.assert_array_equal(trajectory.velocity, truth.velocity[:2])
        return np.mean(prefix, axis=0)

    denoiser.registered_mean.side_effect = registered_mean
    arms = BurstFusionArms(
        denoiser, frame_counts=[1, 2], frames_per_retake=4, align="truth",
        truth={"sources": {"7": {"bursts": [truth.to_json()]}}},
    )
    outputs = arms.outputs(source, 0)
    np.testing.assert_array_equal(outputs["regavg1"], source.frames[0].astype(np.float64) / 255.0)
    np.testing.assert_array_equal(arms._trajectories[(7, 0, 1)].velocity, np.zeros((1, 2)))
    denoiser.register.assert_not_called()
    denoiser.registered_mean.assert_called_once()


def test_direct_one_frame_fusion_ignores_external_trajectory() -> None:
    denoiser = Mock(spec=FusionDenoiser)
    denoiser.register.side_effect = AssertionError("one frame must not be registered")
    denoiser.registered_mean.side_effect = AssertionError("the anchor must not be resampled")
    denoiser.denoise_mean.side_effect = lambda image, count, **kwargs: image.copy()
    anchor = np.arange(16, dtype=np.float64).reshape(4, 4) / 255.0
    trajectory = Trajectory.identity(2)
    trajectory.velocity[:] = 2.5
    for supplied in (None, trajectory):
        result = FusionDenoiser.fuse(denoiser, [anchor, np.ones_like(anchor)], 1, trajectory=supplied)
        np.testing.assert_array_equal(result, anchor)


def test_repeatability_seed_stride_picks_the_first_frame_of_each_retake(tmp_path: Path) -> None:
    from burst_diffusion.config import Config as BurstConfig
    from burst_diffusion.repeatability import repeatability

    dataset = write_burst(tmp_path / "data", num_sources=3, replicas=6, height=20, width=24, bar=True)
    seen: list[list[np.ndarray]] = []

    def generate(seeds01: list[np.ndarray]) -> dict[str, list[np.ndarray]]:
        seen.append(seeds01)
        return {"echo": list(seeds01)}

    bridge = BurstConfig.model_validate(
        {
            "data": {"dataset_dir": str(dataset), "image_size": 16, "channels": 1, "val_fraction": 0.34, "split_seed": 7},
            "schedule": {"num_steps": 1},
            "model": {"ch": 8, "ch_mult": [1], "num_res_blocks": 1, "attn_resolutions": []},
            "training": {"run_dir": str(tmp_path / "rep"), "device": "cpu"},
            "sampling": {},
        }
    )
    provider = RealizationProvider(method_names=("echo",), generate=generate)
    with pytest.warns(UserWarning, match="clamping"):
        results = repeatability(
            bridge, {}, out_dir=tmp_path / "rep", num_seeds=5, avg_counts=(3,), extra_providers={"p": provider}, seed_stride=3,
        )
    assert results["num_seeds"] == 2  # 6 frames / stride 3
    cache = BurstCache(dataset, val_fraction=0.34, split_seed=7)
    source = cache.val_sources[0]
    window = np.s_[2:18, 4:20]
    np.testing.assert_allclose(seen[0][1][:, :, 0], source.frames[3][window] / 255.0)


def test_finefeat_retake_rows_average_inside_each_retake() -> None:
    from edge_denoise.finefeat import _retake_rows

    frames = [np.full((4, 4), float(k)) for k in range(8)]
    seeds, rows, count = _retake_rows(frames, num_seeds=3, frames_per_retake=4)
    assert count == 2 and [float(s.mean()) for s in seeds] == [0.0, 4.0]
    np.testing.assert_allclose([float(r.mean()) for r in rows["avg_of_4"]], [1.5, 5.5])
    assert "avg_of_16" not in rows
    seeds, rows, count = _retake_rows(frames, num_seeds=3, frames_per_retake=1)
    assert count == 3 and len(rows["avg_of_16"]) == 1
