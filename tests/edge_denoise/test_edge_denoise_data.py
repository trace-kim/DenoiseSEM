from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from burst_diffusion.data import BurstCache

from edge_denoise.data import PairFactory

from conftest import write_burst


def _cache(root: Path, **overrides) -> BurstCache:
    kwargs = dict(channels=1, min_replicas=2, min_size=16, val_fraction=0.34, split_seed=7)
    kwargs.update(overrides)
    return BurstCache(root, **kwargs)


def _factory(cache: BurstCache, **overrides) -> PairFactory:
    kwargs = dict(image_size=16, batch_size=6, target="noisy", need_second=False, seed=11)
    kwargs.update(overrides)
    return PairFactory(cache, **kwargs)


def test_noisy_target_comes_from_a_different_replica(burst_dataset: Path) -> None:
    factory = _factory(_cache(burst_dataset), need_second=True)
    _, info = factory.sample_batch(count=32, return_info=True)
    for sample in info:
        assert sample.target_replica is not None
        assert sample.second_replica is not None
        assert sample.input_replica != sample.target_replica
        assert sample.second_replica not in (sample.input_replica, sample.target_replica)


def test_clean_target_has_no_target_replica(burst_dataset: Path) -> None:
    factory = _factory(_cache(burst_dataset), target="clean")
    batch, info = factory.sample_batch(count=8, return_info=True)
    assert all(sample.target_replica is None for sample in info)
    assert batch.second is None


def test_crops_are_aligned_across_input_target_and_second(burst_dataset: Path) -> None:
    cache = _cache(burst_dataset)
    factory = _factory(cache, need_second=True)
    batch, info = factory.sample_batch(count=4, return_info=True)
    by_index = {source.source_index: source for source in cache.train_sources}
    for position, sample in enumerate(info):
        source = by_index[sample.source_index]
        top, left = sample.crop_yx
        window = np.s_[top : top + 16, left : left + 16]

        def expected(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(array[window].astype(np.float32) / 255.0 * 2.0 - 1.0)

        assert torch.allclose(batch.inputs[position, 0], expected(source.frames[sample.input_replica]))
        assert torch.allclose(batch.targets[position, 0], expected(source.frames[sample.target_replica]))
        assert torch.allclose(batch.second[position, 0], expected(source.frames[sample.second_replica]))


def test_batches_are_deterministic_given_the_seed(burst_dataset: Path) -> None:
    first = _factory(_cache(burst_dataset)).sample_batch()
    second = _factory(_cache(burst_dataset)).sample_batch()
    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.targets, second.targets)


def test_state_dict_round_trip_resumes_the_stream(burst_dataset: Path) -> None:
    factory = _factory(_cache(burst_dataset))
    factory.sample_batch()
    state = factory.state_dict()
    expected = factory.sample_batch()
    fresh = _factory(_cache(burst_dataset))
    fresh.load_state_dict(state)
    resumed = fresh.sample_batch()
    assert torch.equal(resumed.inputs, expected.inputs)
    assert torch.equal(resumed.targets, expected.targets)


def test_val_batch_is_deterministic_and_carries_clean(burst_dataset: Path) -> None:
    factory = _factory(_cache(burst_dataset))
    first = factory.val_batch(count=2)
    second = factory.val_batch(count=2)
    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.clean, second.clean)
    assert not torch.equal(first.inputs, first.second)  # independent replicas


def test_too_few_replicas_is_rejected_up_front(tmp_path: Path) -> None:
    root = write_burst(tmp_path / "data", replicas=2)
    cache = _cache(root)
    with pytest.raises(ValueError, match="at least 3 distinct replicas"):
        _factory(cache, need_second=True)
    factory = _factory(cache)  # noisy target alone needs only 2
    batch = factory.sample_batch(count=4)
    assert batch.inputs.shape == (4, 1, 16, 16)


def test_crop_size_larger_than_source_is_rejected(burst_dataset: Path) -> None:
    with pytest.raises(ValueError, match="crops"):
        _factory(_cache(burst_dataset), image_size=32)


def test_gradient_targets_default_to_absent(burst_dataset: Path) -> None:
    factory = _factory(_cache(burst_dataset))
    assert factory.sample_batch(count=2).gradient_targets is None
    assert factory.val_batch(count=1).gradient_targets is None


def test_gradient_target_clean_matches_the_clean_crop(burst_dataset: Path) -> None:
    cache = _cache(burst_dataset)
    factory = _factory(cache, gradient_target="clean")
    batch, info = factory.sample_batch(count=6, return_info=True)
    by_index = {source.source_index: source for source in cache.train_sources}
    for position, sample in enumerate(info):
        source = by_index[sample.source_index]
        top, left = sample.crop_yx
        window = np.s_[top : top + 16, left : left + 16]
        expected = torch.from_numpy(
            source.clean[window].astype(np.float32) / 255.0 * 2.0 - 1.0
        )
        assert torch.allclose(batch.gradient_targets[position, 0], expected)


def test_gradient_target_noisy_mean_is_the_leave_one_out_average(burst_dataset: Path) -> None:
    cache = _cache(burst_dataset)
    factory = _factory(cache, gradient_target="noisy_mean")
    batch, info = factory.sample_batch(count=12, return_info=True)
    by_index = {source.source_index: source for source in cache.train_sources}
    for position, sample in enumerate(info):
        source = by_index[sample.source_index]
        top, left = sample.crop_yx
        window = np.s_[top : top + 16, left : left + 16]
        others = [
            source.frames[replica][window].astype(np.float64)
            for replica in range(len(source.frames))
            if replica != sample.input_replica
        ]
        expected = np.mean(others, axis=0) / 255.0 * 2.0 - 1.0
        actual = batch.gradient_targets[position, 0].numpy()
        assert np.allclose(actual, expected, atol=1e-5)


def _write_file_targets(directory: Path, cache: BurstCache) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(5)
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[int, np.ndarray] = {}
    for source in list(cache.train_sources) + list(cache.val_sources):
        array = rng.random(source.clean.shape[:2]).astype(np.float32)
        np.save(directory / f"{source.source_index:05d}.npy", array)
        arrays[source.source_index] = array
    return arrays


def test_gradient_target_file_serves_aligned_crops(burst_dataset: Path, tmp_path: Path) -> None:
    cache = _cache(burst_dataset)
    arrays = _write_file_targets(tmp_path / "targets", cache)
    factory = _factory(
        cache, gradient_target="file", gradient_target_dir=tmp_path / "targets"
    )
    batch, info = factory.sample_batch(count=6, return_info=True)
    for position, sample in enumerate(info):
        top, left = sample.crop_yx
        expected = arrays[sample.source_index][top : top + 16, left : left + 16] * 2.0 - 1.0
        assert np.allclose(batch.gradient_targets[position, 0].numpy(), expected, atol=1e-6)
    val = factory.val_batch(count=2)
    assert val.gradient_targets is not None
    assert val.gradient_targets.shape == (2, 1, 16, 16)


def test_gradient_target_file_missing_or_misshapen_is_rejected(
    burst_dataset: Path, tmp_path: Path
) -> None:
    cache = _cache(burst_dataset)
    with pytest.raises(ValueError, match="distill-targets"):
        _factory(cache, gradient_target="file", gradient_target_dir=tmp_path / "empty")
    targets_dir = tmp_path / "targets"
    _write_file_targets(targets_dir, cache)
    first = cache.train_sources[0]
    np.save(targets_dir / f"{first.source_index:05d}.npy", np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        _factory(cache, gradient_target="file", gradient_target_dir=targets_dir)


def test_noisy_mean_target_is_the_leave_one_out_mean_of_the_other_replicas(tmp_path: Path) -> None:
    """``target: noisy_mean`` feeds the fidelity terms the mean of every replica
    EXCEPT the input: unbiased like a fresh frame, ~1/(N-1) of its variance,
    and never containing the input's own noise (which would pull the optimum
    toward the identity)."""
    dataset = write_burst(tmp_path / "data", replicas=5)
    cache = BurstCache(dataset, channels=1, min_replicas=2, min_size=16, val_fraction=0.34, split_seed=7)
    factory = PairFactory(cache, image_size=16, batch_size=3, target="noisy_mean", need_second=True, seed=1)
    batch, info = factory.sample_batch(return_info=True)
    assert batch.gradient_targets is None
    by_index = {source.source_index: source for source in cache.train_sources}
    for item, sample in enumerate(info):
        assert sample.target_replica is None
        assert sample.second_replica != sample.input_replica
        source = by_index[sample.source_index]
        top, left = sample.crop_yx
        window = np.s_[top : top + 16, left : left + 16]
        others = [
            frame[window].astype(np.float64)
            for replica, frame in enumerate(source.frames)
            if replica != sample.input_replica
        ]
        expected = np.mean(others, axis=0) / 255.0 * 2.0 - 1.0
        np.testing.assert_allclose(batch.targets[item, 0].numpy(), expected, atol=1e-5)
        # The input's own frame is absent from the target (it differs from the
        # all-replica mean exactly by the input's share).
        full_mean = np.mean([f[window].astype(np.float64) for f in source.frames], axis=0)
        assert not np.allclose(full_mean / 255.0 * 2.0 - 1.0, expected, atol=1e-6)

    validation = factory.val_batch(count=2)
    for item in range(2):
        source = cache.val_sources[item % len(cache.val_sources)]
        height, width = source.clean.shape
        window = np.s_[(height - 16) // 2 : (height - 16) // 2 + 16, (width - 16) // 2 : (width - 16) // 2 + 16]
        others = [f[window].astype(np.float64) for f in source.frames[1:]]
        expected = np.mean(others, axis=0) / 255.0 * 2.0 - 1.0
        np.testing.assert_allclose(validation.targets[item, 0].numpy(), expected, atol=1e-5)


def test_noisy_mean_target_requires_two_replicas(tmp_path: Path) -> None:
    dataset = write_burst(tmp_path / "data", replicas=1)
    cache = BurstCache(dataset, channels=1, min_replicas=1, min_size=16, val_fraction=0.34, split_seed=7)
    with pytest.raises(ValueError, match="at least 2"):
        PairFactory(cache, image_size=16, batch_size=1, target="noisy_mean")


def test_defect_augmentation_adds_one_field_to_every_tensor_of_a_sample(tmp_path: Path) -> None:
    """With augmentation on, input / target / second / gradient target of a
    sample all move by the SAME additive field (so Noise2Noise's independence
    argument is untouched), the field is bounded by the configured contrast,
    and the (source, crop, replica) draws match the un-augmented stream."""
    from edge_denoise.config import DefectAugmentConfig

    dataset = write_burst(tmp_path / "data", replicas=5)
    cache = BurstCache(dataset, channels=1, min_replicas=2, min_size=16, val_fraction=0.34, split_seed=7)
    common = dict(image_size=16, batch_size=1, target="noisy", need_second=True, gradient_target="clean", seed=4)
    plain = PairFactory(cache, **common)
    aug = PairFactory(
        cache,
        defect_augment=DefectAugmentConfig(probability=1.0, max_count=2, contrast=(0.05, 0.08)),
        **common,
    )
    plain_batch, plain_info = plain.sample_batch(return_info=True)
    aug_batch, aug_info = aug.sample_batch(return_info=True)
    assert plain_info == aug_info  # same source, crop and replicas for the first sample
    delta_in = (aug_batch.inputs - plain_batch.inputs)[0, 0]
    delta_t = (aug_batch.targets - plain_batch.targets)[0, 0]
    delta_s = (aug_batch.second - plain_batch.second)[0, 0]
    delta_g = (aug_batch.gradient_targets - plain_batch.gradient_targets)[0, 0]
    assert float(delta_in.abs().max()) > 0.0
    # Model units are 2x the [0, 1] contrast; clipping at the rails can only shrink a delta.
    assert float(delta_in.abs().max()) <= 2.0 * 0.08 * 2 + 1e-6
    unclipped = (aug_batch.inputs.abs() < 1.0) & (plain_batch.inputs.abs() < 1.0)
    unclipped = unclipped[0, 0] & (aug_batch.targets.abs() < 1.0)[0, 0] & (aug_batch.second.abs() < 1.0)[0, 0]
    assert torch.allclose(delta_in[unclipped], delta_t[unclipped], atol=1e-6)
    assert torch.allclose(delta_in[unclipped], delta_s[unclipped], atol=1e-6)
    assert torch.allclose(delta_in[unclipped], delta_g[unclipped], atol=1e-6)
    # Validation batches are never augmented.
    assert torch.equal(aug.val_batch(count=1).inputs, plain.val_batch(count=1).inputs)

    off = PairFactory(cache, defect_augment=DefectAugmentConfig(probability=0.0), **common)
    off_batch, off_info = off.sample_batch(return_info=True)
    assert off_info == plain_info and torch.equal(off_batch.inputs, plain_batch.inputs)


def test_clip_debiaser_inverts_the_clipped_poisson_mean() -> None:
    """The stored frames are min(Pois(peak x), peak)/peak, so their mean sits
    below x (by ~0.06 at x = 0.85 for peak 10); the debiaser maps that mean
    back onto x, and a Monte-Carlo mean of 15 clipped frames lands within the
    sampling error of x after the correction."""
    from edge_denoise.data import ClipDebiaser, clipped_poisson_mean

    peak = 10.0
    x = np.array([0.15, 0.4, 0.85])
    g = clipped_poisson_mean(x, peak)
    assert (g < x).all() and g[0] > 0.149 and 0.05 < x[2] - g[2] < 0.07
    debias = ClipDebiaser(peak)
    np.testing.assert_allclose(debias(g), x, atol=2e-3)
    rng = np.random.default_rng(0)
    for value in x:
        frames = np.minimum(rng.poisson(value * peak, size=(15, 20000)), peak) / peak
        raw_mean = frames.mean(axis=0)
        assert abs(raw_mean.mean() - clipped_poisson_mean(np.array([value]), peak)[0]) < 2e-3
        corrected = debias(raw_mean)
        # Residual = the second-order Jensen term of the convex inverse at the
        # bright end (~+0.004 at x = 0.85 for 15 frames) -- a 13x reduction of
        # the -0.057 clipping bias, at the level of the 8-bit quantization.
        assert abs(corrected.mean() - value) < 6e-3
        assert abs(corrected.mean() - value) < 0.15 * abs(raw_mean.mean() - value) + 1e-3
    assert float(debias(np.array([0.99]))[0]) == 1.0  # above g(1): clamped


def test_noisy_mean_target_can_be_debiased(tmp_path: Path) -> None:
    from edge_denoise.data import ClipDebiaser

    dataset = write_burst(tmp_path / "data", replicas=5)
    cache = BurstCache(dataset, channels=1, min_replicas=2, min_size=16, val_fraction=0.34, split_seed=7)
    common = dict(image_size=16, batch_size=2, target="noisy_mean", seed=9)
    plain = PairFactory(cache, **common).sample_batch()
    corrected = PairFactory(cache, target_debias_peak=10.0, **common).sample_batch()
    expected = ClipDebiaser(10.0)((plain.targets.numpy() + 1.0) / 2.0) * 2.0 - 1.0
    np.testing.assert_allclose(corrected.targets.numpy(), expected, atol=1e-5)
    assert (corrected.targets >= plain.targets - 1e-6).all()
    with pytest.raises(ValueError, match="noisy_mean"):
        PairFactory(cache, image_size=16, batch_size=2, target="noisy", target_debias_peak=10.0)
