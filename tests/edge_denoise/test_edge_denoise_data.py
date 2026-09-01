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
