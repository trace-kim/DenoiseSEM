from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import make_config, write_burst

from burst_diffusion.data import BurstCache

from edge_denoise.distill import (
    TARGETS_MANIFEST_NAME,
    _positions,
    denoise_full_frame,
    write_distill_targets,
)


def test_positions_cover_the_full_extent() -> None:
    assert _positions(50, 16, 12) == [0, 12, 24, 34]
    assert _positions(16, 16, 12) == [0]
    assert _positions(64, 16, 16) == [0, 16, 32, 48]


def test_identity_denoiser_round_trips_through_the_blend() -> None:
    """Every pixel is a convex combination of tile predictions, so an identity
    denoiser must reconstruct the frame exactly -- including at the borders,
    where only tile edges (the window floor) contribute."""
    rng = np.random.default_rng(3)
    frame = rng.random((50, 70))
    out = denoise_full_frame(lambda t: t, frame, tile=16, stride=12, tile_batch=5)
    assert np.allclose(out, frame, atol=1e-6)


def test_frame_smaller_than_tile_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than the tile"):
        denoise_full_frame(lambda t: t, np.zeros((8, 30)), tile=16, stride=12)


def test_write_distill_targets_averages_the_replicas(tmp_path: Path) -> None:
    dataset = write_burst(tmp_path / "data")
    config = make_config(dataset, tmp_path / "run", representation="image")
    manifest = write_distill_targets(
        config,
        denoise_fn=lambda t: t,
        out_dir=tmp_path / "targets",
        splits=("train", "val"),
        stride=12,
        tile_batch=7,
    )
    cache = BurstCache(
        dataset, channels=1, min_replicas=2, min_size=16, val_fraction=0.34, split_seed=7
    )
    covered = set(manifest["splits"]["train"]) | set(manifest["splits"]["val"])
    expected_sources = list(cache.train_sources) + list(cache.val_sources)
    assert covered == {source.source_index for source in expected_sources}
    for source in expected_sources:
        stored = np.load(tmp_path / "targets" / f"{source.source_index:05d}.npy")
        expected = np.mean(
            [frame.astype(np.float64) / 255.0 for frame in source.frames], axis=0
        )
        assert stored.dtype == np.float32
        assert stored.shape == source.clean.shape[:2]
        assert np.allclose(stored, expected, atol=1e-5)
    assert (tmp_path / "targets" / TARGETS_MANIFEST_NAME).is_file()


def test_locked_test_split_is_refused(tmp_path: Path) -> None:
    dataset = write_burst(tmp_path / "data")
    config = make_config(dataset, tmp_path / "run")
    with pytest.raises(ValueError, match="splits"):
        write_distill_targets(
            config, denoise_fn=lambda t: t, out_dir=tmp_path / "t", splits=("test",)
        )
