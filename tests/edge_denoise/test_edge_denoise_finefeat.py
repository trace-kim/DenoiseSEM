from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from edge_denoise.finefeat import (
    BAND_LABELS,
    band_maps,
    find_features,
    fine_features,
    flat_and_edge_masks,
    gaussian_blur,
    label_components,
    robust_sigma,
    summarize,
)


def test_gaussian_blur_preserves_constants_and_mass() -> None:
    flat = np.full((32, 40), 0.37)
    np.testing.assert_allclose(gaussian_blur(flat, 2.0), flat, atol=1e-12)
    impulse = np.zeros((33, 33))
    impulse[16, 16] = 1.0
    blurred = gaussian_blur(impulse, 1.5)
    assert abs(blurred.sum() - 1.0) < 1e-9
    assert blurred[16, 16] == blurred.max()


def test_band_maps_partition_the_image() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((40, 48))
    everywhere = np.ones(image.shape, dtype=np.int64)
    bands = band_maps(image, everywhere)
    assert len(bands) == len(BAND_LABELS)
    # Sum of bands = image - G_{16}(image)  (telescoping difference of Gaussians).
    np.testing.assert_allclose(sum(bands), image - gaussian_blur(image, 16.0), atol=1e-10)


def test_masked_blur_does_not_bleed_across_the_mask() -> None:
    from edge_denoise.finefeat import masked_blur

    image = np.full((40, 40), 0.2)
    image[:, 20:] = 0.9  # a bright half in a different region
    labels = np.ones(image.shape, dtype=np.int64)
    labels[:, 18:22] = 0  # the edge ramp: structure, no region
    labels[:, 22:] = 2
    blurred = masked_blur(image, labels, 4.0)
    np.testing.assert_allclose(blurred[:, :18], 0.2, atol=1e-9)  # no bleed from the bright half
    np.testing.assert_allclose(blurred[:, 22:], 0.9, atol=1e-9)
    assert float(gaussian_blur(image, 4.0)[20, 16]) > 0.25  # the plain blur does bleed


def test_label_components_is_8_connected() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1, 1] = mask[2, 2] = True  # diagonal neighbours -> one component
    mask[6:8, 6:9] = True  # a separate block
    labels, count = label_components(mask)
    assert count == 2
    assert labels[1, 1] == labels[2, 2]
    assert labels[6, 6] != labels[1, 1]
    assert (labels[~mask] == 0).all()
    empty, none = label_components(np.zeros((4, 4), dtype=bool))
    assert none == 0 and not empty.any()


def test_robust_sigma_matches_normal_std_and_ignores_outliers() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(0.0, 0.01, size=20000)
    assert abs(robust_sigma(values) - 0.01) < 0.0005
    values[:50] = 5.0
    assert abs(robust_sigma(values) - 0.01) < 0.0005


def _planted_scene(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A flat 0.4 field with a bright bar (strong edges) and one soft dark
    blemish (Gaussian blob, peak contrast -0.06, sigma 2.5 px) inside the
    flat; returns (clean, blemish core mask)."""
    clean = np.full((96, 96), 0.4)
    clean[:, 60:72] = 0.7  # a bar: strong edges the region mask must exclude
    yy, xx = np.mgrid[0:96, 0:96]
    blob = np.exp(-((yy - 32.0) ** 2 + (xx - 22.0) ** 2) / (2.0 * 2.5**2))
    clean -= 0.06 * blob
    blemish = blob > 0.5
    return clean, blemish


def test_find_features_locates_a_planted_blemish_with_its_snr() -> None:
    from edge_denoise.finefeat import region_labels

    clean, blemish = _planted_scene()
    flat, edge = flat_and_edge_masks(clean)
    assert edge[:, 59:61].any() and edge[:, 71:73].any()
    regions = region_labels(clean)
    assert regions[32, 22] > 0  # the soft blemish stays inside its region
    assert regions[48, 60] == 0 and regions[48, 5] > 0  # the bar edge is structure, the far flat a region
    features, labels, band, threshold = find_features(clean, regions, peak=10.0)
    assert len(features) == 1
    feature = features[0]
    assert feature.contrast < 0  # dark blemish
    cy, cx = feature.centroid_yx
    assert abs(cy - 32.0) < 1.0 and abs(cx - 22.0) < 1.0
    assert (labels[blemish] == feature.label).mean() > 0.6
    expected_snr = abs(feature.contrast) * np.sqrt(feature.area) / np.sqrt(feature.local_intensity / 10.0)
    assert abs(feature.snr_single - expected_snr) < 1e-9


def test_retention_measures_the_contrast_fraction_an_estimator_keeps(tmp_path: Path) -> None:
    """Identity -> gain 1 / retention 1; a half-contrast estimator -> ~0.5;
    a blur-everything estimator -> ~0 and no false features."""
    from conftest import write_burst

    from burst_diffusion.data import BurstCache

    from edge_denoise.config import Config

    clean, blemish = _planted_scene()
    root = tmp_path / "data"
    burst = root / "burst"
    (burst / "clean").mkdir(parents=True)
    (burst / "noisy").mkdir(parents=True)
    from PIL import Image

    rng = np.random.default_rng(0)
    rows = []
    for source_index in range(3):
        scene = clean.copy()
        scene[10 + source_index, 10] = 0.41  # distinct content per source (split groups)
        Image.fromarray(np.rint(scene * 255).astype(np.uint8)).save(burst / "clean" / f"{source_index:05d}.png")
        for replica in range(4):
            noisy = np.clip(scene + rng.normal(0.0, 0.05, scene.shape), 0.0, 1.0)
            name = f"noisy/{source_index:05d}_{replica:05d}.png"
            Image.fromarray(np.rint(noisy * 255).astype(np.uint8)).save(burst / name)
            rows.append(
                json.dumps(
                    {
                        "source_index": source_index,
                        "replica_index": replica,
                        "clean_path": f"clean/{source_index:05d}.png",
                        "noisy_path": name,
                    }
                )
            )
    (burst / "manifest.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    config = Config.model_validate(
        {
            "data": {"dataset_dir": str(root), "image_size": 32, "val_fraction": 0.34, "split_seed": 7},
            "objective": {"representation": "image", "target": "noisy"},
            "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": []},
            "training": {"run_dir": str(tmp_path / "run")},
        }
    )
    # "Oracle" estimators that ignore their noisy input and return functions of clean.
    cache = BurstCache(root, channels=1, min_replicas=2, min_size=32, val_fraction=0.34, split_seed=7)
    scenes = {s.source_index: s.clean.astype(np.float64) / 255.0 for s in cache.all_sources}
    base = np.full_like(clean, 0.4)
    base[:, 60:72] = 0.7

    def make(fn):
        def denoise(frames: torch.Tensor) -> torch.Tensor:
            out = []
            for frame in frames:
                noisy01 = (frame[0].numpy() + 1.0) / 2.0
                # Locate the tile by matching the least-noisy scene crop is overkill;
                # tiles of every scene share the same layout, so use the planted clean.
                out.append(fn(noisy01))
            return torch.from_numpy(np.stack(out)[:, None].astype(np.float32)) * 2.0 - 1.0

        return denoise

    def tile_of(image: np.ndarray, noisy01: np.ndarray) -> np.ndarray:
        # Recover the crop window from the noisy tile's mean position: tiles are
        # taken at fixed strides, so compare against every window (tiny image).
        best, best_err = None, None
        for top in range(0, 96 - 32 + 1, 16):
            for left in range(0, 96 - 32 + 1, 16):
                window = image[top : top + 32, left : left + 32]
                err = float(np.mean((window - noisy01) ** 2))
                if best_err is None or err < best_err:
                    best, best_err = window, err
        assert best is not None
        return best

    arms = {
        "identity_oracle": make(lambda noisy01: tile_of(clean, noisy01)),
        "half_contrast": make(lambda noisy01: tile_of(base + 0.5 * (clean - base), noisy01)),
        "erased": make(lambda noisy01: tile_of(base, noisy01)),
    }
    results = fine_features(config, arms, out_dir=tmp_path / "out", split="val", num_seeds=2, stride=16)
    summary = results["summary"]
    assert summary["features_total"] >= 1
    ident = summary["methods"]["identity_oracle"]
    half = summary["methods"]["half_contrast"]
    erased = summary["methods"]["erased"]
    assert abs(ident["feature_retention_median"] - 1.0) < 0.05
    assert abs(half["feature_retention_median"] - 0.5) < 0.05
    assert abs(erased["feature_retention_median"]) < 0.05
    assert ident["false_features_total"] == 0 and erased["false_features_total"] == 0
    # Band gains on the blemish's 2-8 px scales follow the same ordering.
    assert ident["band_gain"][2] > half["band_gain"][2] > erased["band_gain"][2]
    # The classical rows are unbiased (gain ~1) but noisy (corr < identity).
    single = summary["methods"]["single_frame"]
    assert single["band_corr"][2] < ident["band_corr"][2]
    assert (tmp_path / "out" / "summary.md").is_file()
    assert (tmp_path / "out" / "feature_plate.png").is_file()
    assert (tmp_path / "out" / "fine_features.json").is_file()


def test_summarize_handles_sources_without_features() -> None:
    record = {
        "source_index": 0,
        "clean_band_rms": [1.0] * 5,
        "noise_band_rms_single": [1.0] * 5,
        "wiener_gain_single": [0.5] * 5,
        "wiener_gain_avg16": [0.9] * 5,
        "features": [],
        "methods": {
            "m": {
                "psnr_frame0": 30.0,
                "rms_flat": 0.01,
                "rms_edge": 0.02,
                "texture_rms_flat": 0.001,
                "corr_residual_grain": -0.5,
                "band": [{"gain": 0.1, "corr": 0.2, "rms_ratio": 0.3, "rms_out": 0.1}] * 5,
                "feature_retention": [],
                "false_features": 0,
                "false_feature_area": 0,
                "flat_pixels_searched": 100,
            }
        },
    }
    summary = summarize([record])
    assert summary["features_total"] == 0
    assert summary["methods"]["m"]["feature_retention_median"] is None
    assert summary["methods"]["m"]["false_features_per_1000_flat_px"] == 0.0
