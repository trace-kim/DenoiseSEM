"""The ``ddim`` noise type must reproduce DDIM's forward process q(x_t | x_0).

Reference: ``ddim/functions/losses.py`` builds the training input as
``x0 * a.sqrt() + e * (1 - a).sqrt()`` with ``a = (1 - b).cumprod(0)[t]`` and
``b`` from ``ddim/runners/diffusion.py::get_beta_schedule``, on data mapped to
[-1, 1] by ``data_transform`` (``rescaled: true``). Tests may import ``ddim``;
the package under test must not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from noising_pipeline import create_noisy_dataset
from noising_pipeline import pipeline

# The schedule of this repository's SEM DDIM configs (ddim/configs/sem*.yml).
SEM_SCHEDULE: dict[str, float | int | str] = {
    "beta_schedule": "linear",
    "beta_start": 0.001,
    "beta_end": 0.2,
    "num_diffusion_timesteps": 100,
}


def _save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _manifest_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _reference_betas(params: dict) -> np.ndarray:
    from ddim.runners.diffusion import get_beta_schedule

    return get_beta_schedule(
        params["beta_schedule"],
        beta_start=params["beta_start"],
        beta_end=params["beta_end"],
        num_diffusion_timesteps=params["num_diffusion_timesteps"],
    )


class _PresetNoise:
    """Stands in for ``numpy.random.Generator`` and hands back one known eps."""

    def __init__(self, epsilon: np.ndarray) -> None:
        self.epsilon = epsilon
        self.calls = 0

    def standard_normal(self, size=None, dtype=np.float64):
        assert tuple(size) == self.epsilon.shape
        self.calls += 1
        return self.epsilon.astype(dtype)


@pytest.mark.parametrize("schedule", ["linear", "quad", "const", "jsd", "sigmoid"])
def test_beta_schedules_are_identical_to_ddim_get_beta_schedule(schedule: str) -> None:
    params = {**SEM_SCHEDULE, "beta_schedule": schedule}
    np.testing.assert_array_equal(pipeline._ddim_betas(params), _reference_betas(params))


def test_fused_alpha_bar_is_ddim_cumprod_and_steps_is_the_timestep() -> None:
    betas = _reference_betas(SEM_SCHEDULE)
    for t in (1, 7, 100):
        fused = pipeline._fuse_noise_params({"ddim": dict(SEM_SCHEDULE)}, steps=t)["ddim"]
        assert fused["alpha_bar"] == pytest.approx(
            float(np.cumprod(1.0 - betas)[t - 1]), abs=1e-15
        )
        assert fused["t"] == t
        assert {key: fused[key] for key in SEM_SCHEDULE} == SEM_SCHEDULE
    with pytest.raises(ValueError, match="num_diffusion_timesteps"):
        pipeline._fuse_noise_params({"ddim": dict(SEM_SCHEDULE)}, steps=101)


def test_fused_latent_equals_the_ddim_training_formula_for_the_same_epsilon() -> None:
    import torch

    rng = np.random.default_rng(3)
    observation = rng.random((6, 5), dtype=np.float32)
    epsilon = rng.standard_normal((6, 5), dtype=np.float32)
    t = 37
    resolved = pipeline._resolve_noise_params(("ddim",), {"ddim": SEM_SCHEDULE})
    effective = pipeline._fuse_noise_params(resolved, steps=t)
    noise = _PresetNoise(epsilon)
    latent = pipeline._apply_ddim(
        observation, resolved["ddim"], effective["ddim"], noise, step_count=t, step_mode="fused"
    )

    # ddim/functions/losses.py::noise_estimation_loss on data_transform's 2x - 1.
    b = torch.from_numpy(_reference_betas(SEM_SCHEDULE)).float()
    a = (1 - b).cumprod(dim=0).index_select(0, torch.tensor([t - 1]))
    x0 = torch.from_numpy(2.0 * observation - 1.0)
    expected = x0 * a.sqrt() + torch.from_numpy(epsilon) * (1.0 - a).sqrt()

    assert noise.calls == 1
    assert latent.dtype == np.float32
    np.testing.assert_allclose(latent, expected.numpy(), rtol=0.0, atol=1e-5)


def test_iterative_chain_and_fused_marginal_share_the_ddim_moments(tmp_path: Path) -> None:
    value = 191  # u = 0.749, so x0 = 0.498
    source = tmp_path / "source"
    _save_image(source / "flat.png", np.full((512, 512), value, dtype=np.uint8))
    t = 10
    latents: dict[str, np.ndarray] = {}
    for mode in ("fused", "iterative"):
        manifest = create_noisy_dataset(
            source,
            tmp_path / mode,
            1,
            t,
            "ddim",
            noise_params={"ddim": SEM_SCHEDULE},
            step_mode=mode,
            progress=False,
            seed=5,
        )
        row = _manifest_rows(manifest)[0]
        alpha_bar = row["effective_noise_params"]["ddim"]["alpha_bar"]
        latent = np.load(tmp_path / mode / row["noisy_path"])
        latents[mode] = latent
        x0 = 2.0 * (value / 255.0) - 1.0
        # 262,144 pixels: standard errors are 6e-4 on the mean and 0.3% on the variance.
        assert float(latent.mean()) == pytest.approx(np.sqrt(alpha_bar) * x0, abs=5e-3)
        assert float(latent.var()) == pytest.approx(1.0 - alpha_bar, rel=0.03)
    assert not np.array_equal(latents["fused"], latents["iterative"])


def test_latents_are_unclipped_float32_npy_with_a_self_describing_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    ramp = np.linspace(0, 255, 32 * 24).reshape(32, 24).astype(np.uint8)
    deep = ramp.astype(np.uint16) * 257
    _save_image(source / "a_ramp.png", ramp)
    _save_image(source / "b_deep.tif", deep)
    t = SEM_SCHEDULE["num_diffusion_timesteps"]

    manifest = create_noisy_dataset(
        source, tmp_path / "out", 2, t, "DDIM", noise_params={"ddim": SEM_SCHEDULE}, progress=False
    )
    rows = _manifest_rows(manifest)

    assert [row["noisy_path"] for row in rows] == [
        "noisy/00000_00000.npy",
        "noisy/00000_00001.npy",
        "noisy/00001_00000.npy",
        "noisy/00001_00001.npy",
    ]
    assert [row["bit_depth"] for row in rows] == [8, 8, 16, 16]
    with Image.open(tmp_path / "out" / rows[0]["clean_path"]) as clean:
        np.testing.assert_array_equal(np.asarray(clean), ramp)
    with Image.open(tmp_path / "out" / rows[2]["clean_path"]) as clean:
        np.testing.assert_array_equal(np.asarray(clean), deep)
    alpha_bar_t = float(np.cumprod(1.0 - _reference_betas(SEM_SCHEDULE))[-1])
    for row in rows:
        latent = np.load(tmp_path / "out" / row["noisy_path"])
        assert latent.dtype == np.float32
        assert latent.shape == tuple(row["shape"])
        # At t = T alpha_bar is ~2e-5: the latent is essentially eps, and nothing clipped it.
        assert latent.min() < -1.0 and latent.max() > 1.0
        assert row["noisy_encoding"] == {
            "clipped": False,
            "dtype": "float32",
            "scale": 1.0,
            "signal_range": [-1.0, 1.0],
        }
        assert row["noise_types"] == ["ddim"]
        assert row["noise_params"] == {"ddim": SEM_SCHEDULE}
        assert row["effective_noise_params"]["ddim"] == {
            **SEM_SCHEDULE,
            "t": t,
            "alpha_bar": pytest.approx(alpha_bar_t),
        }


def test_epsilon_is_recoverable_from_the_latent_clean_png_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    texture = np.random.default_rng(11).integers(0, 256, size=(256, 256), dtype=np.uint8)
    _save_image(source / "texture.png", texture)
    t = 40

    manifest = create_noisy_dataset(
        source, tmp_path / "out", 1, t, "ddim", noise_params={"ddim": SEM_SCHEDULE}, progress=False
    )
    row = _manifest_rows(manifest)[0]
    encoding = row["noisy_encoding"]
    alpha_bar = row["effective_noise_params"]["ddim"]["alpha_bar"]
    with Image.open(tmp_path / "out" / row["clean_path"]) as clean:
        x0 = 2.0 * (np.asarray(clean, dtype=np.float32) / np.float32(255.0)) - 1.0
    latent = np.load(tmp_path / "out" / row["noisy_path"]) / encoding["scale"]
    epsilon = (latent - np.sqrt(alpha_bar) * x0) / np.sqrt(1.0 - alpha_bar)

    # 65,536 pixels: standard errors are 0.004 on the mean and the correlation, 0.3% on the std.
    assert float(epsilon.mean()) == pytest.approx(0.0, abs=0.02)
    assert float(epsilon.std()) == pytest.approx(1.0, rel=0.015)
    assert abs(np.corrcoef(epsilon.ravel(), x0.ravel())[0, 1]) < 0.02


def test_latents_are_seed_deterministic_with_independent_replicas(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _save_image(source / "gray.png", np.full((8, 8), 100, dtype=np.uint8))
    for name in ("first", "second"):
        create_noisy_dataset(source, tmp_path / name, 2, 3, "ddim", seed=99, progress=False)
    first = [np.load(tmp_path / "first" / "noisy" / f"00000_{r:05d}.npy") for r in range(2)]
    second = [np.load(tmp_path / "second" / "noisy" / f"00000_{r:05d}.npy") for r in range(2)]
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[0], first[1])


def test_sequences_run_observation_noise_first_then_ddim_once_on_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    seen: dict[str, object] = {}
    real_apply_noise = pipeline._apply_noise

    def recording_apply(image, noise_type, params, rng):
        calls.append(noise_type)
        return real_apply_noise(image, noise_type, params, rng)

    def recording_ddim(observation, params, effective, rng, *, step_count, step_mode):
        calls.append("ddim")
        seen["observation"] = observation.copy()
        seen["step_count"] = step_count
        seen["step_mode"] = step_mode
        return 2.0 * observation - 1.0

    monkeypatch.setattr(pipeline, "_apply_noise", recording_apply)
    monkeypatch.setattr(pipeline, "_apply_ddim", recording_ddim)
    source = tmp_path / "source"
    _save_image(source / "gray.png", np.full((4, 4), 128, dtype=np.uint8))
    noise_params = {"gaussian": {"std": 0.0}, "poisson": {"peak": 1e9}}

    create_noisy_dataset(
        source,
        tmp_path / "iterative",
        1,
        3,
        ["gaussian", "poisson", "ddim"],
        noise_params=noise_params,
        step_mode="iterative",
        progress=False,
    )
    assert calls == ["gaussian", "poisson"] * 3 + ["ddim"]
    assert seen["step_count"] == 3 and seen["step_mode"] == "iterative"
    observation = seen["observation"]
    assert observation.dtype == np.float32
    assert 0.0 <= observation.min() and observation.max() <= 1.0
    np.testing.assert_allclose(observation, 128 / 255, atol=1e-3)

    calls.clear()
    create_noisy_dataset(
        source,
        tmp_path / "fused",
        1,
        3,
        ["gaussian", "poisson", "ddim"],
        noise_params=noise_params,
        progress=False,
    )
    assert calls == ["gaussian", "poisson", "ddim"]


def test_ddim_must_be_last_and_unique() -> None:
    assert pipeline._normalize_noise_types(["Poisson", " DDIM "]) == ("poisson", "ddim")
    for sequence in (["ddim", "poisson"], ["poisson", "ddim", "ddim"], ["ddim", "ddim"]):
        with pytest.raises(ValueError, match="exactly once and last"):
            pipeline._normalize_noise_types(sequence)


@pytest.mark.parametrize(
    ("override", "error", "match"),
    [
        ({"beta_schedule": "cosine"}, ValueError, "beta_schedule must be one of"),
        ({"beta_schedule": 3}, TypeError, "must be a string"),
        ({"num_diffusion_timesteps": 10.5}, ValueError, "must be an integer"),
        ({"num_diffusion_timesteps": 0}, ValueError, "positive integer"),
        ({"beta_start": -0.1}, ValueError, "greater than or equal to zero"),
        ({"beta_end": 1.5}, ValueError, r"within \[0, 1\]"),
        ({"beta_end": float("inf")}, ValueError, "must be finite"),
        ({"alpha_bar": 0.5}, ValueError, "unknown ddim parameter"),
    ],
)
def test_ddim_parameter_validation(override: dict, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        pipeline._resolve_noise_params(("ddim",), {"ddim": override})


def test_ddim_parameters_are_normalised_to_their_declared_types() -> None:
    resolved = pipeline._resolve_noise_params(
        ("ddim",), {"ddim": {"num_diffusion_timesteps": 100.0, "beta_schedule": " Linear "}}
    )["ddim"]
    assert resolved["num_diffusion_timesteps"] == 100
    assert isinstance(resolved["num_diffusion_timesteps"], int)
    assert resolved["beta_schedule"] == "linear"
    assert resolved["beta_start"] == 0.0001 and resolved["beta_end"] == 0.02


def test_png_rows_describe_their_clipped_integer_encoding(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _save_image(source / "a.png", np.full((3, 3), 10, dtype=np.uint8))
    _save_image(source / "b.tif", np.full((3, 3), 1000, dtype=np.uint16))
    rows = _manifest_rows(
        create_noisy_dataset(source, tmp_path / "out", 1, 1, "gaussian", progress=False)
    )
    assert rows[0]["noisy_encoding"] == {
        "clipped": True, "dtype": "uint8", "scale": 255.0, "signal_range": [0.0, 1.0],
    }
    assert rows[1]["noisy_encoding"] == {
        "clipped": True, "dtype": "uint16", "scale": 65535.0, "signal_range": [0.0, 1.0],
    }
    assert rows[0]["noisy_path"].endswith(".png") and rows[1]["noisy_path"].endswith(".png")
