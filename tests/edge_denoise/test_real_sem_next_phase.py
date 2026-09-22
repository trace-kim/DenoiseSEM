from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from typer.testing import CliRunner

cv2 = pytest.importorskip("cv2")

from burst_diffusion.data import BurstCache
from edge_denoise.cli import app
from edge_denoise.config import Config, RealMatchingConfig, load_config
from edge_denoise.gradient import reconstruct_from_sobel, sobel
from edge_denoise.model import build_model, initialize_from_image
from edge_denoise.real_data import prepare_real_dataset
from edge_denoise.real_matching import MatchedRealPairFactory, match_prediction
from edge_denoise.real_suite import PIPELINES, ROOT
from edge_denoise.train import Trainer, load_checkpoint, load_init_weights


@pytest.fixture
def prepared_uint8(tmp_path: Path) -> Path:
    """Actual uint8 affine/brightness repeats, distinct content and whole sites."""
    rng = np.random.default_rng(51)
    raw = tmp_path / "raw_uint8"
    yy, xx = np.mgrid[:56, :64]
    for site in range(4):
        directory = raw / f"site_{site}"
        directory.mkdir(parents=True)
        scene = 135 + 28 * np.sin(xx / 5.2) + 22 * np.cos(yy / 6.8)
        scene -= 60 * np.exp(-((xx - 20 - site) ** 2 + (yy - 23) ** 2) / 45)
        scene += cv2.GaussianBlur(rng.normal(0, 15, scene.shape), (0, 0), 1)
        for frame in range(8):
            matrix = np.array([[1 + frame * .0008, .001 * frame, .09 * frame],
                               [-.001 * frame, 1, -.07 * frame]])
            moved = cv2.warpAffine(scene, matrix, (64, 56), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REFLECT_101)
            pixels = np.rint(np.clip((1 + .002 * frame) * moved + frame * .1
                                     + rng.normal(0, .8, moved.shape), 0, 255)).astype(np.uint8)
            Image.fromarray(np.repeat(pixels[..., None], 3, axis=2)).save(directory / f"frame_{frame}.png")
    dataset = tmp_path / "prepared_uint8"
    prepare_real_dataset(raw, dataset, image_size=16, align="none")
    return dataset


def tiny_config(dataset: Path, run_dir: Path) -> Config:
    return Config.model_validate({
        "data": {"dataset_dir": dataset, "image_size": 16},
        "model": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1,
                  "attn_resolutions": [], "attention": False},
        "objective": {"representation": "image", "target": "noisy", "lambda_gradient": 0},
        "training": {"run_dir": run_dir, "max_steps": 1, "batch_size": 1,
                     "cpu_threads": 1, "device": "cpu", "val_images": 1},
    })


@pytest.fixture
def real_teacher(prepared_uint8, tmp_path):
    return Trainer(tiny_config(prepared_uint8, tmp_path / "teacher")).run()


@pytest.mark.parametrize("name", PIPELINES)
def test_every_recipe_trains_resumes_and_exports_native_uint8(name, prepared_uint8, real_teacher, tmp_path):
    recipe = load_config(ROOT / "edge_denoise/configs" / f"sem_real_{name}.yml").model_dump(mode="json")
    teacher_config = Config.model_validate(load_checkpoint(real_teacher)["config"])
    recipe["model"] = teacher_config.model.model_dump(mode="json")
    recipe["data"].update(dataset_dir=str(prepared_uint8), image_size=16,
                           real_matching_cache=str(tmp_path / "shared_matching.json"))
    run_dir = tmp_path / name
    recipe["training"].update(run_dir=str(run_dir), init_checkpoint=None if name == "grad" else str(real_teacher),
                               max_steps=1, device="cpu", batch_size=1, accumulation_steps=1,
                               cpu_threads=1, val_images=1, profile=True)
    path = tmp_path / f"{name}.yml"
    path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["train", "--config", str(path)])
    assert result.exit_code == 0, (result.output, result.exception)
    checkpoint = run_dir / "ckpt_latest.pt"
    first = load_checkpoint(checkpoint)
    assert first["step"] == 1
    assert first["config"]["data"]["real_matching"]["registration"] == "affine"
    with pytest.warns(UserWarning, match="different config"):
        result = runner.invoke(app, ["train", "--config", str(path), "--max-steps", "2", "--resume"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert load_checkpoint(checkpoint)["step"] == 2
    source = tmp_path / "raw_uint8/site_0/frame_0.png"
    destination = tmp_path / f"images_{name}"
    result = runner.invoke(app, ["denoise", "--checkpoint", str(checkpoint), "--input", str(source),
                                  "--out", str(destination), "--device", "cpu", "--tile-batch", "3"])
    assert result.exit_code == 0, (result.output, result.exception)
    with Image.open(destination / "frame_0_denoised.png") as image:
        pixels = np.asarray(image)
        assert image.mode == "RGB" and pixels.dtype == np.uint8 and pixels.shape == (56, 64, 3)
        np.testing.assert_array_equal(pixels[..., 0], pixels[..., 1])
    with Image.open(destination / "frame_0_input.png") as image, Image.open(source) as original:
        np.testing.assert_array_equal(np.asarray(image), np.asarray(original))
    assert json.loads((run_dir / "training_status.json").read_text())["status"] == "complete"
    timing = json.loads((run_dir / "timings.json").read_text())
    assert all(timing["stage_seconds"][stage] >= 0 for stage in
               ("sample_batch", "transfer", "forward_loss_backward", "validation", "checkpoint"))


def test_hybrid_initialization_preserves_teacher_function(real_teacher, prepared_uint8, tmp_path):
    cfg = tiny_config(prepared_uint8, tmp_path / "hybrid")
    state = load_init_weights(real_teacher)
    image = build_model(cfg).eval()
    image.load_state_dict(state)
    cfg.objective.representation = "hybrid"
    hybrid = build_model(cfg).eval()
    initialize_from_image(hybrid, state)
    inputs = torch.rand(3, 1, 16, 16) * 2 - 1
    with torch.no_grad():
        torch.testing.assert_close(image(inputs), hybrid(inputs), rtol=0, atol=0)
    assert torch.count_nonzero(hybrid.unet.conv_in.weight[:, 1:]) == 0
    broken = dict(state)
    broken.pop("unet.conv_out.bias")
    with pytest.raises(RuntimeError, match="Missing key"):
        initialize_from_image(hybrid, broken)


def test_gradient_consistency_maps_images_before_sobel_and_erodes_invalid_support(tmp_path):
    cfg = tiny_config(tmp_path, tmp_path / "unused")
    cfg.objective.lambda_consistency = 1
    cfg.objective.consistency_domain = "gradient"
    trainer = Trainer.__new__(Trainer)
    trainer.config = cfg
    other = torch.randn(1, 1, 20, 20, requires_grad=True)
    matrix = torch.tensor([[[1.01, .07, -.8], [-.02, .99, .5]]])
    brightness = torch.tensor([[1.4, .3]])
    mapped = match_prediction(other, matrix, brightness)
    prediction = (mapped.detach() + .2).requires_grad_()  # Constant offset has no Sobel contribution.
    valid = torch.ones(1, 1, 20, 20, dtype=torch.bool)
    valid[..., 8, 9] = False
    contaminated = prediction.clone()
    contaminated[..., 8, 9] += 100
    terms = trainer._loss_terms(contaminated, torch.zeros_like(prediction), other,
                                prediction_second=other, second_matrices=matrix,
                                second_brightness=brightness, second_valid=valid, margin=3)
    assert terms["consistency"].item() < 1e-12
    changed = trainer._loss_terms(prediction + torch.randn_like(prediction) * .01,
                                  torch.zeros_like(prediction), other, prediction_second=other,
                                  second_matrices=matrix, second_brightness=brightness, second_valid=valid, margin=3)
    changed["consistency"].backward()
    assert torch.isfinite(other.grad).all() and other.grad.abs().sum() > 0
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="no valid"):
        trainer._loss_terms(prediction, prediction, other, prediction_second=other,
                            second_valid=torch.zeros_like(valid), margin=3)


def test_matching_cache_and_resume_do_not_reestimate_geometry(prepared_uint8, tmp_path, monkeypatch):
    cfg = tiny_config(prepared_uint8, tmp_path / "first")
    cfg.data.real_matching = RealMatchingConfig(registration="affine", brightness="percentile")
    cfg.data.real_matching_cache = tmp_path / "matching.json"
    trainer = Trainer(cfg)
    checkpoint = trainer.run()
    def forbidden(*args, **kwargs):
        raise AssertionError("unchanged measurements were recomputed")
    monkeypatch.setattr(MatchedRealPairFactory, "_measure", forbidden)
    restored = Trainer(cfg, resume_from=checkpoint)
    expected, actual = trainer.factory.sample_batch(), restored.factory.sample_batch()
    assert torch.equal(expected.inputs, actual.inputs) and torch.equal(expected.targets, actual.targets)
    new_config = cfg.model_copy(deep=True)
    new_config.training.run_dir = tmp_path / "second"
    another = Trainer(new_config)
    assert another.factory.measurements == trainer.factory.measurements
    corrupted = json.loads(cfg.data.real_matching_cache.read_text())
    corrupted["dataset_fingerprint"] = "changed"
    cfg.data.real_matching_cache.write_text(json.dumps(corrupted))
    new_config.training.run_dir = tmp_path / "bad"
    with pytest.raises(ValueError, match="cache identity differs"):
        Trainer(new_config)
    # A valid checkpoint remains independently resumable even if cache is absent.
    cfg.data.real_matching_cache.unlink()
    Trainer(cfg, resume_from=checkpoint)


def test_gradient_target_is_sobel_of_matched_image_and_inputs_stay_native(prepared_uint8, tmp_path):
    cfg = tiny_config(prepared_uint8, tmp_path / "grad")
    cfg.objective.representation = "gradient"
    cfg.objective.lambda_image, cfg.objective.lambda_gradient = 0, 1
    cfg.data.real_matching = RealMatchingConfig(registration="affine", brightness="percentile")
    trainer = Trainer(cfg)
    batch, infos = trainer.factory.sample_batch(return_info=True)
    info = infos[0]
    source = next(s for s in trainer.cache.train_sources if s.source_index == info.source_index)
    y, x = info.crop_yx
    expected = source.frames[info.input_replica, y:y + 16, x:x + 16].astype(np.float32) / 255 * 2 - 1
    np.testing.assert_array_equal(batch.inputs[0, 0], expected)
    predicted = sobel(batch.targets)
    assert trainer._loss_terms(predicted, batch.targets, None, target_valid=batch.target_valid,
                                margin=batch.loss_margin)["gradient"] == 0
    batch.targets[..., 8, 8] += 100
    valid = batch.target_valid.clone()
    valid[..., 8, 8] = False
    assert trainer._loss_terms(predicted, batch.targets, None, target_valid=valid,
                                margin=batch.loss_margin)["gradient"] == 0
    cfg.objective.lambda_consistency = 1
    cfg.training.run_dir = tmp_path / "invalid"
    with pytest.raises(ValueError, match="gradient representation does not support consistency"):
        Trainer(cfg)


def test_gradient_reconstruction_delivered_uint8_retains_mean_and_edge_geometry(tmp_path):
    from edge_denoise.uint8_output import prediction_uint8

    yy, xx = np.mgrid[:48, :48]
    raw = np.rint(50 + 140 / (1 + np.exp(-(xx - 17.4) / 1.2))).astype(np.uint8)
    tensor = torch.from_numpy(raw.astype(np.float32) / 255 * 2 - 1)[None, None]
    recovered = reconstruct_from_sobel(sobel(tensor), mean=tensor.mean(dim=(1, 2, 3)))
    pixels, _ = prediction_uint8((recovered[0, 0].numpy().astype(np.float64) + 1) / 2, 0, 255)
    Image.fromarray(pixels).save(tmp_path / "recovered.png")
    with Image.open(tmp_path / "recovered.png") as image:
        pixels = np.asarray(image).copy()
    assert np.max(np.abs(pixels.astype(int) - raw.astype(int))) <= 1
    assert abs(float(pixels.mean()) - float(raw.mean())) < .1


def test_gradient_reconstruction_full_frame_has_no_tile_seams(tmp_path):
    from edge_denoise.infer import Denoiser
    from edge_denoise.uint8_output import prediction_uint8

    class IdentityGradient(torch.nn.Module):
        def predict_image(self, frames):
            return reconstruct_from_sobel(sobel(frames), mean=frames.mean(dim=(1, 2, 3)))
    cfg = tiny_config(tmp_path, tmp_path / "unused")
    cfg.data.black_level, cfg.data.white_level = 0, 255
    yy, xx = np.mgrid[:56, :64]
    raw = np.rint(110 + 35 * np.sin(xx / 7) + 20 * np.cos(yy / 8)).astype(np.uint8)
    denoiser = Denoiser(IdentityGradient(), config=cfg, device=torch.device("cpu"))
    outputs = []
    for batch in (1, 5):
        result = denoiser.denoise_full(raw / 255., stride=8, tile_batch=batch, clip_output=False)
        pixels, _ = prediction_uint8(result, 0, 255)
        path = tmp_path / f"batch_{batch}.png"
        Image.fromarray(pixels).save(path)
        with Image.open(path) as image:
            outputs.append(np.asarray(image).copy())
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert np.abs(outputs[0].astype(int) - raw.astype(int)).max() <= 1


def test_nonfinite_loss_is_reported_as_failure_without_a_completed_checkpoint(prepared_uint8, tmp_path, monkeypatch):
    trainer = Trainer(tiny_config(prepared_uint8, tmp_path / "failed"))
    monkeypatch.setattr(trainer, "_combine", lambda terms: torch.tensor(float("nan"), requires_grad=True))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        trainer.run()
    record = json.loads((trainer.run_dir / "training_status.json").read_text())
    assert record["status"] == "failed" and record["step"] == 0
    assert not trainer.latest_checkpoint_path.exists()


def test_real_validation_does_not_report_float_prediction_noise(prepared_uint8, tmp_path):
    class Writer:
        def __init__(self):
            self.tags = []
        def add_scalar(self, tag, *args):
            self.tags.append(tag)
        def add_image(self, *args):
            pass
    trainer = Trainer(tiny_config(prepared_uint8, tmp_path / "run"))
    writer = Writer()
    trainer._validate(writer)
    assert "val/loss_image" in writer.tags
    assert "val/consistency_sigma" not in writer.tags


def test_single_gpu_selection_preserves_scheduler_mask(monkeypatch):
    from edge_denoise.distributed import DistributedRuntime
    from runctl.control import select_device

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second,GPU-third,GPU-fourth")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    calls = []
    monkeypatch.setattr(torch.cuda, "set_device", calls.append)
    runtime = DistributedRuntime("cuda:2")
    assert runtime.device == torch.device("cuda:2") and calls == [torch.device("cuda:2")]
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-first,GPU-second,GPU-third,GPU-fourth"
    with pytest.raises(RuntimeError, match="one isolated"):
        select_device("cuda")  # Existing runctl workers keep their isolation rule.
    with pytest.raises(RuntimeError, match="outside"):
        DistributedRuntime("cuda:4")


def test_checkpoint_rng_uses_only_selected_gpu(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from edge_denoise.train import save_checkpoint

    class Model:
        def parameters(self):
            return iter([SimpleNamespace(device=torch.device("cuda:2"))])
        def state_dict(self):
            return {"weight": torch.zeros(1)}
    state = SimpleNamespace(state_dict=lambda: {})
    calls = []
    def selected(device):
        calls.append(device)
        return torch.arange(8, dtype=torch.uint8)
    def forbidden():
        raise AssertionError("unselected GPU RNG was requested")
    monkeypatch.setattr(torch.cuda, "get_rng_state", selected)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model=Model(), ema=None, optimizer=state, factory=state, step=1,
                    config=tiny_config(tmp_path, tmp_path / "run"))
    assert calls == [torch.device("cuda:2")]
    assert len(load_checkpoint(path)["cuda_rng"]) == 1


def test_resume_rng_targets_selected_gpu_for_new_and_legacy_saves(prepared_uint8, tmp_path, monkeypatch):
    trainer = Trainer(tiny_config(prepared_uint8, tmp_path / "run"))
    path = trainer.run()
    payload = load_checkpoint(path)
    trainer.device = torch.device("cuda:2")
    trainer.ema = None  # Exercise RNG routing with CPU tensors, never real CUDA.
    calls = []
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: calls.append((int(state[0]), device)))
    def forbidden(*args):
        raise AssertionError("unselected GPU RNG was changed")
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden)
    payload["cuda_rng"] = [torch.tensor([5], dtype=torch.uint8)]
    trainer._restore(path, payload=payload)
    payload["cuda_rng"] = [torch.tensor([i], dtype=torch.uint8) for i in range(4)]
    trainer._restore(path, payload=payload)
    assert calls == [(5, torch.device("cuda:2")), (2, torch.device("cuda:2"))]
