from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from ddim.spec import DdimTrainingSpec
from runctl.bundles import (
    ConfigurationError,
    DuplicateConfigurationKey,
    load_training_spec,
)
from runctl.schemas import ReproducibilityMode


BASE_CONFIG = """
schema_version: 1
data:
  dataset: SEM
  image_size: 32
  channels: 1
  logit_transform: false
  uniform_dequantization: false
  gaussian_dequantization: false
  random_flip: false
  rescaled: true
  num_workers: 0
  cache_in_memory: false
  recursive: false
  validation_split: 0.1
  split_seed: 2019
  extensions: [.png, .tif]
model:
  type: simple
  in_channels: 1
  out_ch: 1
  ch: 64
  ch_mult: [1, 2, 2, 2]
  num_res_blocks: 2
  attn_resolutions: [16]
  dropout: 0.0
  var_type: fixedlarge
  ema_rate: 0.999
  ema: true
  resamp_with_conv: true
diffusion:
  beta_schedule: linear
  beta_start: 0.001
  beta_end: 0.2
  num_diffusion_timesteps: 100
training:
  batch_size: 7
  max_steps: 20000
  checkpoint_every: 2500
  validation_every: 2500
  sample_every: 2500
  checkpoint_minutes: 30
sampling:
  batch_size: 8
  last_only: true
optim:
  weight_decay: 0.0
  optimizer: Adam
  lr: 0.0002
  beta1: 0.9
  amsgrad: false
  eps: 0.00000001
  grad_clip: 1.0
"""


def write_base(path: Path, text: str = BASE_CONFIG) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_ddim_spec_is_strict_and_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DdimTrainingSpec.model_validate({"max_steps": "20000"})
    with pytest.raises(ValidationError):
        DdimTrainingSpec.model_validate({"not_a_setting": 1})


def test_ddim_spec_rejects_silent_model_and_schedule_mistakes() -> None:
    with pytest.raises(ValidationError, match="GroupNorm"):
        DdimTrainingSpec(model_ch=48)
    with pytest.raises(ValidationError, match="downsampling factor"):
        DdimTrainingSpec(image_size=30)
    with pytest.raises(ValidationError, match="attention resolutions"):
        DdimTrainingSpec(attn_resolutions=(12,))
    with pytest.raises(ValidationError, match="checkpoint_every cannot exceed"):
        DdimTrainingSpec(max_steps=10, checkpoint_every=11, validation_every=10, sample_every=10)
    with pytest.raises(ValidationError, match="incompatible"):
        DdimTrainingSpec(cache_in_memory=True, random_flip=True)


def test_config_loader_applies_typed_explicit_overrides(tmp_path: Path) -> None:
    config = write_base(tmp_path / "sem.yml")
    spec = load_training_spec(
        config,
        {
            "label": "larger-image",
            "image_size": 64,
            "attn_resolutions": [16],
            "max_steps": 30_000,
            "reproducibility": "deterministic",
            "sample_every": None,
        },
        flow="ddim",
    )
    assert spec.label == "larger-image"
    assert spec.image_size == 64
    assert spec.max_steps == 30_000
    assert spec.reproducibility is ReproducibilityMode.DETERMINISTIC
    assert spec.extensions == (".png", ".tif")


def test_config_loader_rejects_duplicate_unknown_and_machine_specific_keys(tmp_path: Path) -> None:
    duplicate = BASE_CONFIG + "\ntraining:\n  max_steps: 7\n"
    with pytest.raises(DuplicateConfigurationKey):
        load_training_spec(write_base(tmp_path / "duplicate.yml", duplicate), flow="ddim")

    unknown = BASE_CONFIG.replace("  image_size: 32", "  image_size: 32\n  typo_size: 31")
    with pytest.raises(ConfigurationError, match="data.typo_size"):
        load_training_spec(write_base(tmp_path / "unknown.yml", unknown), flow="ddim")

    machine_path = BASE_CONFIG.replace("  dataset: SEM", "  dataset: SEM\n  data_path: C:/secret")
    with pytest.raises(ConfigurationError, match="data.data_path"):
        load_training_spec(write_base(tmp_path / "machine-path.yml", machine_path), flow="ddim")

    with pytest.raises(ConfigurationError, match="unknown training override"):
        load_training_spec(write_base(tmp_path / "override.yml"), {"batch_szie": 8}, flow="ddim")
