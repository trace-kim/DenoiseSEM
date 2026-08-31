"""Registers the ``ddim`` flow with :mod:`runctl`.

This module is the *only* seam between the reproducible orchestration layer and
the original DDIM research code.  It declares:

* which spec model describes a DDIM run (:class:`ddim.spec.DdimTrainingSpec`),
* how a ``configs/sem.yml`` style recipe maps onto that spec's flat fields,
* which trainer ``runctl``'s worker should call,
* the extra settings the wizard prompts for and the plan table displays.

``runctl`` imports this lazily through its flow registry, so importing
``runctl`` never pulls in PyTorch or the DDIM model code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runctl.bundles import (
    ConfigurationError,
    coerce_reproducibility,
    flatten_config,
    load_config_mapping,
    map_config_keys,
)
from runctl.registry import Flow, SpecOption

from .spec import DdimTrainingSpec


CONFIG_DIR = Path(__file__).resolve().parent / "configs"

#: The single active SEM recipe.  Experiment-varying values belong on the
#: command line, not in new copies of this file.
DEFAULT_CONFIG = CONFIG_DIR / "sem.yml"


#: Dotted recipe key -> flat spec field.  ``None`` means "recognised but not a
#: spec field" (``data.dataset`` is a guard, checked separately below).
CONFIG_KEY_MAP: dict[str, str | None] = {
    "schema_version": "schema_version",
    "data.dataset": None,
    "data.dataset_alias": "dataset_alias",
    "data.image_size": "image_size",
    "data.channels": "channels",
    "data.logit_transform": "logit_transform",
    "data.uniform_dequantization": "uniform_dequantization",
    "data.gaussian_dequantization": "gaussian_dequantization",
    "data.random_flip": "random_flip",
    "data.rescaled": "rescaled",
    "data.num_workers": "num_workers",
    "data.cache_in_memory": "cache_in_memory",
    "data.recursive": "recursive",
    "data.validation_split": "validation_split",
    "data.split_seed": "split_seed",
    "data.extensions": "extensions",
    "model.type": "model_type",
    "model.in_channels": "in_channels",
    "model.out_ch": "out_channels",
    "model.ch": "model_ch",
    "model.ch_mult": "ch_mult",
    "model.num_res_blocks": "num_res_blocks",
    "model.attn_resolutions": "attn_resolutions",
    "model.dropout": "dropout",
    "model.var_type": "var_type",
    "model.ema_rate": "ema_rate",
    "model.ema": "ema",
    "model.resamp_with_conv": "resamp_with_conv",
    "diffusion.beta_schedule": "beta_schedule",
    "diffusion.beta_start": "beta_start",
    "diffusion.beta_end": "beta_end",
    "diffusion.num_diffusion_timesteps": "diffusion_steps",
    "training.batch_size": "batch_size",
    "training.max_steps": "max_steps",
    "training.snapshot_freq": "checkpoint_every",
    "training.checkpoint_every": "checkpoint_every",
    "training.validation_freq": "validation_every",
    "training.validation_every": "validation_every",
    "training.sample_freq": "sample_every",
    "training.sample_every": "sample_every",
    "training.checkpoint_minutes": "checkpoint_minutes",
    "sampling.batch_size": "sampling_batch_size",
    "sampling.last_only": "sampling_last_only",
    "optim.weight_decay": "weight_decay",
    "optim.optimizer": "optimizer",
    "optim.lr": "lr",
    "optim.beta1": "beta1",
    "optim.amsgrad": "amsgrad",
    "optim.eps": "eps",
    "optim.grad_clip": "grad_clip",
    "experiment.label": "label",
    "experiment.dataset_alias": "dataset_alias",
    "experiment.seed": "seed",
    "experiment.reproducibility": "reproducibility",
}

TUPLE_FIELDS = {"extensions", "ch_mult", "attn_resolutions"}


def parse_config(config_path: Path | str) -> dict[str, Any]:
    """Turn a DDIM YAML recipe into ``DdimTrainingSpec`` keyword arguments."""

    flat = flatten_config(load_config_mapping(config_path))
    if flat.get("data.dataset", "SEM") != "SEM":
        raise ConfigurationError(
            "the active training configuration supports only data.dataset: SEM"
        )
    return map_config_keys(
        flat,
        CONFIG_KEY_MAP,
        tuple_fields=TUPLE_FIELDS,
        coercions={"reproducibility": coerce_reproducibility},
    )


def _train_from_manifest(*args: Any, **kwargs: Any) -> Any:
    """Import the trainer lazily so ``runctl --help`` does not import torch."""

    from .training import train_from_manifest

    return train_from_manifest(*args, **kwargs)


def _plan_rows(spec: DdimTrainingSpec) -> tuple[tuple[str, str], ...]:
    return (
        ("Resolution / model", f"{spec.image_size}px, ch={spec.model_ch}, mult={spec.ch_mult}"),
        ("Diffusion", f"{spec.diffusion_steps} steps, beta={spec.beta_start}..{spec.beta_end}"),
        ("EMA", f"{'on' if spec.ema else 'off'}, rate={spec.ema_rate}"),
    )


#: Flow-specific settings surfaced by ``runctl train wizard`` and restated on
#: the canonical command line as ``--set NAME=VALUE``.
OPTIONS: tuple[SpecOption, ...] = (
    SpecOption("image_size", "Image size", "int"),
    SpecOption("model_ch", "Model base channels", "int"),
    SpecOption("ch_mult", "Channel multipliers", "int_list"),
    SpecOption("diffusion_steps", "Diffusion steps", "int"),
    SpecOption("beta_start", "Beta start", "float"),
    SpecOption("beta_end", "Beta end", "float"),
    SpecOption("ema_rate", "EMA rate", "float"),
)


FLOW = Flow(
    name="ddim",
    spec_model=DdimTrainingSpec,
    parse_config=parse_config,
    trainer=_train_from_manifest,
    default_config=DEFAULT_CONFIG,
    description="Original DDIM (Song/Meng/Ermon) diffusion training on SEM images.",
    options=OPTIONS,
    plan_rows=_plan_rows,
)
