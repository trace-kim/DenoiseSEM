"""Run provenance for edge_denoise: which code, data, and environment.

Reuses burst_diffusion's audited building blocks (``git_state``,
``environment_state``, ``file_sha256``, ``content_key``) and records the same
facts its provenance module records, against this package's config schema.
Written automatically when a training run finishes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from burst_diffusion.data import BurstCache, content_key, resolve_burst_dir
from burst_diffusion.provenance import environment_state, file_sha256, git_state

from .config import Config

PROVENANCE_NAME = "provenance.json"
SCHEMA_VERSION = 1


def dataset_fingerprint(config: Config, *, cache: BurstCache | None = None) -> dict:
    """Content identity and fixed split composition for synthetic or real data."""
    cache = cache if cache is not None else BurstCache(
        config.data.dataset_dir,
        channels=config.data.channels,
        min_replicas=1,
        min_size=1,
        val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction,
        split_seed=config.data.split_seed,
    )
    if cache.real_metadata is not None:
        return {
            "kind": "real_sem", "manifest_sha256": cache.real_fingerprint,
            "sources": len(cache.all_sources),
            "normalization": cache.real_metadata["normalization"],
            "registration": cache.real_metadata["registration"],
            "split": {split: [source.source_index for source in cache.sources_for_split(split)]
                      for split in ("train", "val", "test")},
        }
    keys = sorted(content_key(source.clean) for source in cache.all_sources)
    digest = hashlib.sha256("".join(keys).encode("ascii")).hexdigest()
    return {
        "burst_dir": str(resolve_burst_dir(config.data.dataset_dir)),
        "sources": len(keys),
        "distinct_contents": len(set(keys)),
        "content_digest_sha256": digest,
        "split": {
            "train": [s.source_index for s in cache.train_sources],
            "val": [s.source_index for s in cache.val_sources],
            "test": [s.source_index for s in cache.test_sources],
        },
        "duplicate_groups": [list(group) for group in cache.duplicate_groups],
    }


def _checkpoint_record(checkpoint: str | Path | None) -> dict | None:
    """Path, content hash, step, size and kind of a checkpoint file (None when
    absent), for either pipeline's payload."""
    if checkpoint is None:
        return None
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        return None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return {
        "path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "step": int(payload.get("step", -1)),
        "bytes": checkpoint_path.stat().st_size,
        "kind": str(payload.get("kind", "burst_diffusion")),
    }


def write_provenance(
    run_dir: str | Path,
    config: Config,
    *,
    config_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
    command: str | None = None,
    repo: str | Path | None = None,
    cache: BurstCache | None = None,
) -> Path:
    """Write ``provenance.json`` into ``run_dir``; returns its path."""
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo) if repo is not None else Path(__file__).resolve().parent.parent

    checkpoint_record = _checkpoint_record(checkpoint)
    # A warm start (training.init_checkpoint) is part of the recipe: record
    # which file, by content, the run started from, not just its path.
    init_record = _checkpoint_record(config.training.init_checkpoint)

    config_json = config.model_dump(mode="json")
    record = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "edge_denoise",
        "run_dir": str(destination),
        "config_path": str(config_path) if config_path is not None else None,
        "config": config_json,
        "config_sha256": hashlib.sha256(
            json.dumps(config_json, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "git": git_state(repo_root),
        "dataset": dataset_fingerprint(config, cache=cache),
        "environment": environment_state(),
        "checkpoint": checkpoint_record,
        "init_checkpoint": init_record,
        "command": command,
    }
    path = destination / PROVENANCE_NAME
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
