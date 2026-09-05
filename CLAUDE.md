# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Five independently owned packages, separated so that the original DDIM research
code and the add-ons built on top of it do not share a namespace:

1. **`ddim/`** — the original Song/Meng/Ermon DDIM implementation, namespaced as
   a package. Imports nothing else in this repo. `python -m ddim.main` is the
   legacy entry point, retained for sampling/FID/interpolation and upstream
   compatibility; it emits a deprecation warning for training.
2. **`runctl/`** — the supported, **flow-agnostic** orchestrator for reproducible
   training: run bundles, dataset fingerprinting, source snapshotting, executors,
   checkpoints, metrics, machine profiles, optional MLflow publication. It knows
   nothing about any model.
3. **`burst_diffusion/`** — the burst-averaging diffusion denoiser. Own U-Net,
   EMA, trainer, sampler, and CLI. Depends only on `noising_pipeline`.
4. **`edge_denoise/`** — deterministic single-pass denoisers aimed at metrology
   *precision* (Noise2Noise, pure Sobel-gradient domain, hybrid). Deliberately
   imports `burst_diffusion` for the audited `BurstCache` content-group split,
   the U-Net backbone (equal-capacity comparisons), metrics, and the
   repeatability harness; never imports `ddim` or `runctl`. Math and
   feasibility study: `edge_denoise/docs/edge_denoise_method.md`; the
   fine-feature / burst-mean-target / diffusion-prior study and the current
   best arms: `edge_denoise/docs/fine_feature_report.md`.
5. **`noising_pipeline/`** — standalone paired clean/noisy image generator.
   Depends on nothing in this repo.

`docs/workflows.md` is the entry point for running anything.
`runctl/docs/training_workflow.md` is the authoritative spec for run bundles,
machine profiles and executors. `ddim/docs/sem_dataset_migration.md` explains the
SEM loader and a DataLoader-worker perf pitfall (see Gotchas).

## Commands

```
python -m pip install -e ".[dev]"        # package + runctl entry point + pytest
python -m pip install -e ".[tracking]"   # MLflow/psutil/nvidia-ml-py (tracking host only)

python -m pytest                         # full suite (testpaths = tests/)
python -m pytest tests/runctl -q         # one area
python -m pytest tests/runctl/test_runctl_cli.py::test_canonical_command_exposes_every_varying_setting

runctl doctor --machine <id> --flow ddim --exercise-executor
runctl train wizard --machine <id> --flow ddim
runctl train plan   --machine <id> --flow ddim ...   # prints canonical command, writes nothing
runctl run status|logs|stop|resume <run_dir>
runctl track serve | track publish <run_dir>

python -m burst_diffusion train --config burst_diffusion/configs/<name>.yml
python -m edge_denoise train --config edge_denoise/configs/<name>.yml
python -m edge_denoise repeatability --config <cfg> --checkpoint a=<pt> --burst-checkpoint b=<pt> ...
python -m edge_denoise fine-features --config <cfg> --checkpoint a=<pt> --out <dir>   # per-band gain / blemish retention
python -m edge_denoise train-prior --config edge_denoise/configs/miic_p10_dedup_prior.yml  # DDPM prior; --prior-checkpoint --posterior-arm on the two commands above
python -m ddim.main --config <name>.yml --exp <path> --doc <name> --ni
```

The environment is the re-pointed venv at `E:\PythonProjects\ddim\.venv`
(Python 3.13, torch 2.13+cu126). Its editable install resolves `ddim`,
`runctl`, `burst_diffusion` and `noising_pipeline` to **this** repo; the old
`E:\PythonProjects\ddim` checkout is no longer importable through it.

There is no configured linter/formatter — match the style of nearby code.

## Architecture: the flow boundary

This is the load-bearing design decision. `runctl` is generic; a *flow* supplies
the model-specific half:

- `runctl/schemas.py` — `BaseTrainingSpec` holds only what orchestration needs
  (label, dataset alias, dataset scan rules, batch/steps/intervals, lr, seed,
  reproducibility). `MachineProfile`, `RunManifest`, `DatasetFingerprint`,
  `SourceSnapshot`, `AttemptState` are fully generic. `RunManifest.flow` names
  the flow; a `field_validator("training", mode="before")` re-types the nested
  payload with that flow's spec model on load.
- `runctl/registry.py` — `Flow` (spec model, config parser, trainer, options,
  plan rows) resolved via the `runctl.flows` entry-point group, with
  `BUILTIN_FLOWS` as a source-checkout fallback.
- `runctl/control.py` — flow-agnostic training primitives: `configure_reproducibility`,
  `select_device`, `EpochSeededDataset`, `ResumableEpochSampler`, `StopController`,
  `TrainingResult`. New flows compose these; they must not reimplement them.
- `runctl/bundles.py` — run bundles plus *reusable* strict config machinery
  (`flatten_config`, `load_config_mapping`, `map_config_keys`, `_UniqueKeyLoader`).
  It no longer contains any DDIM YAML knowledge.
- `runctl/worker.py` — scheduler-safe entry point. Verifies the dataset
  fingerprint, owns the attempt lifecycle and exit-code contract, then calls
  `get_flow(manifest.flow).trainer(...)`. **Adding a pipeline never touches this file.**
- `ddim/flow.py` — the *only* place the orchestrator and DDIM research code meet:
  `CONFIG_KEY_MAP`, `parse_config`, `DdimTrainingSpec`, the lazily imported trainer.
- `ddim/spec.py`, `ddim/training.py` — DDIM's spec and manifest-driven trainer.

**When adding a setting, put it on the flow's spec, not `BaseTrainingSpec`,
unless every conceivable flow needs it.** When adding a pipeline, write a
`Flow` and register an entry point; do not edit `runctl`.

### CLI option tiers

`runctl train` exposes shared settings as typed flags and flow-specific settings
as repeatable `--set NAME=VALUE` (YAML-typed). This is deliberate: Typer
registers commands at import time, but the valid options depend on the runtime
`--flow`. Unknown names, repeated names and wrong types are still rejected, so
the "no silently misspelled settings" invariant holds. `_canonical_argv`
restates *every* varying setting in both tiers — a flow must never let a knob
escape the reproducible command.

## Architecture: legacy DDIM code

`ddim/main.py` parses args/config (`dict2namespace`, `resolve_config_path`), then
hands off to `ddim/runners/diffusion.py`'s `Diffusion` class (`train`, `sample`,
`test`). `ddim/models/diffusion.py` is the U-Net; `ddim/functions/` holds the beta
schedule, denoising step and loss. `ddim/datasets/__init__.py::get_dataset` is the
single dispatch point for all datasets (`CIFAR10`, `CELEBA`, `LSUN`, `FFHQ`, `SEM`)
— adding one means a branch there plus a loader module.

`ddim/datasets/sem.py` (`SEMImageDataset`) loads SEM PNGs from `data.data_dir`
with a deterministic seeded train/val split. `ddim/training.py` reuses this same
dataset/model code — it does not reimplement the diffusion maths, only the
orchestration around it.

`resolve_config_path` looks in `ddim/configs/` first, so `--config sem.yml` works
from any working directory; absolute and cwd-relative paths still win.

## Architecture: cross-pipeline evaluation

`burst_diffusion/repeatability.py` owns the CD/registration precision harness.
Other pipelines join its comparison table through
`RealizationProvider` (`extra_providers=` on `repeatability()`) instead of
growing parallel evaluators — `edge_denoise/provider.py` is the worked example,
and `python -m edge_denoise repeatability` produces one table holding classical,
burst, and edge arms measured on identical sources, seeds, crops, and CD sites.
All burst arms in one call must share `schedule.num_steps`. The dev split for
MIIC experiments is `val`; the `test` split is locked (report once, only for a
frozen method).

## Gotchas

- **Duplicate source content = silent validation leakage.** Real image corpora
  ship the same picture under several filenames — the MIIC corpus had **185
  unique images among 1050 files**, with nominal val/test directories overlapping
  train completely. Splitting by *filename/index* therefore puts byte-identical
  content on both sides and quietly invalidates every held-out number; this
  happened and cost a full retrain. `burst_diffusion` deduplicates by content
  hash at generation (`select_unique_sources`) and splits by content group
  (`BurstCache`), with regression tests. **Any new dataset path must hash content
  before splitting** — the legacy `ddim/datasets/sem.py` + `get_dataset` index
  split does *not* and will leak on that corpus. See
  `burst_diffusion/docs/burst_diffusion_audit_handoff.md`.
- **DataLoader worker restart trap** (`ddim/docs/sem_dataset_migration.md`): the
  legacy runner creates a new DataLoader iterator every epoch. On a small dataset
  with `num_workers > 0` and no `persistent_workers`, workers are torn down and
  recreated almost every step — measured 3.3s/batch vs 17ms/batch. Fix is
  `num_workers: 0` plus `cache_in_memory: true`, not adding persistent-workers
  support. `cache_in_memory` cannot combine with `random_flip` (the transform
  would be cached pre-augmentation).
- **The legacy runner's stop condition is a step budget, not epochs.**
  `ddim/runners/diffusion.py` reads `training.max_steps`, falling back to
  `training.n_iters`; if either is present it iterates epochs with
  `itertools.count()` (unbounded) and stops on the step budget, so `n_epochs` is
  ignored. Only a config with *neither* key falls back to
  `range(start_epoch, n_epochs)`. Verified: `sem_ddim_local_32.yml` has
  `n_iters: 20000`, and `--set training.n_epochs=2` had no effect on it. Set
  `--max-steps` (not `n_epochs`) to bound a run. Note `snapshot_freq` is a
  *step* interval and each SEM checkpoint is ~144 MB — a small `snapshot_freq`
  over a long budget fills a disk fast.
- Multi-GPU: `runctl` isolates one GPU per run via `gpu_index` before the worker
  imports PyTorch (single-GPU-per-run by design). The legacy `ddim.main` path
  instead wraps the model in `DataParallel` across every visible GPU — set
  `CUDA_VISIBLE_DEVICES` explicitly when benchmarking one GPU through it.
- A pydantic `model_validator(mode="before")` on `RunManifest` **breaks JSON
  input mode for the whole model** (ISO datetimes and JSON arrays stop validating
  under `strict=True`). The spec coercion is a `field_validator` on `training`
  for exactly this reason — do not "simplify" it back.
- Never embed credentials in an MLflow tracking URI or machine profile —
  validation rejects them; supply auth at publish time via the environment.
- Machine profiles, absolute local paths, and generated artifacts (`runs/`,
  `experiments/`, `output/`, `data/`, `tmp/`, checkpoints) must never be
  committed — see `.gitignore`.

## Coding style & conventions

Python 3.10+, four-space indent, PEP 8 (`snake_case` functions/modules,
`PascalCase` classes/Pydantic models, `UPPER_SNAKE_CASE` constants). Type new
public APIs; keep filesystem/process code cross-platform (developed on Windows,
must also run on Linux HPC). Group imports stdlib / third-party / local.

Tests live in `tests/<package>/test_<area>.py`, functions named `test_*`, using
`tmp_path`/`monkeypatch`/mocks for filesystem, scheduler, network and GPU
behavior — never hit a real network/GPU/scheduler. Every bug fix gets a
regression test.

Commits favor concise imperative subjects with `<type>: <summary>` prefixes
(`feat:`, `fix:`, `docs:`). PRs should explain problem + solution, list
verification commands, and call out config/compatibility changes; never attach
datasets, credentials, checkpoints, or run bundles.
