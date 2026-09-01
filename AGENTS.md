# Repository Guidelines

## Project Structure & Module Organization

Five independently owned packages. `ddim/` is the original DDIM implementation
(models, runners, functions, datasets, configs, legacy `main.py`) namespaced as a
package; it imports nothing else here. `runctl/` is the flow-agnostic
orchestrator — schemas, run bundles, executors, checkpoints, tracking, and the
flow registry — and knows nothing about any model. `burst_diffusion/` is the
self-contained burst-averaging denoiser, depending only on `noising_pipeline/`,
the standalone paired clean/noisy image generator. `edge_denoise/` holds the
deterministic metrology-precision denoisers (N2N, gradient, hybrid) and imports
`burst_diffusion` for the dataset cache, U-Net backbone, and evaluation
harness. YAML recipes live inside the package that owns them (`ddim/configs/`,
`burst_diffusion/configs/`, `edge_denoise/configs/`), and docs likewise
(`ddim/docs/`, `runctl/docs/`, `burst_diffusion/docs/`, `edge_denoise/docs/`),
with `docs/workflows.md` at the root as the cross-cutting how-to-run guide. Tests
mirror the split under `tests/<package>/`; utilities are in `tools/`. Treat
`runs/`, `experiments/`, `output/`, `data/`, `tmp/`, and checkpoints as generated
artifacts.

## Build, Test, and Development Commands

- `python -m pip install -e ".[dev]"` installs the package, the `runctl` entry
  point, the `ddim` flow plugin, and pytest.
- `python -m pytest` runs the complete suite configured in `pyproject.toml`.
- `python -m pytest tests/runctl -q` runs one area during development.
- `runctl doctor --machine <id> --flow ddim --exercise-executor` validates a
  training target without launching.
- `runctl train wizard --machine <id> --flow ddim` plans and launches the
  supported workflow; `runctl train plan ...` prints the canonical command and
  writes nothing.
- `python -m burst_diffusion train --config burst_diffusion/configs/<name>.yml`
  runs the burst pipeline.
- `python -m edge_denoise train --config edge_denoise/configs/<name>.yml` runs
  the edge/metrology pipeline (same burst datasets).
- `python -m ddim.main ...` is retained only for legacy compatibility and
  sampling.

The environment is the re-pointed venv at `E:\PythonProjects\ddim\.venv`; see
`docs/workflows.md`.

## Coding Style & Naming Conventions

Python 3.10+ and four-space indentation. PEP 8: `snake_case` for modules,
functions, fixtures, and variables; `PascalCase` for classes and Pydantic models;
`UPPER_SNAKE_CASE` for constants. Type new public APIs and keep
filesystem/process behavior cross-platform (developed on Windows, must run on
Linux HPC). No formatter or linter is enforced, so match nearby code and group
imports as standard library, third-party, then local.

## Architectural Rules

Respect the flow boundary. `runctl` must not import model code; a pipeline plugs
in by declaring a `Flow` (spec model, config parser, trainer) and registering a
`runctl.flows` entry point — `ddim/flow.py` is the worked example and the only
file where the two halves meet. Put new settings on the flow's spec, not on
`BaseTrainingSpec`, unless every flow needs them. Compose the shared primitives
in `runctl/control.py` rather than reimplementing seeding, device selection, or
stop handling. Any new dataset path must hash content before splitting.

## Testing Guidelines

Tests use pytest, named `tests/<package>/test_<area>.py` with functions beginning
`test_`. Use `tmp_path`, `monkeypatch`, and mocks for filesystem, scheduler,
network, or GPU behavior — never touch a real one. Cover both successful behavior
and validation/error paths. There is no coverage threshold, but every bug fix
should include a regression test.

## Commit & Pull Request Guidelines

Use concise, imperative subjects with `<type>: <summary>` prefixes (for example,
`fix: reject duplicate training options`) and keep commits focused. Pull requests
should explain the problem and solution, list verification commands, link issues,
and call out config or compatibility changes. Include logs or screenshots only
when visible output changes; never attach datasets, credentials, checkpoints, or
run bundles.

## Security & Configuration Tips

Keep machine profiles outside the repository as documented in
`runctl/docs/training_workflow.md`. Do not commit secrets, private package-index
credentials, absolute local paths, or generated training artifacts. Review
explicit launch plans before adding `--yes`, especially for Slurm or external HPC
execution.
