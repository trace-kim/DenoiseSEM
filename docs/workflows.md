# Running every workflow from this repository root

This repository holds five independent pieces of work that used to share one
folder. They are now separated by ownership, and **every command below is run
from the repository root** (`E:\PythonProjects\DenoiseSEM`).

| Package | What it is | Depends on |
|---|---|---|
| `ddim/` | The original DDIM implementation (Song, Meng & Ermon), namespaced as a package | nothing in this repo |
| `noising_pipeline/` | Standalone generator of paired clean/noisy microscopy images | nothing in this repo |
| `burst_diffusion/` | Burst-averaging diffusion denoiser (own U-Net, trainer, sampler, CLI) | `noising_pipeline` |
| `edge_denoise/` | Edge-preserving deterministic denoisers for metrology precision (N2N, gradient, hybrid) | `burst_diffusion` |
| `runctl/` | Flow-agnostic reproducible-run orchestrator (bundles, executors, tracking) | a *flow* plugin, loaded lazily |

The dependency arrows only ever point one way. `ddim` does not import
`runctl`; `runctl` does not import `ddim` until you ask it for the `ddim` flow.

---

## 1. Environment

The project uses the existing virtual environment, re-pointed at this
repository:

```
E:\PythonProjects\ddim\.venv        Python 3.13.3, torch 2.13.0+cu126, CUDA available
```

It was re-pointed with:

```powershell
E:\PythonProjects\ddim\.venv\Scripts\python.exe -m pip uninstall -y ddim-training-workflow
E:\PythonProjects\ddim\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

> **Consequence, by design.** That venv's editable install now resolves
> `ddim`, `runctl`, `burst_diffusion` and `noising_pipeline` to
> **this** repository. The old `E:\PythonProjects\ddim` checkout is no longer
> importable through it. That was the intended trade for not copying 4.9 GB.

Activate it once per shell:

```powershell
E:\PythonProjects\ddim\.venv\Scripts\Activate.ps1
```

Or call the interpreter directly (no activation needed):

```powershell
$py = "E:\PythonProjects\ddim\.venv\Scripts\python.exe"
& $py -m pytest
```

### Verify the environment

```powershell
python -c "import ddim, runctl, burst_diffusion, noising_pipeline; print(ddim.__file__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
runctl --help
```

`ddim.__file__` must print a path under `DenoiseSEM`. If it prints one under
the old `ddim` project, the editable install is stale — rerun the
`pip install -e .` above.

### Optional extras

```powershell
python -m pip install -e ".[dev]"        # pytest
python -m pip install -e ".[tracking]"   # MLflow, psutil, nvidia-ml-py
```

### Rebuilding the environment elsewhere

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Install a CUDA-enabled PyTorch build matching the target machine's driver
*before* installing this project, so pip does not pull the CPU-only wheel.

---

## 2. Tests

```powershell
python -m pytest                          # whole suite (257 tests)
python -m pytest tests/runctl -q          # orchestration only
python -m pytest tests/ddim -q            # DDIM flow, spec, legacy entry points
python -m pytest tests/burst_diffusion -q # burst pipeline
python -m pytest tests/noising_pipeline -q
python -m pytest tests/runctl/test_runctl_cli.py::test_canonical_command_exposes_every_varying_setting
```

Tests never touch a real GPU, scheduler, or network.

---

## 3. Workflow A — `noising_pipeline`: build paired clean/noisy data

A library, not a CLI. One public function, `create_noisy_dataset`.

```powershell
python -c @'
from noising_pipeline import create_noisy_dataset
manifest = create_noisy_dataset(
    source_dir="data/BBBC038v1",
    output_dir="data/BBBC038-noisy",
    n=100, steps=1000, noise_type="poisson",
    download="bbbc038",
)
print(manifest)
'@
```

`download="bbbc038"` fetches the BBBC038v1 stage-one set (670 CC0 microscopy
images) into `source_dir` if it is not already there. Drop it to use your own
clean images. Output is `clean/`, `noisy/`, and a `manifest.jsonl`.

Full option reference: [`noising_pipeline/README.md`](../noising_pipeline/README.md).

---

## 4. Workflow B — `burst_diffusion`: burst-averaging denoiser

Self-contained pipeline with its own CLI. Recipes live in
`burst_diffusion/configs/`; they use paths relative to the repository root, so
run these from the root.

### Quickest end-to-end path (CPU smoke, seconds)

```powershell
python -m burst_diffusion generate --source-dir data/BBBC038v1 `
  --output-dir data/BBBC038-burst-smoke --num-sources 6 --replicas 6 `
  --noise-type poisson --peak 10 --margin 0.15 --max-side 64 --overwrite

python -m burst_diffusion train --config burst_diffusion/configs/smoke.yml
```

### Full sequence

```powershell
# 1. Generate a burst dataset (deduplicates sources by content hash)
python -m burst_diffusion generate --source-dir data/BBBC038v1 `
  --output-dir data/BBBC038-burst-p10 --num-sources 96 --replicas 16 `
  --noise-type poisson --peak 10 --margin 0.15

# 2. Look at it before training on it
python -m burst_diffusion preview --dataset data/BBBC038-burst-p10 --out tmp/preview.png

# 3. Train
python -m burst_diffusion train --config burst_diffusion/configs/bbbc038_p10.yml

# 4. Denoise a single measurement
python -m burst_diffusion sample `
  --checkpoint runs/burst_diffusion/bbbc038_p10/ckpt_latest.pt `
  --dataset data/BBBC038-burst-p10 --source-index 0 --out tmp/sample

# 5. Compare against the classical baselines on held-out sources
python -m burst_diffusion evaluate --config burst_diffusion/configs/bbbc038_p10.yml `
  --checkpoint runs/burst_diffusion/bbbc038_p10/ckpt_latest.pt --out tmp/eval

# 6. Metrology + provenance
python -m burst_diffusion repeatability --config burst_diffusion/configs/bbbc038_p10.yml `
  --checkpoint runs/burst_diffusion/bbbc038_p10/ckpt_latest.pt --out tmp/repeat
python -m burst_diffusion provenance --config burst_diffusion/configs/bbbc038_p10.yml `
  --command "python -m burst_diffusion train --config burst_diffusion/configs/bbbc038_p10.yml"
```

`--resume` continues from the run directory's latest checkpoint.

Docs: [user guide](../burst_diffusion/docs/burst_diffusion_guide.md) ·
[method](../burst_diffusion/docs/burst_diffusion_method.md) ·
[report](../burst_diffusion/docs/burst_diffusion_report.md) ·
[audit handoff](../burst_diffusion/docs/burst_diffusion_audit_handoff.md) ·
[research idea](../burst_diffusion/docs/research_idea.md)

> Read the audit handoff before quoting any number from the report — it records
> a validation-leakage defect and which claims it invalidates.

---

## 4b. Workflow B2 — `edge_denoise`: edge-preserving metrology denoisers

Deterministic single-pass denoisers aimed at metrology *precision* (CD and
registration repeatability), not just PSNR: plain Noise2Noise (`image`), the
pure Sobel-gradient-domain variant (`gradient`, with exact FFT least-squares
image recovery), and the hybrid (image + Sobel input channels, image output,
gradient-weighted loss). Trains on the burst datasets from Workflow B —
generate data there; the U-Net capacity and the content-group split are shared
with `burst_diffusion`, so cross-pipeline comparisons are exact.

```powershell
# CPU smoke (reuses the burst smoke dataset from Workflow B)
python -m edge_denoise train --config edge_denoise/configs/smoke.yml

# Real arms on the deduplicated MIIC SEM dataset
python -m edge_denoise train --config edge_denoise/configs/miic_p10_dedup_hybrid.yml
python -m edge_denoise train --config edge_denoise/configs/miic_p10_dedup_grad.yml
python -m edge_denoise train --config edge_denoise/configs/miic_p10_dedup_n2n.yml

# Denoise one measurement (single deterministic forward pass)
python -m edge_denoise denoise `
  --checkpoint runs/edge_denoise/miic_p10_dedup_hybrid/ckpt_latest.pt `
  --dataset data/MIIC-burst-p10-dedup --source-index 0 --out tmp/denoise

# Accuracy vs classical baselines (means AND medians)
python -m edge_denoise evaluate --config edge_denoise/configs/miic_p10_dedup_hybrid.yml `
  --checkpoint runs/edge_denoise/miic_p10_dedup_hybrid/ckpt_latest.pt --out tmp/eval

# ONE metrology-precision table: classical ladder + edge arms + burst/N2N arms
python -m edge_denoise repeatability --config edge_denoise/configs/miic_p10_dedup_hybrid.yml `
  --checkpoint hybrid=runs/edge_denoise/miic_p10_dedup_hybrid/ckpt_latest.pt `
  --checkpoint grad=runs/edge_denoise/miic_p10_dedup_grad/ckpt_latest.pt `
  --burst-checkpoint n2n=runs/burst_diffusion/miic_p10_dedup_n2n/ckpt_latest.pt `
  --out tmp/repeat --split val
```

Training writes `provenance.json` automatically at completion; `--resume`
continues from the run directory's latest checkpoint. All burst arms passed to
one `repeatability` call must share one `schedule.num_steps` — run arms with
different schedules separately.

Docs: [method & feasibility study](../edge_denoise/docs/edge_denoise_method.md) ·
[experiment report](../edge_denoise/docs/edge_denoise_report.md)

---

## 5. Workflow C — `ddim`: the original DDIM entry point

Legacy, retained for sampling and compatibility. It prints a deprecation
warning; use `runctl` (Workflow D) for new training.

Recipes are packaged in `ddim/configs/`, and `--config` accepts a bare
filename, so this works from anywhere:

```powershell
# Train
python -m ddim.main --config sem_ddim_local_32.yml --exp experiments/sem32 --doc run1 --ni

# Sample for FID
python -m ddim.main --config sem_ddim_local_32.yml --exp experiments/sem32 --doc run1 `
  --sample --fid --timesteps 100 --eta 0 --ni

# Sample the denoising sequence / interpolation
python -m ddim.main --config cifar10.yml --exp experiments/c10 --doc run1 --sample --sequence --ni
python -m ddim.main --config cifar10.yml --exp experiments/c10 --doc run1 --sample --interpolation --ni
```

Override YAML values without editing files — convenience flags plus repeatable
`--set SECTION.KEY=VALUE` (YAML-typed; unknown keys and bad types are rejected):

```powershell
python -m ddim.main --config sem_ddim_local_32.yml --exp experiments/sem64 --doc run1 `
  --image-size 64 --batch-size 4 --learning-rate 0.0001 --max-steps 20000 `
  --set model.dropout=0.1 --set "model.ch_mult=[1, 2, 2, 4]" --ni
```

Resolved config is written to `<exp>/logs/<doc>/config.yml`.

> **Bounding a legacy run.** The runner stops on a *step budget*, not epochs.
> If `training.max_steps` or `training.n_iters` is set (every shipped SEM recipe
> sets `n_iters`), it iterates epochs unbounded and `n_epochs` is ignored — so
> `--set training.n_epochs=2` will *not* shorten a run. Use `--max-steps`.
> Also note `training.snapshot_freq` is a step interval and each SEM checkpoint
> is ~144 MB; a small `snapshot_freq` over a long budget will fill a disk.


Multi-GPU: this path wraps the model in `DataParallel` across every visible
GPU. Set `CUDA_VISIBLE_DEVICES` explicitly to pin one.

Docs: [`ddim/README.md`](../ddim/README.md) ·
[SEM dataset loader + a DataLoader perf trap](../ddim/docs/sem_dataset_migration.md)

---

## 6. Workflow D — `runctl`: reproducible, resumable training

`runctl` is the supported way to run new training. It separates:

- **machine facts** (executor, GPU index, dataset paths, runs root) → a
  *machine profile* stored **outside** the repo
  (`%APPDATA%\runctl\profiles` on Windows, `$XDG_CONFIG_HOME/runctl/profiles`
  on Linux), and
- **experiment recipe** (batch size, LR, steps, model geometry) → a versioned
  YAML plus explicit command-line overrides.

Every run becomes an immutable **run bundle** under
`<runs-root>/YYYY-MM-DD/<timestamp>__<label>__<hash>/`, containing the
manifest, a source snapshot, dataset fingerprint, metrics, checkpoints, and
logs. That bundle is the source of truth; TensorBoard and MLflow are views.

### 6.1 One-time machine setup

```powershell
runctl machine configure --id local-4060ti
runctl machine list
runctl machine show local-4060ti
```

### 6.2 Preflight

```powershell
runctl doctor --machine local-4060ti --flow ddim --exercise-executor
```

Checks the interpreter, dataset directories, runs directory writability, the
selected GPU, a dataset scan, free disk against the estimated checkpoint size,
and the executor itself.

### 6.3 Plan and launch

Interactive:

```powershell
runctl train wizard --machine local-4060ti --flow ddim
```

Non-interactive — `plan` writes nothing and prints the exact reproducing
command:

```powershell
runctl train plan --machine local-4060ti --flow ddim `
  --label sem32-baseline --dataset sem `
  --max-steps 20000 --batch-size 7 --learning-rate 0.0002 `
  --checkpoint-every 2500 --validation-every 2500 --sample-every 2500 `
  --seed 1234 --reproducibility seeded --num-workers 0 --no-cache-in-memory `
  --set image_size=32 --set model_ch=64 --set "ch_mult=[1,2,2,2]" `
  --set diffusion_steps=100 --set beta_start=0.001 --set beta_end=0.2 `
  --set ema_rate=0.999
```

Then run the printed `runctl train launch ... --yes` command verbatim.

**Two tiers of options, on purpose:**

- Settings every flow has are **typed flags** (`--label`, `--batch-size`,
  `--max-steps`, `--seed`, `--reproducibility`, …).
- Settings specific to one flow are **`--set NAME=VALUE`**, repeatable, with
  YAML-typed values. Unknown names, repeated names, and wrong types are
  rejected — nothing is silently ignored.

`--config` defaults to the flow's single active recipe
(`ddim/configs/sem.yml`), and passing a different file is refused: keep fixed
choices in the recipe and varying choices on the command line.

### 6.4 Monitor and control

```powershell
runctl run status <run_dir>
runctl run logs   <run_dir> --lines 120
runctl run logs   <run_dir> --stream stdout --follow
runctl run stop   <run_dir>            # graceful: checkpoints, then exits
runctl run resume <run_dir>            # new attempts/NNN/, needs checkpoints/latest.json
```

There are no automatic retries by design — a failure stays visible in
`state.json` for a human to inspect before an explicit `resume`.

### 6.5 Tracking (optional, local)

```powershell
runctl track serve --port 5000
runctl track publish <run_dir> --tracking-uri http://127.0.0.1:5000 --experiment ddim-sem
```

Training never depends on MLflow. Never put credentials in a tracking URI —
schema validation rejects them; supply auth through the environment.

### 6.6 Air-gapped / HPC targets

```powershell
runctl environment bundle --output h100-wheelhouse --yes   # on a matching connected builder
runctl environment verify h100-wheelhouse
python -m runctl.hpc_probe --output h100-compute-probe.json
```

Full detail: [`runctl/docs/training_workflow.md`](../runctl/docs/training_workflow.md) ·
quick start EN/KR: [`runctl/docs/training_guide.md`](../runctl/docs/training_guide.md)

---

## 7. Utilities

```powershell
python tools/benchmark_sem_loader.py --config sem_directory_benchmark.yml
python tools/probe_burst_predictions.py --help
python tools/edge_denoise_report_figures.py --help   # regenerate the edge_denoise report figures
```

---

## 8. Adding a new flow to `runctl`

`runctl` knows nothing about DDIM. A pipeline plugs in by declaring a `Flow`:

1. Subclass `runctl.schemas.BaseTrainingSpec` with your settings.
2. Write `parse_config(path) -> dict` mapping your YAML onto those field names
   (reuse `runctl.bundles.flatten_config` / `map_config_keys` to inherit the
   strict unknown-key and duplicate-target checks).
3. Write a trainer taking `(manifest, run_dir, *, resume, stop_controller,
   metric_logger, progress_callback, device)`. Compose the shared primitives in
   `runctl.control` (seeding, device selection, epoch-deterministic ordering,
   resumable sampling, stop requests) rather than reimplementing them.
4. Expose `FLOW = Flow(...)` in a module and register it:

```toml
[project.entry-points."runctl.flows"]
myflow = "mypackage.flow"
```

`ddim/flow.py` is the worked example, and is the *only* file where the
orchestrator and the DDIM research code meet.

Then everything else — bundles, executors, resume, status, tracking, the
wizard, the canonical command — works for the new flow with no changes to
`runctl`.

---

## 9. Generated artifacts

`runs/`, `experiments/`, `output/`, `data/`, `tmp/`, checkpoints, and machine
profiles are generated and git-ignored. Machine profiles deliberately live
outside the repository. Never commit datasets, checkpoints, credentials, or
run bundles.
