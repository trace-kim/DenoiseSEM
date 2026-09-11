# DenoiseSEM

Research code for denoising scanning-electron-microscope (SEM) and microscopy
images with diffusion methods.

**Start here: [`docs/workflows.md`](docs/workflows.md)** — how to set up the
environment and run every workflow from this directory.

## Layout

Six independent pieces of work, separated by ownership rather than sharing one
folder:

```
ddim/                 Original DDIM (Song, Meng & Ermon), namespaced as a package
  configs/            YAML recipes (sem.yml, cifar10.yml, ...)
  docs/               SEM dataset loader notes, flow diagram
  flow.py             Registers the "ddim" flow with runctl  <- the only seam
  spec.py             DdimTrainingSpec
  training.py         Manifest-driven DDIM trainer

burst_diffusion/      Burst-averaging diffusion denoiser (own U-Net + CLI)
  configs/            Experiment recipes
  docs/               Method, guide, report, audit handoff, research idea

edge_denoise/         Edge-preserving deterministic denoisers for metrology (own CLI)
  configs/            Experiment recipes (N2N, gradient, hybrid arms)
  docs/               Feasibility study & method derivations, experiment report

noising_pipeline/     Standalone paired clean/noisy image generator

sem_noise/            Real repeated-SEM noise, registration, and stability analysis
  configs/            Analysis settings (original detector units)

runctl/               Flow-agnostic reproducible-run orchestrator
  docs/               Training workflow spec + EN/KR quick start
  registry.py         Flow plugin lookup
  control.py          Shared training primitives (seeding, device, stop, ordering)

tests/                Mirrors the package split
tools/                Ad-hoc benchmarking and probe scripts
docs/workflows.md     How to run everything
```

Dependencies point one way only:

```
noising_pipeline  <-  burst_diffusion  <-  edge_denoise

runctl  <-  ddim/flow.py  (loaded lazily, via a `runctl.flows` entry point)
```

`ddim` imports nothing else in this repository. `runctl` imports no model code
until you ask it for a flow, so `runctl --help` never imports PyTorch.

## The four ways to train

For real repeated acquisitions, start with the
[SEM noise analysis guide](sem_noise/README.md). It produces offline reports for
noise distributions, signal dependence, drift, spatial structure, and temporal
stability before choosing training pairs or targets.

| | Command | Use it for |
|---|---|---|
| Reproducible | `runctl train wizard --machine <id> --flow ddim` | New DDIM work: immutable run bundles, executors, resume, tracking |
| Legacy | `python -m ddim.main --config <name>.yml --exp ... --doc ... --ni` | Sampling, FID, interpolation, upstream compatibility |
| Burst | `python -m burst_diffusion train --config burst_diffusion/configs/<name>.yml` | The burst-averaging pipeline |
| Edge | `python -m edge_denoise train --config edge_denoise/configs/<name>.yml` | Metrology-precision denoisers (N2N / gradient / hybrid) |

## Quick start

```powershell
E:\PythonProjects\ddim\.venv\Scripts\Activate.ps1
python -m pytest
runctl --help
```

See [`docs/workflows.md`](docs/workflows.md) for the environment details,
including why that venv path is used and how to rebuild it elsewhere.

## Relationship to upstream DDIM

`ddim/` is the [official DDIM implementation](https://github.com/ermongroup/ddim)
by Jiaming Song, Chenlin Meng and Stefano Ermon. The algorithms are unchanged;
only import paths were namespaced (`models.diffusion` → `ddim.models.diffusion`)
so the pipelines can coexist. Attribution, citation and the original usage
documentation are preserved in [`ddim/README.md`](ddim/README.md).

## License

MIT — see [`LICENSE`](LICENSE).
