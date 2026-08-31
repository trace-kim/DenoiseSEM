# SEM directory loader and workstation speed check

## What changed

- `datasets/sem.py` loads SEM files directly from a configured directory.
- `datasets/__init__.py` accepts `data.dataset: "SEM"` and performs a deterministic train/validation split.
- `tools/benchmark_sem_loader.py` measures file loading, resize, batching, and optional GPU transfer without model compute.
- `ddim/configs/sem_directory_benchmark.yml` is a one-epoch integration benchmark.

`runners/diffusion.py` remains identical to the original repository version.

Set the YAML path with forward slashes, even on Windows:

```yaml
data:
    dataset: "SEM"
    data_dir: "E:/path/to/target_sem"
    image_size: 32
    channels: 1
    num_workers: 0
    cache_in_memory: true
    validation_split: 0.1
```

## Why a faster GPU can appear much slower

The runner creates a new DataLoader iterator for every epoch. A tiny overfit dataset can contain exactly one batch per epoch. With multiple workers and `persistent_workers: false`, the workers can therefore be shut down and recreated for every optimizer step.

This failure mode was reproduced locally with seven training images and batch size seven:

| Loader setting | Mean batch time |
|---|---:|
| 4 workers, persistent workers disabled | 3,365 ms |
| 4 workers, persistent workers enabled | 16.9 ms |
| 0 workers | 14.7 ms |

The original diffusion runner does not expose `persistent_workers`, so the compatible fix is `num_workers: 0`. The SEM dataset can cache resized tensors with `cache_in_memory: true`, which removes repeated PNG decoding without changing the runner.

## Local measured result

Hardware: NVIDIA GeForce RTX 4060 Ti. Data: 1,050 original 512 x 512 SEM PNGs, split into 945 training and 105 validation images. Model: 32 x 32 grayscale, ch 64, batch seven.

- 135 training updates completed successfully with the original runner.
- Warm steady-state mean: 55.35 ms/step.
- Warm steady-state throughput: 18.07 steps/s.
- Median loss: 164.88 over the first 10 steps, 26.92 over the final 20 steps.
- Total process time: 17.1 seconds, including dataset caching and two large checkpoint writes.
- Checkpoint: `experiments/sem_directory_original_runner_benchmark/logs/cached_workers0_v1/ckpt_135.pth`.

Do not use step 1 or total process wall time as steady-state throughput. Step 1 includes worker startup, and steps 1 and 80 write large checkpoints.

## Workstation commands

Run the configured cached loader without model compute:

```bash
python tools/benchmark_sem_loader.py --config sem_directory_benchmark.yml --workers 0 --warmup-batches 10 --batches 100
```

The utility can separately reproduce the worker-restart problem, but these worker options are intentionally not added to the original diffusion runner:

```bash
python tools/benchmark_sem_loader.py --config sem_directory_benchmark.yml --workers 4 --no-persistent-workers --warmup-batches 10 --batches 100
```

Run the actual fixed-step training benchmark:

```bash
python -m ddim.main --config sem_directory_benchmark.yml --exp experiments/sem_directory_original_runner_benchmark --doc a6000_single_gpu --ni
```

The repository wraps the model in `torch.nn.DataParallel` and uses every visible GPU. A global batch of seven spread over four A6000 GPUs is a poor scaling test. First benchmark one A6000.

Linux:

```bash
CUDA_VISIBLE_DEVICES=0 python -m ddim.main --config sem_directory_benchmark.yml --exp experiments/sem_directory_original_runner_benchmark --doc a6000_single_gpu --ni
```

PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES='0'; python -m ddim.main --config sem_directory_benchmark.yml --exp experiments\sem_directory_original_runner_benchmark --doc a6000_single_gpu --ni
```

The original runner ignores `training.n_iters`; it is the number of epochs that determines when training stops. With 945 training images and batch size seven, one epoch is exactly 135 updates. Recalculate `n_epochs` when the remote dataset size or desired update count changes.

If the loader benchmark is fast but training remains slow, capture GPU state while training:

```bash
nvidia-smi --query-gpu=name,driver_version,pstate,power.draw,power.limit,clocks.current.sm,clocks.current.memory,memory.used,utilization.gpu --format=csv -l 1
```

Low utilization with a fast loader points to process launch, DataParallel, CPU scheduling, or synchronization overhead. Sustained high utilization with low clocks points to power, thermal, driver, or GPU configuration issues.
