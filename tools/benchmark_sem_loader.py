import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddim.datasets import get_dataset
from ddim.main import dict2namespace


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SEM image loading, preprocessing, and optional GPU transfer."
    )
    parser.add_argument("--config", default="sem_directory_benchmark.yml")
    parser.add_argument("--exp", default="experiments/sem_dataset_benchmark")
    parser.add_argument("--data-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        config_path = ROOT / "ddim" / "configs" / args.config
    with config_path.open("r", encoding="utf-8") as stream:
        config = dict2namespace(yaml.safe_load(stream))

    if args.workers is not None:
        config.data.num_workers = args.workers
    if args.data_dir is not None:
        config.data.data_dir = args.data_dir
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.pin_memory is not None:
        config.data.pin_memory = args.pin_memory
    if args.persistent_workers is not None:
        config.data.persistent_workers = args.persistent_workers

    dataset_args = argparse.Namespace(exp=args.exp)
    dataset, validation_dataset = get_dataset(dataset_args, config)

    pin_memory = getattr(config.data, "pin_memory", False)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": config.training.batch_size,
        "shuffle": True,
        "num_workers": config.data.num_workers,
        "pin_memory": pin_memory,
    }
    if config.data.num_workers > 0:
        loader_kwargs["persistent_workers"] = getattr(
            config.data, "persistent_workers", False
        )
        loader_kwargs["prefetch_factor"] = getattr(config.data, "prefetch_factor", 2)
    loader = DataLoader(**loader_kwargs)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    iterator = iter(loader)

    def next_batch():
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    for _ in range(args.warmup_batches):
        images, _ = next_batch()
        images = images.to(device, non_blocking=pin_memory)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    durations = []
    sample_count = 0
    for _ in range(args.batches):
        start = time.perf_counter()
        images, _ = next_batch()
        images = images.to(device, non_blocking=pin_memory)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
        sample_count += images.size(0)

    total_seconds = sum(durations)
    result = {
        "config": str(config_path.resolve()),
        "dataset": config.data.dataset,
        "train_images": len(dataset),
        "validation_images": len(validation_dataset),
        "image_shape": list(images.shape[1:]),
        "batch_size": config.training.batch_size,
        "batches_per_epoch": len(loader),
        "num_workers": config.data.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": loader_kwargs.get("persistent_workers", False),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "warmup_batches": args.warmup_batches,
        "timed_batches": args.batches,
        "timed_samples": sample_count,
        "total_seconds": total_seconds,
        "mean_batch_ms": statistics.mean(durations) * 1000,
        "median_batch_ms": statistics.median(durations) * 1000,
        "p95_batch_ms": percentile(durations, 0.95) * 1000,
        "batches_per_second": args.batches / total_seconds,
        "samples_per_second": sample_count / total_seconds,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
