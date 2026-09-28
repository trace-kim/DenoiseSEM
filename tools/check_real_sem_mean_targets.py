"""Check the original CPU target against batched CPU/CUDA targets on real crops."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from burst_diffusion.data import BurstCache
from edge_denoise.config import load_config
from edge_denoise.real_matching import MatchedRealPairFactory, build_mean_targets, sample_target


class SpecFactory(MatchedRealPairFactory):
    def _batch(self, samples, *, validation=False):
        return samples, [sample[-1] for sample in samples]


def check(factory: SpecFactory, *, count: int, device: str) -> dict:
    """Time identical samples; include copying, normalization and device execution."""
    state = factory.state_dict()
    samples = factory.sample_batch(count=count)
    factory.load_state_dict(state)
    maximum, agreement, timings = {}, {}, {}
    # Bound live targets by the training micro-batch rather than --targets.
    for start in range(0, count, factory.batch_size):
        specs = [s[1] for s in samples[start:start + factory.batch_size]]
        reference, supports = [], []
        started = time.perf_counter()
        for spec in specs:
            total = np.zeros((spec.size, spec.size), dtype=np.float64)
            valid = np.ones_like(total, dtype=bool)
            for j, matrix, (gain, offset) in zip(spec.indices, spec.matrices, spec.brightness):
                pixels, support = sample_target(factory.frames_by_site[spec.source_index][j], matrix,
                                                 spec.size, factory.black, factory.white)
                total += float(gain) * pixels + float(offset)
                valid &= support
            reference.append((total / len(spec.indices)).astype(np.float32) * 2 - 1)
            supports.append(valid)
        timings["original_cpu"] = timings.get("original_cpu", 0) + time.perf_counter() - started
        for label, backend in (("batched_cpu", "cpu"), ("candidate", device)):
            if backend.startswith("cuda"):
                torch.cuda.synchronize(backend)
            started = time.perf_counter()
            values, valid = build_mean_targets(factory.frames_by_site, specs, factory.black, factory.white, backend)
            if backend.startswith("cuda"):
                torch.cuda.synchronize(backend)
            timings[label] = timings.get(label, 0) + time.perf_counter() - started
            maximum[label] = max(maximum.get(label, 0), float(np.max(np.abs(values[:, 0].cpu().numpy() - reference))))
            agreement[label] = agreement.get(label, True) and np.array_equal(valid[:, 0].cpu().numpy(), supports)
    return {"targets": count, "device": device, "max_abs_difference": maximum,
            "valid_masks_identical": agreement, "seconds": timings,
            "passed": max(maximum.values()) <= 1e-5 and all(agreement.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "edge_denoise/configs/sem_real_ft_loomean.yml")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--real-matching-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.targets < 1 or args.cpu_threads < 1:
        parser.error("target and thread counts must be positive")
    torch.set_num_threads(args.cpu_threads)
    config = load_config(args.config)
    config.data.dataset_dir, config.data.real_matching_cache = args.dataset_dir, args.real_matching_cache
    if config.objective.target != "noisy_mean":
        parser.error("config must use noisy_mean")
    cache = BurstCache(args.dataset_dir, min_replicas=config.min_replicas, min_size=config.data.image_size)
    factory = SpecFactory(cache, config, seed=0)
    result = check(factory, count=args.targets, device=args.device)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
