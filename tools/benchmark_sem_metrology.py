"""Check single-GPU metrology parity and speed against the CPU on this server.

No checkpoints are needed. Defaults to a quantized synthetic 64-hole array;
repeat --image to measure native uint8 SEM files with the same settings instead.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sem_segment.backends import build_segmenter
from sem_segment.config import Config, load_config
from sem_segment.cuda import CudaRefiner
from sem_segment.image_io import load_image
from sem_segment.pipeline import SegmentationResult, segment_image
from sem_segment.writers import write_json


def synthetic_holes(size: int = 1024) -> np.ndarray:
    from scipy.special import erf

    yy, xx = np.indices((size, size), dtype=float)
    pitch = size / 8
    distance = np.hypot(yy % pitch - pitch / 2, xx % pitch - pitch / 2)
    image = 50 + 160 * .5 * (1 + erf((distance - .24 * pitch) / (np.sqrt(2) * 1.5)))
    image += np.random.default_rng(571).normal(0, 3, image.shape)
    return np.rint(np.clip(image, 0, 255)).astype(np.uint8)


def check_parity(cpu: SegmentationResult, gpu: SegmentationResult, *, tolerance_px: float = 1e-6) -> dict:
    """Fail visibly on topology/rejection changes or shifts exceeding the budget."""
    failures = []
    if not cpu.refined or not gpu.refined:
        failures.append("no refined contours to compare")
    elif not any(c.valid.any() for c in cpu.refined) or not any(c.valid.any() for c in gpu.refined):
        failures.append("no valid refined vertices to compare")
    if not np.array_equal(cpu.label_map(), gpu.label_map()):
        failures.append("segmentation labels differ")
    if len(cpu.refined) != len(gpu.refined) or len(cpu.regions) != len(gpu.regions):
        failures.append("region/contour counts differ")
    coordinate_errors, cd_errors = [], []
    for index, (a, b) in enumerate(zip(cpu.refined, gpu.refined), 1):
        if not np.array_equal(a.valid, b.valid) or not np.array_equal(a.reasons, b.reasons):
            failures.append(f"region {index}: valid/rejection masks differ")
        if a.polygon.shape != b.polygon.shape:
            failures.append(f"region {index}: contour vertex counts differ")
            continue
        if a.polygon.size:
            delta = np.abs(a.polygon - b.polygon)
            coordinate_errors.append(float(delta.max()) if np.isfinite(delta).all() else float("inf"))
    usable = 0
    for index, (a, b) in enumerate(zip(cpu.regions, gpu.regions), 1):
        if a.refined is None or b.refined is None:
            if (a.refined is None) != (b.refined is None):
                failures.append(f"region {index}: refined measurement availability differs")
            continue
        usable += 1
        for key in ("equivalent_diameter_px", "major_axis_px", "minor_axis_px"):
            first, second = getattr(a.refined, key), getattr(b.refined, key)
            if np.isfinite(first) and np.isfinite(second):
                cd_errors.append(abs(first - second))
            elif not (np.isnan(first) and np.isnan(second)):
                failures.append(f"region {index}: nonfinite {key} mismatch")
    if not usable or not cd_errors:
        failures.append("no refined diameter measurements to compare")
    max_coordinate = max(coordinate_errors, default=0.)
    max_cd = max(cd_errors, default=0.)
    if max_coordinate > tolerance_px:
        failures.append(f"maximum contour difference {max_coordinate:.6g} px exceeds {tolerance_px:g}")
    if max_cd > tolerance_px:
        failures.append(f"maximum ECD/axis difference {max_cd:.6g} px exceeds {tolerance_px:g}")
    return {"passed": not failures, "failures": failures, "regions": len(cpu.regions),
            "max_coordinate_difference_px": max_coordinate, "max_diameter_difference_px": max_cd,
            "tolerance_px": tolerance_px}


def benchmark(images: list[tuple[str, np.ndarray]], config: Config, *, device: str = "cuda:0",
              repeats: int = 5, tolerance_px: float = 1e-6) -> dict:
    """Time complete segment_image calls, including transfers and CPU geometry."""
    if repeats < 1 or not np.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("repeats and tolerance_px must be positive and finite")
    if not images:
        raise ValueError("at least one image is required")
    cpu_config = Config.model_validate({**config.model_dump(),
                                       "refine": {**config.refine.model_dump(), "device": "cpu"}})
    gpu_config = Config.model_validate({**config.model_dump(),
                                       "refine": {**config.refine.model_dump(), "device": device}})
    if config.segmentation.backend != "classical":
        raise ValueError("This benchmark requires classical masks to isolate numerical acceleration")
    segmenter = build_segmenter(cpu_config)
    record = {"status": "running", "repeats": repeats, "config": gpu_config.model_dump(mode="json"), "images": []}
    start = time.perf_counter()
    with CudaRefiner(gpu_config.refine) as refiner:
        record["cuda_setup_s"] = time.perf_counter() - start
        record["refinement_backend"] = refiner.describe()
        for name, native in images:
            if native.dtype != np.uint8 or native.ndim != 2:
                raise ValueError(f"{name}: benchmark requires a 2-D native uint8 array")
            image = native.astype(np.float64) / 255.
            timings = {"cpu": [], "gpu": []}
            stages = {"cpu": {}, "gpu": {}}
            warmup = {}
            warm_results = {}
            # Warm-up includes kernels/shape-specific compilation; report it
            # separately so a first-frame cost cannot masquerade as steady speed.
            for label, cfg in (("cpu", cpu_config), ("gpu", gpu_config)):
                start = time.perf_counter()
                warm_results[label] = segment_image(image, cfg, segmenter=segmenter,
                                                    **({"refiner": refiner} if label == "gpu" else {}))
                warmup[label] = time.perf_counter() - start
            parity = check_parity(warm_results["cpu"], warm_results["gpu"], tolerance_px=tolerance_px)
            for repeat in range(repeats):
                # Alternate order to reduce drift from thermal/load changes.
                order = ("cpu", "gpu") if repeat % 2 == 0 else ("gpu", "cpu")
                for label in order:
                    cfg = cpu_config if label == "cpu" else gpu_config
                    start = time.perf_counter()
                    result = segment_image(image, cfg, segmenter=segmenter,
                                           **({"refiner": refiner} if label == "gpu" else {}))
                    timings[label].append(time.perf_counter() - start)
                    for key, value in result.diagnostics.timings_s.items():
                        stages[label].setdefault(key, []).append(value)
            cpu_seconds, gpu_seconds = (float(np.median(timings[k])) for k in ("cpu", "gpu"))
            row = {"image": name, "shape": list(native.shape), "parity": parity,
                   "warmup_s": warmup, "times_s": timings, "median_cpu_s": cpu_seconds,
                   "median_gpu_s": gpu_seconds, "speedup": cpu_seconds / gpu_seconds,
                   "median_stages_s": {k: {s: float(np.median(v)) for s, v in values.items()}
                                       for k, values in stages.items()}}
            record["images"].append(row)
            print(f"{name}: CPU {cpu_seconds:.4f}s/image; GPU {gpu_seconds:.4f}s/image; "
                  f"{row['speedup']:.2f}x; parity {'PASS' if parity['passed'] else 'FAIL'}", flush=True)
            for failure in parity["failures"]:
                print(f"  {failure}", flush=True)
    record["status"] = "passed" if all(r["parity"]["passed"] for r in record["images"]) else "failed"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, action="append", help="Saved uint8 image; repeat for multiple frames")
    parser.add_argument("--config", type=Path, help="Optional sem_segment YAML (same as comparison segmentation_config)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tolerance-px", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=Path("output/sem-metrology-benchmark.json"))
    args = parser.parse_args()
    try:
        config = load_config(args.config) if args.config else Config(
            segmentation={"backend": "classical", "polarity": "dark"})
        images = [(str(p), load_image(p, crop=config.input.crop).native) for p in args.image] if args.image else [
            ("synthetic_uint8_64_holes", synthetic_holes())]
        record = benchmark(images, config, device=args.device, repeats=args.repeats, tolerance_px=args.tolerance_px)
        write_json(args.output, record)
        print(f"Benchmark details: {args.output}")
        return 0 if record["status"] == "passed" else 1
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
