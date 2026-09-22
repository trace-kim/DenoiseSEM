"""Compare CPU reference and batched analysis on saved images, entirely on the server."""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sem_segment.config import Config, MetrologyConfig
from sem_segment.image_io import read_native
from sem_segment.otsu_baseline import OtsuResult, OtsuSettings, otsu_baseline
from sem_segment.otsu_cuda import iter_saved_otsu
from sem_segment.otsu_measurement import measure_otsu_result
from sem_noise.comparison_metrics import native_series_statistics


def signature(result: OtsuResult, settings: OtsuSettings, crop: tuple[int, int, int, int] | None,
              metrology: MetrologyConfig) -> dict:
    measured = measure_otsu_result(result, settings, crop=crop, metrology=metrology)
    digest = hashlib.sha256()
    for path in result.outlines:
        digest.update(np.asarray(path.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(path, dtype="<f8").tobytes())
    return {"mask": np.packbits(result.mask).tobytes(), "outlines": digest.hexdigest(),
            "threshold": result.threshold_dn, "components": result.component_count,
            "retained": result.retained_count,
            "ecd": [r.coarse.equivalent_diameter_px if r.coarse else None for r in measured.regions],
            "timings_s": measured.diagnostics.timings_s}


def benchmark(record_path: Path, *, device: str = "cuda:0", frames_per_source: int = 16,
              batch_size: int = 16, memory_mb: int = 8192, io_workers: int = 2,
              otsu: OtsuSettings | None = None) -> dict:
    """Check every saved source; include frames 9 and 29 where available.

    Images/measurements stay on this machine. Only the returned small timing and
    agreement record is printed or saved. No models, inference or report edits.
    """
    if frames_per_source < 1:
        raise ValueError("frames_per_source must be positive")
    root = record_path.resolve().parent
    record = json.loads(record_path.read_text(encoding="utf-8"))
    config = Config.model_validate(record["segmentation_settings"])
    settings = otsu or OtsuSettings.model_validate(record.get("otsu_settings", {}))
    output = {"device": device, "settings": settings.model_dump(), "batch_size": batch_size,
              "memory_budget_mb": memory_mb, "io_workers": io_workers, "sources": [], "passed": True,
              "scope": "detector + mask metrology and native-statistics agreement; excludes inference, ECC and report writing"}
    warmed = set()
    for site in record["sites"]:
        for name, series in site["series"].items():
            frames = series["frames"]
            preferred = [0, min(8, len(frames) - 1), min(28, len(frames) - 1), len(frames) - 1]
            indices = sorted(list(dict.fromkeys(preferred + list(range(len(frames)))))[:frames_per_source])
            if not indices:
                continue
            paths = []
            for index in indices:
                path = (root / frames[index]["path"]).resolve()
                if root not in path.parents:
                    raise ValueError("Saved image must be inside its report directory")
                paths.append(path)
            source = {"site": site["name"], "source": name,
                      "images": [frames[i]["path"] for i in indices], "differences": []}
            # Warm decoding/library caches before timing either implementation.
            started = time.perf_counter()
            for path in paths:
                signature(otsu_baseline(path, settings, crop=config.input.crop), settings,
                          config.input.crop, config.metrology)
            source["cpu_warmup_s"] = time.perf_counter() - started
            # Keep compressed reference masks, never a series of float images.
            started = time.perf_counter()
            expected = [signature(otsu_baseline(p, settings, crop=config.input.crop), settings,
                                  config.input.crop, config.metrology) for p in paths]
            source["cpu_s"] = time.perf_counter() - started
            shape_key = site.get("name")
            if device != "cpu" and shape_key not in warmed:
                started = time.perf_counter()
                with closing(iter_saved_otsu(paths, settings, crop=config.input.crop, device=device,
                             batch_size=batch_size, memory_mb=memory_mb, io_workers=io_workers)) as results:
                    for result in results:
                        signature(result, settings, config.input.crop, config.metrology)
                source["warmup_s"] = time.perf_counter() - started
                warmed.add(shape_key)
            results = ((otsu_baseline(p, settings, crop=config.input.crop) for p in paths) if device == "cpu" else
                       iter_saved_otsu(paths, settings, crop=config.input.crop, device=device,
                                      batch_size=batch_size, memory_mb=memory_mb, io_workers=io_workers))
            started = time.perf_counter()
            stage_totals = {}
            with closing(results):
                for index, result in enumerate(results):
                    actual = signature(result, settings, config.input.crop, config.metrology)
                    reference = expected[index]
                    failures = [key for key in ("mask", "outlines", "components", "retained", "ecd")
                                if actual[key] != reference[key]]
                    if not np.isclose(actual["threshold"], reference["threshold"], atol=1e-9, rtol=0):
                        failures.append("threshold")
                    if failures:
                        source["differences"].append({"image": source["images"][index], "fields": failures,
                            "cpu_threshold": reference["threshold"], "candidate_threshold": actual["threshold"]})
                    for stage, seconds in actual["timings_s"].items():
                        stage_totals[stage] = stage_totals.get(stage, 0.) + seconds
            source["candidate_s"] = time.perf_counter() - started
            source["images_per_s"] = len(paths) / source["candidate_s"]
            source["cpu_over_candidate"] = source["cpu_s"] / source["candidate_s"]
            source["stage_totals_s"] = stage_totals
            def native(selected_device):
                return native_series_statistics((read_native(p)[0] for p in paths), device=selected_device)

            cpu_rows, cpu_std = native("cpu")
            candidate_rows, candidate_std = native(device)
            row_agreement = all(np.isclose(a["mean_dn"], b["mean_dn"], atol=1e-9, rtol=0)
                                and a["minimum_saved_dn"] == b["minimum_saved_dn"]
                                and a["maximum_saved_dn"] == b["maximum_saved_dn"]
                                for a, b in zip(cpu_rows, candidate_rows))
            std_agreement = (candidate_std is None if cpu_std is None else
                             candidate_std is not None and np.allclose(cpu_std, candidate_std, rtol=1e-12, atol=1e-10))
            source["native_statistics_agree"] = bool(row_agreement and std_agreement)
            source["passed"] = not source["differences"] and source["native_statistics_agree"]
            output["passed"] &= source["passed"]
            output["sources"].append(source)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-comparison", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames-per-source", type=int, default=16)
    parser.add_argument("--analysis-batch", type=int, default=16)
    parser.add_argument("--analysis-memory-mb", type=int, default=8192)
    parser.add_argument("--io-workers", type=int, default=2)
    parser.add_argument("--otsu-config", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    try:
        settings = None
        if args.otsu_config:
            import yaml
            settings = OtsuSettings.model_validate(yaml.safe_load(args.otsu_config.read_text(encoding="utf-8")))
        result = benchmark(args.from_comparison, device=args.device, frames_per_source=args.frames_per_source,
                           batch_size=args.analysis_batch, memory_mb=args.analysis_memory_mb,
                           io_workers=args.io_workers, otsu=settings)
        serialized = json.dumps(result, indent=2)
        print(serialized)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(serialized + "\n", encoding="utf-8")
        return 0 if result["passed"] else 1
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
