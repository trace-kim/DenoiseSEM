"""Remote CPU/GPU inference check measured only on decoded saved uint8 PNGs.

Defaults to one centre crop per checkpoint to bound CPU cost. Use the existing
benchmark_sem_analysis.py separately to check the unchanged measurement backend.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from burst_diffusion.provenance import file_sha256
from edge_denoise.infer import Denoiser
from edge_denoise.real_data import read_native
from edge_denoise.real_suite import DEFAULT_TEST_SITE
from edge_denoise.uint8_output import prediction_uint8
from runctl.run_logging import atomic_write_json


def checkpoints_from_suite(path: Path) -> dict[str, Path]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "complete":
        raise ValueError("equivalence check requires a completed suite; failed methods cannot be omitted")
    plan = record["plan"]
    teacher = plan["teacher"]
    checkpoints = {"n2n": Path(teacher["path"])}
    if file_sha256(checkpoints["n2n"]) != teacher["sha256"]:
        raise ValueError("suite teacher checkpoint changed")
    for pipeline in plan["pipelines"]:
        name = pipeline["name"]
        checkpoint = Path(pipeline["config"]["training"]["run_dir"]) / "ckpt_latest.pt"
        row = record["pipelines"][name]
        if row["status"] != "complete" or file_sha256(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"{name}: suite checkpoint missing, changed or incomplete")
        checkpoints[name] = checkpoint
    return checkpoints


def check_checkpoint(checkpoint: Path, paths: list[Path], output_dir: Path, *, device: str,
                     full_frame: bool = False, max_dn: int = 1,
                     max_changed_fraction: float = .01) -> dict:
    outputs, rows, timings, audits = {}, [], {}, {}
    for label, backend in (("cpu", "cpu"), ("candidate", device)):
        started = time.perf_counter()
        denoiser = Denoiser.from_checkpoint(checkpoint, device=backend)
        timings[f"{label}_load_s"] = time.perf_counter() - started
        saved = []
        for index, source in enumerate(paths):
            raw = read_native(source)
            if raw.dtype != np.uint8:
                raise ValueError("equivalence inputs must be native uint8")
            black, white = denoiser.config.data.black_level, denoiser.config.data.white_level
            if black is None or white is None:
                raise ValueError("equivalence check requires a real-SEM checkpoint")
            frame = np.clip((raw.astype(np.float64) - black) / (white - black), 0, 1)
            size = denoiser.image_size
            if min(frame.shape) < size:
                raise ValueError("input smaller than checkpoint crop")
            if backend.startswith("cuda"):
                torch.cuda.synchronize(torch.device(backend))
            started = time.perf_counter()
            if full_frame:
                predicted = denoiser.denoise_full(frame, stride=size // 2, tile_batch=4, clip_output=False)
            else:
                y, x = (frame.shape[0] - size) // 2, (frame.shape[1] - size) // 2
                tensor = torch.tensor(frame[y:y + size, x:x + size] * 2 - 1, dtype=torch.float32)[None, None]
                predicted = (denoiser.denoise(tensor, clip_output=False)[0, 0].numpy().astype(np.float64) + 1) / 2
            if backend.startswith("cuda"):
                torch.cuda.synchronize(torch.device(backend))
            timings[f"{label}_inference_s"] = timings.get(f"{label}_inference_s", 0) + time.perf_counter() - started
            pixels, audit = prediction_uint8(predicted, black, white)
            destination = output_dir / label / f"{index:03d}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.repeat(pixels[..., None], 3, axis=2)).save(destination)
            saved.append(destination)
            audits[f"{label}/{index}"] = audit  # Production range checks only.
        outputs[label] = saved
        del denoiser
    for index, source in enumerate(paths):
        # Quantize, save, decode; never measure intermediate predictions.
        cpu = read_native(outputs["cpu"][index])
        candidate = read_native(outputs["candidate"][index])
        if cpu.shape != candidate.shape:
            raise ValueError("CPU and candidate output shapes differ")
        difference = np.abs(cpu.astype(np.int16) - candidate.astype(np.int16))
        maximum = int(difference.max())
        changed = float(np.count_nonzero(difference) / difference.size)
        rows.append({"source": str(source), "cpu_png": str(outputs["cpu"][index]),
                     "candidate_png": str(outputs["candidate"][index]),
                     "max_abs_difference_dn": maximum, "changed_fraction": changed,
                     "mean_abs_difference_dn": float(difference.mean()),
                     "passed": maximum <= max_dn and changed <= max_changed_fraction})
    return {"passed": all(row["passed"] for row in rows), "frames": rows, "timings": timings,
            "prediction_ranges": audits, "checkpoint_sha256": file_sha256(checkpoint)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-suite", type=Path)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_TEST_SITE)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--full-frame", action="store_true")
    parser.add_argument("--max-dn", type=int, default=1)
    parser.add_argument("--max-changed-fraction", type=float, default=.01)
    args = parser.parse_args(argv)
    try:
        if args.frames < 1 or args.cpu_threads < 1 or args.max_dn < 0 or not 0 <= args.max_changed_fraction <= 1:
            raise ValueError("invalid frame/thread count or equivalence tolerances")
        checkpoints = checkpoints_from_suite(args.from_suite) if args.from_suite else {}
        for item in args.checkpoint:
            name, separator, filename = item.partition("=")
            if not separator or not filename or name in checkpoints or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("--checkpoint requires a unique NAME=PATH")
            checkpoints[name] = Path(filename)
        if not checkpoints:
            raise ValueError("provide --from-suite or --checkpoint")
        paths = sorted(p for p in args.site_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"})[:args.frames]
        if len(paths) != args.frames:
            raise ValueError("site does not contain the requested number of images")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        torch.set_num_threads(args.cpu_threads)
        report = {"measurement_source": "decoded saved uint8 RGB PNGs", "device": args.device,
                  "mode": "full_frame" if args.full_frame else "center_crop", "models": {},
                  "max_dn": args.max_dn, "max_changed_fraction": args.max_changed_fraction}
        for name, checkpoint in checkpoints.items():
            try:
                result = check_checkpoint(checkpoint, paths, args.output_dir / name, device=args.device,
                                          full_frame=args.full_frame, max_dn=args.max_dn,
                                          max_changed_fraction=args.max_changed_fraction)
            except (ValueError, OSError, RuntimeError) as error:
                result = {"passed": False, "error": str(error)}
            report["models"][name] = result
            print(f"{name}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
        report["passed"] = all(row["passed"] for row in report["models"].values())
        atomic_write_json(args.output_dir / "equivalence.json", report)
        return 0 if report["passed"] else 1
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Equivalence check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
