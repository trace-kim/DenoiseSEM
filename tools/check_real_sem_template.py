"""Time the saved-image template diagnostic and check CPU/device agreement."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sem_noise.comparison_storage import asset_path, load_contours
from tools.real_sem_template import measure_template_limit


def check(record_path: Path, *, device: str, frames: int = 32) -> dict:
    """Include PNG decoding and all hole fits; never modify the source report."""
    if frames < 1:
        raise ValueError("frames must be positive")
    root = record_path.resolve().parent
    record = json.loads(record_path.read_text(encoding="utf-8"))
    tolerances = {"scale": 1e-6, "scale_se": 1e-6, "dy_px": 1e-4,
                  "dx_px": 1e-4, "gain": 1e-5, "offset_dn": 1e-3}
    rows = []
    for saved in record["sites"]:
        if not any(r["series"] == "average128" and r["method"] == "refined"
                   for r in saved["observations"]):
            raise ValueError("rebuild with otsu_refined before checking the template diagnostic")
        site = deepcopy(saved)
        site["contours"] = load_contours(root, saved)
        site["series"]["raw"]["frames"] = site["series"]["raw"]["frames"][:frames]
        for relative in [site["full_average"], *(f["path"] for f in site["series"]["raw"]["frames"])]:
            asset_path(root, relative)
        results = {}
        for label, backend in (("cpu", "cpu"), ("candidate", device)):
            trial = deepcopy(site)
            measure_template_limit(root, trial, device=backend)
            results[label] = trial["template_precision"]
        expected, actual = ({(f["hole"], f["frame"]): f for f in results[label]["fits"]}
                            for label in ("cpu", "candidate"))
        statuses_agree = expected.keys() == actual.keys() and all(
            f["status"] == actual[key]["status"] for key, f in expected.items())
        valid = [key for key, fit in expected.items()
                 if fit["status"] == "valid" and actual.get(key, {}).get("status") == "valid"]
        differences = {name: max((abs(expected[key][name] - actual[key][name]) for key in valid), default=None)
                       for name in tolerances}
        passed = statuses_agree and bool(valid) and all(
            np.isfinite(differences[name]) and differences[name] <= tolerance
            for name, tolerance in tolerances.items())
        rows.append({"site": site["name"], "frames": len(site["series"]["raw"]["frames"]),
            "attempted_fits": len(expected), "comparable_valid_fits": len(valid),
            "statuses_agree": statuses_agree, "max_abs_difference": differences,
            "seconds_with_decode": {label: value["seconds"] for label, value in results.items()},
            "passed": passed})
    return {"passed": bool(rows) and all(r["passed"] for r in rows), "sites": rows,
            "device": device, "absolute_tolerances": tolerances,
            "source": "decoded saved uint8 PNGs; conditional template diagnostic"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-comparison", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.frames < 1 or args.cpu_threads < 1:
        parser.error("frame and CPU thread counts must be positive")
    torch.set_num_threads(args.cpu_threads)
    result = check(args.from_comparison, device=args.device, frames=args.frames)
    serialized = json.dumps(result, indent=2, allow_nan=False)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
