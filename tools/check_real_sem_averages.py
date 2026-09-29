"""Time registered averaging and compare decoded CPU/GPU uint8 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.real_sem_averages import registered_average
from tools.real_sem_compare import read_uint8, save_rgb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-comparison", type=Path)
    source.add_argument("--site-dir", type=Path, help="Check the first block directly from raw acquisitions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", default="raw")
    parser.add_argument("--average-frames", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.average_frames < 2 or args.cpu_threads < 1:
        parser.error("average frames must be >=2 and CPU threads positive")
    torch.set_num_threads(args.cpu_threads)
    import cv2

    cv2.setNumThreads(args.cpu_threads)
    if args.site_dir is not None:
        from tools.prepare_real_sem_blocks import image_files

        if args.source != "raw":
            parser.error("--site-dir supports --source raw only")
        root = args.site_dir.resolve()
        record = {"sites": [{"name": root.name, "series": {"raw": {"frames": [
            {"path": p.name} for p in image_files(root)]}}}]}
    else:
        root = args.from_comparison.resolve().parent
        record = json.loads(args.from_comparison.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for site in record["sites"]:
        started = time.perf_counter()
        frames = site["series"][args.source]["frames"][:args.average_frames]
        if len(frames) != args.average_frames:
            raise ValueError("not enough saved acquisitions")
        paths = [(root / f["path"]).resolve() for f in frames]
        if any(root not in path.parents for path in paths):
            raise ValueError("saved image must be inside its report")
        pixels = np.stack([read_uint8(path) for path in paths])
        row = {"site": site["name"], "source": args.source, "decode_seconds": time.perf_counter() - started}
        outputs, masks = {}, {}
        for label, device in (("cpu", "cpu"), ("candidate", args.device)):
            started = time.perf_counter()
            image, mask, diagnostics = registered_average(pixels, device=device)
            row[label] = diagnostics
            path = args.output_dir / f"{site['name']}_{label}.png"
            save_rgb(path, image)
            outputs[label], masks[label] = read_uint8(path), mask
            row[label]["total_with_export_decode_seconds"] = time.perf_counter() - started
        difference = np.abs(outputs["cpu"].astype(int) - outputs["candidate"].astype(int))
        row.update(max_abs_dn=int(difference.max()), changed_fraction=float((difference > 0).mean()),
                   support_agrees=bool(np.array_equal(masks["cpu"], masks["candidate"])))
        row["passed"] = row["max_abs_dn"] <= 1 and row["changed_fraction"] <= .001 and row["support_agrees"]
        rows.append(row)
    result = {"passed": all(r["passed"] for r in rows), "sources": rows,
              "measurement_source": "decoded saved uint8 PNGs", "device": args.device}
    serialized = json.dumps(result, indent=2)
    (args.output_dir / "equivalence.json").write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
