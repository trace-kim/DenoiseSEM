"""Generate the small synthetic comparison and preview without remote data.

Run from the repository root:
python tests/sem_segment/build_contour_example.py --output-dir tmp/sem_contour_example

Fixture preparation stores current-method outlines from generated uint8 files.
The production preview command then consumes that report without running the
current method. Neither phase calculates ECD or loads a trained model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sem_segment.backends import build_segmenter
from sem_segment.config import Config
from sem_segment.contours import trace_all
from sem_segment.image_io import read_native
from sem_segment.masks import postprocess
from sem_segment.otsu_baseline import OtsuSettings
from sem_segment.refine import adaptive_search_px, refine_all
from synthetic_images import visual_qa_frames
from tools.preview_sem_contours import build_preview


def build_example(output_dir: Path) -> Path:
    """Save two acquisitions per source, their blocks, and the full average."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    source = output_dir / "source"
    source.mkdir()
    selected = {9, 29}
    series, raw_images = {}, []

    def save_frame(name: str, pixels: np.ndarray, index: int, **extra) -> None:
        folder = source / name
        folder.mkdir(exist_ok=True)
        path = folder / f"frame_{index:03d}.png"
        Image.fromarray(np.repeat(pixels[..., None], 3, axis=2)).save(path)
        series.setdefault(name, {"frames": []})["frames"].append({
            "index": index, "order": index, "path": path.relative_to(source).as_posix(), **extra})

    for index, frames in visual_qa_frames():
        raw_images.append(frames["raw"])
        if index in selected:
            for name, pixels in frames.items():
                note = ("Synthetic frame 9: high noise (55 DN before uint8 export)." if index == 9 else
                        "Synthetic frame 29: false split in the committed classical detector.")
                if name != "raw":
                    note = "Simulated output from the same synthetic geometry; no trained model."
                save_frame(name, pixels, index, note=note)
    for block in (2, 4):
        values = np.stack(raw_images[(block - 1) * 8:block * 8])
        pixels = np.rint(values.mean(axis=0)).astype(np.uint8)
        save_frame("average8", pixels, block, first_acquisition=(block - 1) * 8 + 1,
                   last_acquisition=block * 8)
    average = np.rint(np.stack(raw_images).mean(axis=0)).astype(np.uint8)
    save_frame("average128", average, 1, first_acquisition=1, last_acquisition=128)

    config = Config(segmentation={"backend": "classical", "polarity": "dark", "contrast_stretch": None},
                    masks={"min_area_px": 60})
    detector = build_segmenter(config)
    stored = []
    for name, value in series.items():
        for frame in value["frames"]:
            pixels, _ = read_native(source / frame["path"])
            instances = detector.segment(np.repeat(pixels[..., None], 3, axis=2))
            instances, _ = postprocess(instances, pixels.shape, config.masks)
            coarse, holes = trace_all(instances, pixels.shape, config.contours)
            refined = refine_all(coarse, pixels.astype(np.float64) / 255, config.refine,
                                 spacing_px=config.contours.spacing_px,
                                 search_px=[adaptive_search_px(m.crop, config.refine) for m in instances])
            frame["contour_counts"] = {"detected": len(instances)}
            for i, contour in enumerate(coarse):
                stored.append({"series": name, "frame": frame["index"], "coarse": contour.points.tolist(),
                               "holes": [h.points.tolist() for h in holes[i]],
                               "refined": refined[i].polygon.tolist(), "refined_valid": refined[i].valid.tolist()})
    (source / "contours.json").write_text(json.dumps(stored), encoding="utf-8")
    record = {"schema_version": 3, "synthetic": True,
              "study": "Synthetic high-noise and false-split cases (acquisitions 9 and 29)",
              "segmentation_settings": config.model_dump(mode="json"),
              "sites": [{"name": "synthetic_holes", "series": series, "contours_path": "contours.json"}]}
    path = source / "comparison.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    build_preview(path, output_dir / "preview", device="cpu")
    build_preview(path, output_dir / "preview_sigma0", OtsuSettings(sigma_px=0), device="cpu")
    return output_dir / "preview/index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build_example(args.output_dir))
