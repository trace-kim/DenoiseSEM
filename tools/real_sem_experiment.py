"""Edit SETTINGS below, then run: python tools/real_sem_experiment.py.

One folder of repeated uint8 acquisitions of one site per experiment.
Relative paths are resolved from the repository root, on Windows or Linux.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image


# EDIT THESE SETTINGS. Blank paths use the defaults described in the guide.
SETTINGS = {
    "checkpoint": "",
    "source_dir": "",
    "output_dir": "",
    "png_dir": "",
    "raw_report_dir": "",
    "png_report_dir": "",
    "analysis_config": "",
    "device": "auto",  # Use CUDA_VISIBLE_DEVICES to select a GPU on Linux.
    "tile_batch": 4,
    "ema": True,
}

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}


def resolve_settings(settings: dict, root: Path = ROOT) -> dict:
    """Resolve editable paths without creating files or loading a checkpoint."""
    def path(value: str | Path) -> Path:
        result = Path(value).expanduser()
        return (result if result.is_absolute() else root / result).resolve()

    if not str(settings["checkpoint"]).strip():
        raise ValueError("Set checkpoint in SETTINGS or pass --checkpoint")
    result = dict(settings)
    checkpoint = path(settings["checkpoint"])
    run_name = checkpoint.parent.name
    result.update(checkpoint=checkpoint, run_name=run_name)
    output = path(settings["output_dir"] or f"output/{run_name}")
    defaults = {
        "source_dir": root / "data" / "SEM-test" / run_name,
        "output_dir": output,
        "png_dir": output / "png",
        "raw_report_dir": output / "noise-raw",
        "png_report_dir": output / "noise-png",
    }
    for key, default in defaults.items():
        result[key] = path(settings[key] or default)
    result["analysis_config"] = path(settings["analysis_config"]) if settings["analysis_config"] else None
    return result


def png_pixels(normalized: np.ndarray, black: float, white: float) -> np.ndarray:
    """Restore original intensity units and quantize to the original uint8 type."""
    return np.rint(np.clip(normalized * (white - black) + black, 0, 255)).astype(np.uint8)


def run(settings: dict) -> None:
    """Run full-frame inference and analyze matching raw and denoised PNG sets."""
    from edge_denoise.infer import Denoiser
    from edge_denoise.real_data import read_native
    from sem_noise.config import load_config
    from sem_noise.io import natural_key
    from sem_noise.pipeline import analyze_dataset

    config = resolve_settings(settings)
    source, output = config["source_dir"], config["output_dir"]
    if not config["checkpoint"].is_file():
        raise ValueError(f"Checkpoint does not exist: {config['checkpoint']}")
    if not source.is_dir():
        raise ValueError(f"Source folder does not exist: {source}. Set source_dir or populate this folder.")
    if any(p.is_dir() for p in source.iterdir()):
        raise ValueError("Source must be a flat folder of repeats of one site; set source_dir to that site")
    files = sorted((p for p in source.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file()),
                   key=lambda p: natural_key(p.name))
    analysis_config = load_config(config["analysis_config"])
    if len(files) < analysis_config.min_frames:
        raise ValueError(f"Need at least {analysis_config.min_frames} frames; found {len(files)}")
    if len({p.stem.casefold() for p in files}) != len(files):
        raise ValueError("Source filenames must have unique stems")
    if config["tile_batch"] < 1:
        raise ValueError("tile_batch must be positive")

    destinations = [output / "inference", output / "raw", config["png_dir"],
                    config["raw_report_dir"], config["png_report_dir"]]
    for target in [output, *destinations]:
        if target.exists():
            raise ValueError(f"Output already exists: {target}. Choose new output paths.")
        if target == source or source in target.parents or target in source.parents:
            raise ValueError(f"Output must not overlap the source folder: {target}")
    for i, target in enumerate(destinations):
        for other in destinations[i + 1:]:
            if target == other or target in other.parents or other in target.parents:
                raise ValueError(f"Output folders overlap: {target} and {other}")
        if target == output or target in output.parents:
            raise ValueError(f"Output subfolder cannot contain the experiment root: {target}")

    # Validate every input before running the model or creating experiment files.
    shape = None
    for path in files:
        frame = read_native(path)
        if frame.dtype != np.uint8:
            raise ValueError(f"This PNG workflow requires uint8 input: {path}")
        if shape is not None and frame.shape != shape:
            raise ValueError(f"All frames must have identical dimensions: {path}")
        shape = frame.shape

    denoiser = Denoiser.from_checkpoint(config["checkpoint"], device=config["device"], use_ema=config["ema"])
    black, white = denoiser.config.data.black_level, denoiser.config.data.white_level
    if black is None and white is None:
        black, white = 0.0, 255.0
    if black is None or white is None or not np.isfinite([black, white]).all() or white <= black:
        raise ValueError("Invalid checkpoint normalization levels")
    if min(shape) < denoiser.image_size:
        raise ValueError(f"Images must be at least {denoiser.image_size} pixels on each side")
    stride = min(48, denoiser.image_size) if denoiser.image_size <= 64 else denoiser.image_size // 2
    output.mkdir(parents=True)
    inference, raw, png = destinations[:3]
    for folder in (inference, raw, png):
        folder.mkdir(parents=True, exist_ok=False)
    record = {**config, "black_level": black, "white_level": white,
              "frames": [p.name for p in files], "status": "running"}

    def save_record() -> None:
        (output / "experiment.json").write_text(json.dumps(record, default=str, indent=2) + "\n", encoding="utf-8")

    save_record()
    print(json.dumps(record, default=str, indent=2), flush=True)
    try:
        for index, path in enumerate(files, 1):
            print(f"[{index}/{len(files)}] {path.name}", flush=True)
            frame = denoiser.load_measurement(path)
            prediction = np.clip(denoiser.denoise_full(
                frame, stride=stride, tile_batch=config["tile_batch"],
            ), 0, 1).astype(np.float32)
            if not np.isfinite(prediction).all():
                raise ValueError(f"Nonfinite prediction: {path}")
            Image.fromarray(png_pixels(frame, 0, 255)).save(inference / f"{path.stem}_input.png")
            preview = inference / f"{path.stem}_denoised.png"
            Image.fromarray(png_pixels(prediction, 0, 255)).save(preview)
            Image.fromarray(prediction).save(inference / f"{path.stem}_denoised.tif")
            if black == 0 and white == 255:
                shutil.copy2(preview, png / preview.name)
            else:
                Image.fromarray(png_pixels(prediction, black, white)).save(png / preview.name)
            # Decoded raw pixels in lossless PNG give both analyses matching formats.
            Image.fromarray(read_native(path)).save(raw / f"{path.stem}.png")
        statuses = {}
        for label, folder, report in (("raw", raw, config["raw_report_dir"]),
                                      ("png", png, config["png_report_dir"])):
            result = analyze_dataset(folder, report, config=analysis_config,
                                     progress=lambda message: print(message, flush=True))
            statuses[label] = result["status"]
        record["analysis_status"] = statuses
        if any(status != "complete" for status in statuses.values()):
            raise RuntimeError("Analysis reported failed sites; inspect the reports")
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        save_record()
    print(f"Raw report: {config['raw_report_dir'] / 'index.html'}")
    print(f"PNG report: {config['png_report_dir'] / 'index.html'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "source_dir", "output_dir", "png_dir", "raw_report_dir", "png_report_dir", "analysis_config", "device"):
        parser.add_argument("--" + key.replace("_", "-"), default=SETTINGS[key])
    parser.add_argument("--tile-batch", type=int, default=SETTINGS["tile_batch"])
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=SETTINGS["ema"])
    try:
        run(vars(parser.parse_args()))
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
