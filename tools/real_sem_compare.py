"""Compare saved uint8 outputs: python tools/real_sem_compare.py --config PATH.

This coordinator is the only layer coupling training, noise and segmentation.
Paths in YAML are relative to the repository root, as in real_sem_experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.real_sem_experiment import EXTENSIONS
from edge_denoise.uint8_output import RANGE_WARNING, average_uint8, prediction_uint8
from sem_noise.progress import Progress
from sem_segment.otsu_baseline import OtsuSettings

if TYPE_CHECKING:
    from edge_denoise.infer import Denoiser
    from sem_noise.config import AnalysisConfig
    from sem_segment.config import Config as SegmentConfig


class SiteSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_dir: Path
    timestamps_s: list[float] | None = None


class ComparisonArm(BaseModel):
    """The training treatment, not a correction to apply at inference."""
    model_config = ConfigDict(extra="forbid")
    checkpoint: Path
    registration: Literal["affine", "translation", "none"]
    brightness: Literal["percentile", "none"]
    prepared_manifest: Path | None = None


class ComparisonSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoints: dict[str, ComparisonArm]
    sites: dict[str, Path | SiteSettings]
    output_dir: Path
    analysis_config: Path | None = None
    segmentation_config: Path | None = None
    # Older programmatic callers retain the current method. The shipped real
    # comparison recipe explicitly selects the Gaussian + Otsu method.
    contour_method: Literal["current", "otsu"] = "current"
    otsu: OtsuSettings = Field(default_factory=OtsuSettings)
    metrology_device: str | None = Field(default=None, pattern=r"^(cpu|cuda|cuda:[0-9]+)$")
    device: str = "auto"
    tile_batch: int = Field(default=32, ge=1)
    analysis_batch: int = Field(default=16, ge=1)
    analysis_memory_mb: int = Field(default=8192, ge=1)
    io_workers: int = Field(default=2, ge=1)
    tensorboard: bool = True
    difference_limit_dn: float = Field(default=32.0, gt=0, allow_inf_nan=False)
    ema: bool = True
    frame_interval_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    pixel_size_nm: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    match_gate_px: float | None = Field(default=None, gt=0, allow_inf_nan=False)


def read_uint8(path: Path) -> np.ndarray:
    from edge_denoise.real_data import read_native

    frame = read_native(path)
    if frame.dtype != np.uint8:
        raise ValueError(f"Comparison requires uint8 pixels: {path}")
    return frame


def save_rgb(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.repeat(pixels[..., None], 3, axis=2)).save(path)


def load_settings(path: Path, *, root: Path = ROOT) -> ComparisonSettings:
    config = ComparisonSettings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def resolve(value: Path | None) -> Path | None:
        if value is None:
            return None
        return (root / value.expanduser()).resolve()

    config.output_dir = resolve(config.output_dir)
    config.analysis_config = resolve(config.analysis_config)
    config.segmentation_config = resolve(config.segmentation_config)
    config.checkpoints = {name: arm.model_copy(update={"checkpoint": resolve(arm.checkpoint),
                                                    "prepared_manifest": resolve(arm.prepared_manifest)})
                          for name, arm in config.checkpoints.items()}
    config.sites = {name: SiteSettings(source_dir=resolve(value)) if isinstance(value, Path)
                    else value.model_copy(update={"source_dir": resolve(value.source_dir)})
                    for name, value in config.sites.items()}
    return config


def validate_inputs(config: ComparisonSettings) -> dict[str, list[Path]]:
    from sem_noise.io import natural_key

    if not config.checkpoints or not config.sites:
        raise ValueError("At least one named checkpoint and site are required")
    for names in (config.checkpoints, config.sites):
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Names must be unique ignoring case")
        for name in names:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
                raise ValueError(f"Use letters, digits, underscores or hyphens for names: {name}")
    if {name.casefold() for name in config.checkpoints} & {"raw", "average8", "average128"}:
        raise ValueError("raw, average8 and average128 are reserved series names")
    if config.output_dir.exists():
        raise ValueError(f"Output already exists: {config.output_dir}")
    for arm in config.checkpoints.values():
        if not arm.checkpoint.is_file():
            raise ValueError(f"Checkpoint does not exist: {arm.checkpoint}")
    sites = {}
    for name, settings in config.sites.items():
        source = settings.source_dir
        if not source.is_dir():
            raise ValueError(f"Site folder does not exist: {source}")
        if source == config.output_dir or source in config.output_dir.parents or config.output_dir in source.parents:
            raise ValueError("Output must not overlap a site directory")
        if any(p.is_dir() for p in source.iterdir()):
            raise ValueError(f"Site must be a flat directory: {source}")
        files = sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                       key=lambda p: natural_key(p.name))
        if len(files) != 128:
            raise ValueError(f"{name}: expected exactly 128 frames, found {len(files)}")
        if settings.timestamps_s is not None:
            stamps = np.asarray(settings.timestamps_s)
            if stamps.shape != (128,) or not np.isfinite(stamps).all() or not (np.diff(stamps) > 0).all():
                raise ValueError(f"{name}: need 128 finite, strictly increasing timestamps_s")
        shape = None
        for file in files:
            frame = read_uint8(file)
            if min(frame.shape) < 16 or (shape is not None and frame.shape != shape):
                raise ValueError(f"{name}: frames need consistent dimensions of at least 16x16")
            shape = frame.shape
        sites[name] = files
    return sites


def checkpoint_arm_metadata(arm: ComparisonArm, denoiser: Denoiser) -> dict:
    """Verify the six N2N treatments, including the two prepared-data baselines.

    A missing inline setting does NOT mean registration=none: old checkpoints
    get their registration from the original prepared manifest. Only metadata
    are read here, never training arrays, and no treatment is applied to inputs.
    """
    from burst_diffusion.data import resolve_burst_dir
    from sem_noise.io import file_hash

    config = denoiser.config
    objective = config.objective
    if (objective.representation != "image" or objective.target != "noisy"
            or objective.loss != "l2" or objective.lambda_image != 1
            or objective.lambda_gradient != 0 or objective.lambda_consistency != 0
            or objective.fusion is not None):
        raise ValueError("This comparison requires the sem_real_n2n image/noisy L2 objective in every arm")
    manifest_path = arm.prepared_manifest
    if manifest_path is None:
        try:
            manifest_path = resolve_burst_dir(ROOT / config.data.dataset_dir) / "real_dataset.json"
        except FileNotFoundError as error:
            raise ValueError("Set prepared_manifest to this checkpoint's original real_dataset.json on the server") from error
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "real_sem":
        raise ValueError("Comparison arms require a prepared real-SEM manifest")
    digest = file_hash(manifest_path)
    checkpoint_digest = getattr(denoiser, "dataset_fingerprint", None)
    if checkpoint_digest is not None and checkpoint_digest != digest:
        raise ValueError("Prepared manifest hash differs from the checkpoint's training dataset fingerprint")
    levels = manifest["normalization"]
    if (config.data.black_level, config.data.white_level) != (levels["black"], levels["white"]):
        raise ValueError("Checkpoint normalization differs from its prepared manifest")
    matching = config.data.real_matching
    prepared = manifest["registration"]
    if matching is not None:
        if prepared["mode"] != "none":
            raise ValueError("Inline matching expects the original align-none prepared dataset")
        registration, brightness = matching.registration, matching.brightness
        source = "checkpoint inline matching"
    else:
        registration, brightness = prepared["mode"], "none"
        source = "legacy prepared dataset"
    if (arm.registration, arm.brightness) != (registration, brightness):
        raise ValueError(f"Declared arm {arm.registration}+{arm.brightness} conflicts with "
                         f"{source}: {registration}+{brightness}")
    # Registered/unaligned manifests have different hashes even when they use
    # the exact same native acquisitions and held-out sites. Compare raw content.
    splits = {split: sorted(sorted(f["sha256"] for f in site["frames"])
                            for site in manifest["sites"] if site["split"] == split)
              for split in ("train", "val", "test")}
    content_split_hash = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    excluded = sorted({f["sha256"] for site in manifest["sites"] if site["split"] in ("train", "val")
                       for f in site["frames"]})
    return {"registration": registration, "brightness": brightness, "settings_source": source,
            "prepared_manifest": str(manifest_path), "prepared_manifest_sha256": digest,
            "checkpoint_dataset_fingerprint": checkpoint_digest, "raw_content_split_sha256": content_split_hash,
            "split_site_counts": {split: len(sites) for split, sites in splits.items()},
            "train_val_content_hashes": excluded, "prepared_registration": prepared,
            "inline_matching": matching.model_dump(mode="json") if matching else None,
            "warnings": [] if checkpoint_digest else ["Older checkpoint has no dataset fingerprint; manifest linkage cannot be verified."]}


def validate_comparable_arm(metadata: dict, model_config: dict, previous: dict[str, dict]) -> None:
    """Keep architecture, objective, normalization and source splits comparable."""
    if not previous:
        return
    first = next(iter(previous.values()))
    if metadata["raw_content_split_sha256"] != first["arm"]["raw_content_split_sha256"]:
        raise ValueError("Comparison arms must use the same raw acquisition content and train/val/test site splits")
    for section in ("model", "objective"):
        if model_config[section] != first["config"][section]:
            raise ValueError(f"Comparison arms have different {section} settings")
    for field in ("image_size", "channels", "black_level", "white_level"):
        if model_config["data"][field] != first["config"]["data"][field]:
            raise ValueError(f"Comparison arms have different data.{field}")


def registration_tracks(reference: np.ndarray, paths: list[Path], sigma: float) -> list[dict]:
    from sem_noise.pair_matching import estimate_geometry

    tracks = []
    label = "/".join(paths[0].parts[-3:-1]) if paths else "empty series"
    with Progress(f"{label}: translation diagnostics (CPU ECC)", total=len(paths)) as progress:
        for path in paths:
            pixels = read_uint8(path)
            row = {"mean_dn": float(pixels.mean())}
            try:
                matrix, score = estimate_geometry(reference, pixels, motion="translation", sigma=sigma)
                row.update(dy_px=float(matrix[1, 2]), dx_px=float(matrix[0, 2]),
                           registration_status="registered", registration_score=score, registration_error="")
            except ValueError as error:
                row.update(dy_px=None, dx_px=None, registration_status="failed",
                           registration_score=None, registration_error=str(error))
            tracks.append(row)
            progress.update(len(tracks), path.name)
    return tracks


def output_registration_diagnostics(root: Path, reference: np.ndarray, series: dict, sigma: float) -> None:
    """Measure each output's drift independently, without changing correspondence."""
    started = time.perf_counter()
    tracks = registration_tracks(reference, [root / f["path"] for f in series["frames"]], sigma)
    for frame, track in zip(series["frames"], tracks):
        for key in ("dy_px", "dx_px", "registration_status", "registration_score", "registration_error"):
            frame[f"output_{key}"] = track[key]
        valid = frame["registration_status"] == "registered" and track["registration_status"] == "registered"
        for component in ("dy_px", "dx_px"):
            frame[f"output_minus_raw_{component}"] = track[component] - frame[component] if valid else None
    series.setdefault("timings_s", {})["translation_diagnostics"] = time.perf_counter() - started


def measure_series(root: Path, name: str, series: dict, template: np.ndarray,
                   gate: float, segment_config: SegmentConfig, *,
                   contour_method: str = "current", otsu: OtsuSettings | None = None,
                   analysis_batch: int = 16, analysis_memory_mb: int = 8192,
                   io_workers: int = 2) -> tuple[list[dict], list[dict]]:
    if contour_method == "otsu":
        from sem_segment.otsu_measurement import iter_measure_saved_otsu

        settings = otsu or OtsuSettings()
        results = iter_measure_saved_otsu([root / f["path"] for f in series["frames"]], settings,
            crop=segment_config.input.crop, device=segment_config.refine.device, metrology=segment_config.metrology,
            batch_size=analysis_batch, memory_mb=analysis_memory_mb, io_workers=io_workers)
        try:
            return _measure_series(root, name, series, template, gate, segment_config, otsu=settings, results=results)
        finally:
            label = Path(series["frames"][0]["path"]).parent.as_posix() if series["frames"] else name
            with Progress(f"{label}: closing Otsu workers", timings=series.setdefault("timings_s", {}), key="contour_cleanup"):
                results.close()
    if contour_method != "current":
        raise ValueError("contour_method must be current or otsu")
    if segment_config.refine.enabled and segment_config.refine.device != "cpu":
        from sem_segment.cuda import CudaRefiner

        with CudaRefiner(segment_config.refine) as refiner:
            return _measure_series(root, name, series, template, gate, segment_config, refiner=refiner)
    return _measure_series(root, name, series, template, gate, segment_config)


def _measurement_status(region, method: str, config: SegmentConfig, *, mask_only: bool = False) -> str:
    if method == "refined" and mask_only:
        return "not_run"
    shape = getattr(region, method)
    if region.touches_border:
        return "border"
    if shape is None:
        return "measurement_failed"
    if method == "refined" and region.valid_fraction < config.refine.min_valid_fraction:
        return "insufficient_refined_vertices"
    if not np.isfinite([shape.equivalent_diameter_px, shape.major_axis_px, shape.minor_axis_px]).all():
        return "nonfinite_measurement"
    return "valid"


def _measure_series(root: Path, name: str, series: dict, template: np.ndarray,
                    gate: float, segment_config: SegmentConfig, *, refiner=None,
                    otsu: OtsuSettings | None = None, results=None) -> tuple[list[dict], list[dict]]:
    from sem_segment.backends import build_segmenter
    from sem_segment.pipeline import segment_image
    from sem_segment.repeatability import match_centroids

    started = last_progress = time.perf_counter()
    count = len(series["frames"])
    label = "/".join(Path(series["frames"][0]["path"]).parts[:-1]) if count else name
    detector = f"Gaussian + Otsu on {segment_config.refine.device}; no refinement" if otsu is not None else f"refinement {segment_config.refine.device}"
    print(f"{label}: contours/CD starting ({count} saved uint8 images; {detector})", flush=True)
    segmenter = build_segmenter(segment_config) if otsu is None else None
    rows, contours = [], []
    stage_totals = {}
    factor = segment_config.input.pixel_size_nm or 1.0
    for number, frame in enumerate(series["frames"], 1):
        # Measurements always start by decoding the saved, quantized file.
        crop = segment_config.input.crop
        if otsu is not None:
            result = next(results)
            frame["otsu_threshold_dn"] = result.diagnostics.backend["threshold_dn"]
            frame["gaussian_backend"] = result.diagnostics.backend["gaussian_backend"]
            frame["segmentation_execution"] = result.diagnostics.backend["execution"]
            series["segmentation_backend"] = {"backend": "otsu", "settings": otsu.model_dump(),
                                               "device": segment_config.refine.device,
                                               "execution": result.diagnostics.backend["execution"]}
        else:
            pixels = read_uint8(root / frame["path"])
            if crop:
                y0, y1, x0, x1 = crop
                pixels = pixels[y0:y1, x0:x1]
            extra = {"refiner": refiner} if refiner is not None else {}
            result = segment_image(pixels.astype(np.float64) / 255.0, segment_config, segmenter=segmenter, **extra)
            frame.pop("otsu_threshold_dn", None)
            frame.pop("gaussian_backend", None)
            frame.pop("segmentation_execution", None)
            series["segmentation_backend"] = result.diagnostics.backend
        series["refinement_backend"] = result.diagnostics.refinement
        frame["segmentation_timings_s"] = result.diagnostics.timings_s
        for stage, seconds in result.diagnostics.timings_s.items():
            stage_totals[stage] = stage_totals.get(stage, 0.0) + seconds
        frame["segmentation_warnings"] = result.diagnostics.warnings
        frame["detected_regions"] = len(result.regions)
        centers = np.array([(r.coarse.centroid_y, r.coarse.centroid_x) if r.coarse else (np.nan, np.nan)
                            for r in result.regions]).reshape(-1, 2)
        shift = np.array([frame["dy_px"], frame["dx_px"]], dtype=float)
        matches = match_centroids(template, centers, shift, gate)
        frame["unmatched_regions"] = len(result.regions) - sum(m["match_status"] == "matched" for m in matches)
        for match in matches:
            index = match["region_index"]
            region = result.regions[index] if index is not None else None
            for method in (("coarse",) if otsu is not None else ("coarse", "refined")):
                shape = getattr(region, method) if region is not None else None
                status = match["match_status"]
                if status == "matched":
                    status = _measurement_status(region, method, segment_config, mask_only=otsu is not None)
                rows.append({"series": name, "frame": frame["index"], "order": frame["order"],
                             "timestamp_s": frame["timestamp_s"], "filename": frame["path"],
                             "hole": match["hole"], "method": method, "status": status,
                             "unit": "nm" if segment_config.input.pixel_size_nm else "px",
                             "clipped": frame.get("clipped", False), "cd": shape.equivalent_diameter_px * factor if shape else None,
                             "major_axis": shape.major_axis_px * factor if shape else None,
                             "minor_axis": shape.minor_axis_px * factor if shape else None,
                             "valid_fraction": region.valid_fraction if region else None})
        # Every local detection remains inspectable even without correspondence.
        matched = {m["region_index"]: m["hole"] for m in matches if m["region_index"] is not None}
        offset = np.array([crop[0], crop[2]]) if crop else np.zeros(2)
        strengths = {r["region_id"]: r for r in result.diagnostics.edge_strength_change.get("regions", [])}
        counts = {"detected": len(result.regions), "complete": 0, "refined": 0, "border": 0, "review": 0}
        for index, region in enumerate(result.regions):
            statuses = {method: _measurement_status(region, method, segment_config, mask_only=otsu is not None)
                        for method in ("coarse", "refined")}
            counts["complete"] += statuses["coarse"] == "valid"
            counts["refined"] += statuses["refined"] == "valid"
            counts["border"] += region.touches_border
            strength = strengths.get(index + 1)
            review = bool(strength and strength["change"] < -.02)
            counts["review"] += review
            measures = {}
            for method in ("coarse", "refined"):
                shape = getattr(region, method)
                measures[method] = {"ecd": shape.equivalent_diameter_px * factor,
                                    "area_px2": shape.area_px2} if shape else None
            refined = result.refined[index] if index < len(result.refined) else None
            contours.append({"series": name, "frame": frame["index"], "region_id": index + 1,
                             "hole": matched.get(index), "status": statuses, "measures": measures,
                             "review": review, "edge_strength": strength,
                             "refined_fraction": region.valid_fraction,
                             "coarse": (result.coarse[index].points + offset).tolist(),
                             "holes": [(h.points + offset).tolist() for h in result.holes[index]],
                             "refined": (refined.polygon + offset).tolist() if refined is not None else [],
                             "refined_valid": refined.valid.tolist() if refined is not None else []})
            if result.open_paths:
                contours[-1]["open_paths"] = [(path + offset).tolist() for path in result.open_paths[index]]
        frame["contour_counts"] = counts
        frame["contour_status"] = "available" if counts["complete"] else "unavailable"
        frame["correspondence_status"] = ("unavailable" if not len(template) else
                                           "available" if matched else "no_matches")
        now = time.perf_counter()
        if number == 1 or number % 16 == 0 or number == count or now - last_progress >= 30:
            print(f"{label}: contours/CD {number}/{count}; {len(result.regions)} regions; "
                  f"{now - started:.1f}s elapsed ({(now - started) / number:.2f}s/image)", flush=True)
            last_progress = now
    elapsed = time.perf_counter() - started
    series.setdefault("timings_s", {})["contours_cd"] = elapsed
    series["segmentation_stage_totals_s"] = stage_totals
    print(f"{label}: contours/CD complete in {elapsed:.1f}s", flush=True)
    return rows, contours


def analyze_series(root: Path, name: str, series: dict, noise_config: AnalysisConfig,
                   *, raw_frames: list[dict] | None = None, difference_limit: float = 32.0,
                   device: str = "cpu") -> None:
    """Measure delivered pixels only; never call the acquisition-correction pipeline."""
    from sem_noise.comparison_metrics import difference_rgb, native_series_statistics

    started = time.perf_counter()
    print(f"{name}: measuring native uint8 brightness and temporal variation", flush=True)
    timings = series.setdefault("timings_s", {})
    label = Path(series["frames"][0]["path"]).parent.as_posix() if series["frames"] else name
    with Progress(f"{label}: native statistics on {device}", total=len(series["frames"]),
                  timings=timings, key="native_statistics") as progress:
        def images():
            for number, frame in enumerate(series["frames"], 1):
                yield read_uint8(root / frame["path"])
                # GPU work may be queued here; stage completion includes the download.
                progress.update(number, "frames submitted; final statistics may still be pending")

        rows, std = native_series_statistics(images(), device=device)
    for frame, row in zip(series["frames"], rows):
        frame.update(row)
    series["native"] = {"frames": len(rows), "temporal_rms_dn": float(np.sqrt(np.mean(std**2))) if std is not None else None,
                        "backend": "numpy" if device == "cpu" else "cupy", "device": device}
    folder = Path(series["frames"][0]["path"]).parent
    if std is not None:
        relative = folder / "temporal_std.npy"
        with Progress(f"{label}: saving temporal_std.npy", timings=timings, key="save_temporal_std"):
            np.save(root / relative, std.astype(np.float32), allow_pickle=False)
        series["temporal_std"] = relative.as_posix()
    if raw_frames is not None:
        if len(raw_frames) != len(series["frames"]):
            raise ValueError("model and raw acquisition counts differ")
        with Progress(f"{label}: output-minus-raw difference PNGs", total=len(raw_frames),
                      timings=timings, key="difference_pngs") as progress:
            for number, (frame, raw) in enumerate(zip(series["frames"], raw_frames), 1):
                if frame["index"] != raw["index"]:
                    raise ValueError("model and raw acquisition indices differ")
                output, source = read_uint8(root / frame["path"]), read_uint8(root / raw["path"])
                frame["brightness_delta_dn"] = float(output.mean() - source.mean())
                relative = folder / "differences" / Path(frame["path"]).name
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(difference_rgb(output, source, difference_limit)).save(root / relative)
                frame["difference_path"] = relative.as_posix()
                progress.update(number, relative.as_posix())
        series["difference_limit_dn"] = difference_limit
    for key in ("noise_report", "noise_settings", "noise", "noise_status"):
        series.pop(key, None)
    timings["native_analysis"] = time.perf_counter() - started
    print(f"{label}: native analysis complete in {timings['native_analysis']:.1f}s", flush=True)


def resolve_segmentation_settings(config: ComparisonSettings) -> SegmentConfig:
    """Resolve native-pixel metrology and keep the comparison on one GPU."""
    from sem_segment.config import Config as SegmentConfig, load_config as load_segment

    segment = load_segment(config.segmentation_config) if config.segmentation_config else SegmentConfig(
        segmentation={"backend": "classical", "polarity": "dark"})
    if config.metrology_device is not None:
        segment.refine = type(segment.refine).model_validate(
            {**segment.refine.model_dump(), "device": config.metrology_device})
    if segment.refine.device != "cpu":
        index = int(segment.refine.device.split(":")[1]) if ":" in segment.refine.device else 0
        device = f"cuda:{index}"

        def single_device(value: str, label: str) -> str:
            if value in ("auto", "cuda"):
                return device
            if value.startswith("cuda:") and int(value.split(":")[1]) != index:
                raise ValueError(f"Single-GPU comparison requires {label} and refinement on {device}; got {value}")
            return value

        # Resolve automatic devices together; reject explicit conflicting GPU
        # choices rather than silently using a second GPU or overriding them.
        config.device = single_device(config.device, "denoiser")
        if config.contour_method == "current" and segment.needs_model_weights:
            segment.segmentation.device = single_device(segment.segmentation.device, "mask model")
    segment.input.black_level, segment.input.white_level = 0.0, 255.0
    segment.segmentation.contrast_stretch = None
    segment.masks.include_border_regions = False
    if config.pixel_size_nm is not None:
        segment.input.pixel_size_nm = config.pixel_size_nm
    return segment


def prepare_reference(root: Path, site: dict, segment: SegmentConfig, *,
                      contour_method: str = "current", otsu: OtsuSettings | None = None,
                      analysis_memory_mb: int = 8192, io_workers: int = 2) -> tuple[np.ndarray, np.ndarray]:
    from sem_segment.pipeline import segment_image
    from sem_segment.repeatability import correspondence_gate

    reference = read_uint8(root / site["full_average"])
    site["image_shape"] = list(reference.shape)
    crop = segment.input.crop
    if crop and (crop[1] > reference.shape[0] or crop[3] > reference.shape[1]):
        raise ValueError("Segmentation crop extends outside the saved image")
    template_image = reference[crop[0]:crop[1], crop[2]:crop[3]] if crop else reference
    print(f"{site['name']}: locating holes on the saved uint8 full average", flush=True)
    if contour_method == "otsu":
        from sem_segment.otsu_measurement import measure_saved_otsu

        result = measure_saved_otsu(root / site["full_average"], otsu, crop=crop,
                                    device=segment.refine.device, metrology=segment.metrology,
                                    memory_mb=analysis_memory_mb, io_workers=io_workers)
    elif contour_method == "current":
        result = segment_image(template_image.astype(float) / 255, segment)
    else:
        raise ValueError("contour_method must be current or otsu")
    template = np.array([(r.coarse.centroid_y, r.coarse.centroid_x) for r in result.regions
                         if r.coarse and not r.touches_border]).reshape(-1, 2)
    site["template_centroids"] = template.tolist()
    site["correspondence_reference"] = "average128"
    site["match_gate_px"] = correspondence_gate(template, site.get("requested_match_gate_px"))
    site["warnings"] = ([] if len(template) else
                        ["Hole correspondence unavailable: no complete holes detected in the full average. Local contours remain inspectable."])
    raw = site["series"]["raw"]["frames"]
    stamps = [f["timestamp_s"] for f in raw]
    site["series"]["average128"] = {"step": 0, "frames": [{
        "index": 1, "order": 64.5, "first_acquisition": 1, "last_acquisition": 128,
        "timestamp_s": float(np.mean(stamps)) if all(t is not None for t in stamps) else None,
        "path": site["full_average"], "clipped": False}]}
    return reference, template


def save_record(root: Path, record: dict) -> None:
    from sem_noise.pipeline import write_csv, write_json
    from sem_noise.comparison_storage import checkpoint_contours, write_record

    for site in record["sites"]:
        checkpoint_contours(root, site)
    # Running snapshots reference immutable parts; complete reports keep their
    # existing site-wide contours_path. Never re-encode old coordinates here.
    sites = [{k: v for k, v in site.items() if k != "contours" and
              (k != "contours_parts" or "contours_path" not in site)}
             for site in record["sites"]]
    timings = record.setdefault("timings_s", {})
    with Progress("Saving comparison.json (serialize/write; 0 embedded contours)",
                  timings=timings, key="save_comparison_json"):
        write_record(root / "comparison.json", {**record, "sites": sites})
    print(f"Saved comparison.json: {(root / 'comparison.json').stat().st_size / 1024**2:.1f} MiB", flush=True)
    with Progress(f"Saving prediction_ranges.csv ({len(record['prediction_ranges']):,} rows)",
                  timings=timings, key="save_prediction_ranges_csv"):
        write_csv(root / "prediction_ranges.csv", record["prediction_ranges"])
    # A small sidecar includes the just-finished save, without rewriting the
    # large comparison a second time to store the duration of its own write.
    write_json(root / "timings.json", {
        "note": "Wall seconds; nested stages overlap. Save times accumulate. Render-only series analysis timings come from the source report.",
        "timings_s": timings,
        "sites": {site["name"]: {
            "timings_s": site.get("timings_s", {}),
            "series": {name: {"timings_s": series.get("timings_s", {}),
                              "report_timings_s": series.get("report_timings_s", {})}
                       for name, series in site.get("series", {}).items()}}
                  for site in record["sites"]}})


def finish_comparison(root: Path, record: dict, *, reuse_contours: bool = False) -> None:
    from sem_noise.pipeline import write_csv
    from sem_noise.comparison_report import render_comparison, write_tensorboard
    from sem_noise.comparison_storage import finish_contours
    from sem_segment.repeatability import summarize_observations

    started_finalization = time.perf_counter()
    print("Finalizing comparison: summaries and exports before report rendering", flush=True)
    for site in record["sites"]:
        timings = site.setdefault("timings_s", {})
        names = [name for name in site["series"] if name != "average128"]
        models = [name for name in record["models"] if name in names]
        with Progress(f"{site['name']}: repeatability summary ({len(site['observations']):,} observations)",
                      timings=timings, key="repeatability_summary"):
            holes, summaries = summarize_observations(site["observations"], names, comparison_series=models)
        site.update(per_hole=holes, repeatability=summaries)
        for filename, rows in (("observations", site["observations"]), ("per_hole", holes), ("repeatability", summaries)):
            with Progress(f"{site['name']}/{filename}.csv: serialize/write ({len(rows):,} rows)",
                          timings=timings, key=f"export_{filename}_csv"):
                write_csv(root / site["name"] / f"{filename}.csv", rows)
        if reuse_contours and "contours_path" in site:
            print(f"{site['contours_path']}: reusing saved contour export", flush=True)
        else:
            site.pop("contours_path", None)
            finish_contours(root, site)
        print(f"Saved {site['contours_path']}: {(root / site['contours_path']).stat().st_size / 1024**2:.1f} MiB", flush=True)
        with Progress(f"{site['name']}/frames.csv: serialize/write", timings=timings, key="export_frames_csv"):
            write_csv(root / site["name"] / "frames.csv", [dict(series=name, **frame)
                      for name, series in site["series"].items() for frame in series["frames"]])
        frames = [f for name in names for f in site["series"][name]["frames"]]
        available = sum(f.get("contour_status") == "available" for f in frames)
        site["contour_status"] = "available" if available == len(frames) else "partial" if available else "unavailable"
    record.update(schema_version=3, status="complete", measurement_source="saved uint8 RGB, identical channels")
    print("Rendering interactive comparison report", flush=True)
    timings = record.setdefault("timings_s", {})
    with Progress("Interactive comparison report", timings=timings, key="comparison_report"):
        render_comparison(root, record)
    if record.get("settings", {}).get("tensorboard", True):
        print("Combined comparison report finished; writing TensorBoard images", flush=True)
        with Progress("TensorBoard comparison", timings=timings, key="tensorboard"):
            write_tensorboard(root, record)
        print("TensorBoard comparison finished", flush=True)
    else:
        record["timings_s"].pop("tensorboard", None)
        print("Combined comparison report finished; TensorBoard disabled", flush=True)
    timings["finalization_before_record_save"] = time.perf_counter() - started_finalization
    save_record(root, record)
    print(f"Comparison finalization complete in {time.perf_counter() - started_finalization:.1f}s", flush=True)


def run(config: ComparisonSettings) -> dict:
    from burst_diffusion.data import content_key
    from edge_denoise.infer import Denoiser
    from sem_noise.config import AnalysisConfig, load_config as load_noise
    from sem_noise.io import file_hash, pixel_hash

    with Progress("Validating inputs (decode every source frame)") as validation:
        files_by_site = validate_inputs(config)
    noise = load_noise(config.analysis_config) if config.analysis_config else AnalysisConfig()
    noise = replace(noise, registration="none", compare_direct_registration="none")
    segment = resolve_segmentation_settings(config)
    if config.contour_method == "current" and segment.refine.enabled and segment.refine.device != "cpu":
        from sem_segment.cuda import CudaRefiner

        print(f"Checking CUDA contour refinement on {segment.refine.device}", flush=True)
        with CudaRefiner(segment.refine) as refiner:
            print(f"CUDA refinement ready: {refiner.describe()}", flush=True)
    if config.pixel_size_nm is not None:
        noise = replace(noise, pixel_size_nm=config.pixel_size_nm)
    if config.frame_interval_s is not None:
        noise = replace(noise, frame_interval_s=config.frame_interval_s)
    root = config.output_dir
    root.mkdir(parents=True)
    record = {"schema_version": 3, "study": "real N2N registration/brightness comparison", "status": "running", "settings": config.model_dump(mode="json"),
              "segmentation_settings": segment.model_dump(mode="json"), "noise_settings": asdict(noise),
              "unit": "nm" if segment.input.pixel_size_nm else "px", "range_warning": RANGE_WARNING,
              "models": {}, "sites": [], "prediction_ranges": [], "artifacts": [], "warnings": []}
    record.update(contour_method=config.contour_method, otsu_settings=config.otsu.model_dump())
    record["timings_s"] = {"validate_inputs": validation.elapsed_s}
    contour_options = {"contour_method": "otsu", "otsu": config.otsu} if config.contour_method == "otsu" else {}
    execution_options = ({key: getattr(config, key) for key in ("analysis_batch", "analysis_memory_mb", "io_workers")}
                         if config.contour_method == "otsu" else {})
    native_options = {"device": segment.refine.device} if config.contour_method == "otsu" else {}

    def save() -> None:
        save_record(root, record)

    save()
    try:
        for name, files in files_by_site.items():
            print(f"Preparing {name}: 128 raw frames and 16 averages", flush=True)
            site = {"name": name, "series": {}, "warnings": [], "observations": [], "contours": []}
            record["sites"].append(site)
            timestamps = config.sites[name].timestamps_s
            if timestamps is None and noise.frame_interval_s is not None:
                timestamps = (np.arange(128) * noise.frame_interval_s).tolist()
            raw_frames, blocks, total = [], [], None
            with Progress(f"{name}: saving raw images and averages", total=len(files),
                          timings=site.setdefault("timings_s", {}), key="prepare_saved_images") as progress:
                for i, source in enumerate(files):
                    pixels = read_uint8(source)
                    if total is None:
                        total = np.zeros(pixels.shape, dtype=np.float64)
                    total += pixels
                    path = Path(name) / "raw" / f"frame_{i + 1:03d}.png"
                    save_rgb(root / path, pixels)
                    raw_frames.append({"index": i + 1, "order": i + 1, "timestamp_s": timestamps[i] if timestamps else None,
                                       "path": path.as_posix(), "source": str(source), "source_sha256": file_hash(source),
                                       "pixel_sha256": pixel_hash(pixels), "content_sha256": content_key(pixels), "clipped": False})
                    blocks.append(pixels)
                    if len(blocks) == 8:
                        block = i // 8 + 1
                        average_path = Path(name) / "average8" / f"block_{block:02d}_{i - 6:03d}-{i + 1:03d}.png"
                        save_rgb(root / average_path, average_uint8(np.stack(blocks)))
                        series = site["series"].setdefault("average8", {"frames": [], "step": 0})
                        series["frames"].append({"index": block, "order": i - 2.5, "first_acquisition": i - 6,
                                                 "last_acquisition": i + 1, "path": average_path.as_posix(),
                                                 "timestamp_s": float(np.mean(timestamps[i-7:i+1])) if timestamps else None,
                                                 "clipped": False})
                        blocks.clear()
                    progress.update(i + 1, source.name)
            full = Path(name) / "visual_reference_only" / "full_average.png"
            save_rgb(root / full, np.rint(total / 128).astype(np.uint8))
            site["full_average"] = full.as_posix()
            site["series"] = {"raw": {"frames": raw_frames, "step": 0}, **site["series"]}
            site["requested_match_gate_px"] = config.match_gate_px
            reference, template = prepare_reference(root, site, segment, **contour_options,
                **{key: value for key, value in execution_options.items() if key != "analysis_batch"})
            for series_name, series in site["series"].items():
                started_registration = time.perf_counter()
                tracks = registration_tracks(reference, [root / f["path"] for f in series["frames"]], noise.registration_sigma)
                for frame, track in zip(series["frames"], tracks):
                    frame.update(track)
                series.setdefault("timings_s", {})["translation_diagnostics"] = time.perf_counter() - started_registration
                analyze_series(root, series_name, series, noise, **native_options)
                observations, contours = measure_series(root, series_name, series, template, site["match_gate_px"], segment,
                                                        **contour_options, **execution_options)
                site["observations"].extend(observations)
                site["contours"].extend(contours)
            save()

        # One model resident at a time, across all requested sites.
        for model_name, arm in config.checkpoints.items():
            print(f"Loading {model_name}", flush=True)
            checkpoint = arm.checkpoint
            with Progress(f"{model_name}: checkpoint hash, weights and training metadata",
                          timings=record.setdefault("timings_s", {}), key="load_models"):
                checkpoint_hash = file_hash(checkpoint)
                denoiser = Denoiser.from_checkpoint(checkpoint, device=config.device, use_ema=config.ema)
                metadata = checkpoint_arm_metadata(arm, denoiser)
                model_config = denoiser.config.model_dump(mode="json")
                validate_comparable_arm(metadata, model_config, record["models"])
                forbidden = set(metadata.pop("train_val_content_hashes"))
                if any(f["content_sha256"] in forbidden for site in record["sites"] for f in site["series"]["raw"]["frames"]):
                    raise ValueError("A comparison test acquisition appears in this checkpoint's train/validation sites")
                record["warnings"].extend(f"{model_name}: {warning}" for warning in metadata["warnings"])
                black, white = denoiser.config.data.black_level, denoiser.config.data.white_level
                if black is None and white is None:
                    black, white = 0.0, 255.0
                prediction_uint8(np.zeros((1, 1)), black, white)  # Validate fixed levels.
            model_record = {"checkpoint": str(checkpoint), "sha256": checkpoint_hash,
                            "step": denoiser.checkpoint_step, "black_level": black, "white_level": white,
                            "config": model_config, "arm": metadata,
                            "ema": getattr(denoiser, "using_ema", config.ema),
                            "margin": denoiser.default_margin,
                            "stride": min(48, denoiser.image_size) if denoiser.image_size <= 64 else denoiser.image_size // 2}
            model_record["stride"] = max(1, min(model_record["stride"], denoiser.image_size - 2 * denoiser.default_margin))
            record["models"][model_name] = model_record
            for site in record["sites"]:
                series = {"frames": [], "step": denoiser.checkpoint_step}
                site["series"][model_name] = series
                with Progress(f"{site['name']}/{model_name}: inference and saved PNG export", total=128,
                              timings=series.setdefault("timings_s", {}), key="inference_and_pngs") as progress:
                    for raw in site["series"]["raw"]["frames"]:
                        path = Path(site["name"]) / model_name / Path(raw["path"]).name
                        audit = {"site": site["name"], "model": model_name, "filename": path.as_posix(), "status": "pending"}
                        record["prediction_ranges"].append(audit)
                        try:
                            pixels = read_uint8(root / raw["path"])
                            normalized = np.clip((pixels.astype(float) - black) / (white - black), 0, 1)
                            prediction = denoiser.denoise_full(normalized, stride=model_record["stride"],
                                                               tile_batch=config.tile_batch, clip_output=False)
                            if prediction.shape != pixels.shape:
                                raise ValueError("Prediction dimensions differ from the source")
                            quantized, stats = prediction_uint8(prediction, black, white)
                        except Exception as error:
                            audit.update(status="failed", error=str(error))
                            raise
                        audit.update(stats, status="complete")
                        save_rgb(root / path, quantized)
                        # Reuse RAW translations even if the prediction changes feature positions.
                        frame = {k: raw[k] for k in ("index", "order", "timestamp_s", "dy_px", "dx_px", "registration_status",
                                                     "registration_score", "registration_error")}
                        frame.update(path=path.as_posix(), **stats)
                        series["frames"].append(frame)
                        progress.update(len(series["frames"]), path.as_posix())
                reference = read_uint8(root / site["full_average"])
                output_registration_diagnostics(root, reference, series, noise.registration_sigma)
                analyze_series(root, model_name, series, noise, raw_frames=site["series"]["raw"]["frames"],
                               difference_limit=config.difference_limit_dn, **native_options)
                observations, contours = measure_series(root, model_name, series, np.array(site["template_centroids"]),
                                                        site["match_gate_px"], segment, **contour_options, **execution_options)
                site["observations"].extend(observations)
                site["contours"].extend(contours)
                save()
            with Progress(f"{model_name}: releasing denoiser"):
                del denoiser
        finish_comparison(root, record)
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        save()
    return record


def require_reusable_analysis(record: dict) -> None:
    """Contour-only experiments must never silently reuse incomplete analysis."""
    error = "--contours-only needs a complete saved analysis; rebuild once without this flag"
    if record.get("status") != "complete" or record.get("schema_version", 0) < 3:
        raise ValueError(error)
    for site in record["sites"]:
        average = site["series"].get("average128", {}).get("frames", [])
        if len(average) != 1 or average[0]["path"] != site["full_average"]:
            raise ValueError(error)
        for name, series in site["series"].items():
            frames = series["frames"]
            if not frames or series.get("native", {}).get("frames") != len(frames):
                raise ValueError(error)
            if len(frames) > 1 and not series.get("temporal_std"):
                raise ValueError(error)
            keys = {"mean_dn", "minimum_saved_dn", "maximum_saved_dn", "dy_px", "dx_px",
                    "registration_status", "registration_score", "registration_error"}
            if name in record["models"]:
                keys |= {"brightness_delta_dn", "difference_path", "output_dy_px", "output_dx_px",
                         "output_registration_status", "output_registration_score", "output_registration_error",
                         "output_minus_raw_dy_px", "output_minus_raw_dx_px"}
            if any(not keys <= frame.keys() for frame in frames):
                raise ValueError(error)


def rebuild(record_path: Path, output_dir: Path, *, metrology_device: str | None = None,
            segmentation_config: Path | None = None, contour_method: str | None = None,
            otsu_config: Path | None = None, render_only: bool = False, contours_only: bool = False,
            analysis_batch: int | None = None, analysis_memory_mb: int | None = None,
            io_workers: int | None = None, tensorboard: bool | None = None) -> dict:
    """Rebuild from saved uint8 images. Never load a denoiser or the raw dataset."""
    from sem_noise.config import AnalysisConfig
    from sem_noise.comparison_storage import load_contours
    from sem_segment.config import Config as SegmentConfig, load_config as load_segment

    record_path = record_path.resolve()
    source_root, root = record_path.parent, output_dir.resolve()
    with Progress(f"Loading saved comparison {record_path}") as loading:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    record["timings_s"] = {"load_comparison_json": loading.elapsed_s}
    if contours_only and render_only:
        raise ValueError("--contours-only and --render-only are alternative rebuild modes")
    if contours_only:
        require_reusable_analysis(record)
    execution_options = {}
    for key, override, default in (("analysis_batch", analysis_batch, 16),
                                   ("analysis_memory_mb", analysis_memory_mb, 8192), ("io_workers", io_workers, 2)):
        value = override if override is not None else record["settings"].get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        execution_options[key] = value
    if render_only and record.get("schema_version", 0) < 3:
        raise ValueError("This older report needs contour remeasurement; omit --render-only")
    if render_only and any(value is not None for value in (
            metrology_device, segmentation_config, contour_method, otsu_config,
            analysis_batch, analysis_memory_mb, io_workers)):
        raise ValueError("--render-only reuses measurements; omit segmentation/device overrides")
    method = contour_method or record.get("contour_method", record.get("settings", {}).get("contour_method", "current"))
    if method not in {"current", "otsu"}:
        raise ValueError("contour_method must be current or otsu")
    if otsu_config is not None and method != "otsu":
        raise ValueError("--otsu-config requires --contour-method otsu")
    otsu = (load_otsu_settings(otsu_config) if otsu_config is not None else
            OtsuSettings.model_validate(record.get("otsu_settings", record.get("settings", {}).get("otsu", {}))))
    contour_options = {"contour_method": "otsu", "otsu": otsu} if method == "otsu" else {}
    if root == source_root and not render_only:
        raise ValueError("Use a new --output-dir for remeasurement; saved images are preserved")
    if root != source_root:
        if root.exists() or source_root in root.parents or root in source_root.parents:
            raise ValueError("Rebuild output must be a new directory outside the original report")
    paths = set()

    def source(relative: str) -> Path:
        path = (source_root / relative).resolve()
        if path == source_root or source_root not in path.parents:
            raise ValueError(f"Report asset must be inside its directory: {relative}")
        return path

    for site in record["sites"]:
        site["timings_s"] = {}
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", site["name"]):
            raise ValueError("Invalid saved site name")
        paths.add(site["full_average"])
        if render_only:
            with Progress(f"{site['name']}: loading saved contours", timings=site["timings_s"], key="load_contours_json"):
                site["contours"] = load_contours(source_root, site)
            if "contours_path" in site:
                paths.add(site["contours_path"])
        else:
            site["contours"] = []
            site.pop("contours_path", None)
        # Old parts belong to the source bundle; new measurements start afresh.
        site.pop("contours_parts", None)
        for name, series in site["series"].items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
                raise ValueError("Invalid saved series name")
            for frame in series["frames"]:
                paths.add(frame["path"])
                if (render_only or contours_only) and frame.get("difference_path"):
                    paths.add(frame["difference_path"])
            if (render_only or contours_only) and series.get("temporal_std"):
                paths.add(series["temporal_std"])
    with Progress("Checking saved comparison assets", total=len(paths)) as progress:
        for number, relative in enumerate(paths, 1):
            if not source(relative).is_file():
                raise ValueError(f"Missing saved image/measurement: {relative}")
            progress.update(number, relative)
    if root != source_root:
        root.mkdir(parents=True)
        with Progress("Copying saved comparison assets", total=len(paths),
                      timings=record["timings_s"], key="copy_saved_assets") as progress:
            copied_bytes = 0
            for number, relative in enumerate(sorted(paths), 1):
                destination = root / source(relative).relative_to(source_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source(relative), destination)
                copied_bytes += destination.stat().st_size
                progress.update(number, f"{copied_bytes / 1024**2:.1f} MiB copied; {relative}")
    record["rebuilt_from"] = str(record_path)
    record["settings"]["output_dir"] = str(root)
    if tensorboard is not None:
        record["settings"]["tensorboard"] = tensorboard
    if not render_only:
        record.update(contour_method=method, otsu_settings=otsu.model_dump())
        record["settings"].update(contour_method=method, otsu=otsu.model_dump())
        record["settings"].update(execution_options)
        segment = load_segment(segmentation_config) if segmentation_config else SegmentConfig.model_validate(record["segmentation_settings"])
        if metrology_device is not None:
            segment.refine = type(segment.refine).model_validate({**segment.refine.model_dump(), "device": metrology_device})
        segment.input.black_level, segment.input.white_level = 0., 255.
        segment.segmentation.contrast_stretch = None
        segment.masks.include_border_regions = False
        if segment.refine.device != "cpu" and segment.segmentation.device == "auto":
            segment.segmentation.device = segment.refine.device
        record["segmentation_settings"] = segment.model_dump(mode="json")
        record["unit"] = "nm" if segment.input.pixel_size_nm else "px"
        noise = AnalysisConfig(registration="none", registration_sigma=record.get("noise_settings", {}).get("registration_sigma", 1.0))
        if not contours_only:
            record["noise_settings"] = asdict(noise)
        native_options = {"device": segment.refine.device} if method == "otsu" else {}
        for site in record["sites"]:
            site.update(observations=[], contours=[])
            site.setdefault("requested_match_gate_px", record["settings"].get("match_gate_px"))
            previous_average = site["series"].get("average128")
            reference, template = prepare_reference(root, site, segment, **contour_options,
                **{key: value for key, value in execution_options.items() if key != "analysis_batch"})
            if contours_only:
                site["series"]["average128"] = previous_average
            # Model correspondence reuses freshly measured raw drift, regardless
            # of the key order in the saved JSON.
            names = ["raw", *(name for name in site["series"] if name != "raw")]
            for name in names:
                series = site["series"][name]
                series["timings_s"] = {}
                raw = site["series"]["raw"]["frames"] if name in record["models"] else None
                if contours_only:
                    series["analysis_reused_from"] = str(record_path)
                    print(f"{site['name']}/{name}: reusing saved brightness, variation and translation diagnostics", flush=True)
                elif raw is not None:
                    for frame, original in zip(series["frames"], raw):
                        for key in ("dy_px", "dx_px", "registration_status", "registration_score", "registration_error"):
                            frame[key] = original[key]
                    output_registration_diagnostics(root, reference, series, noise.registration_sigma)
                else:
                    started_registration = time.perf_counter()
                    for frame, track in zip(series["frames"], registration_tracks(reference, [root / f["path"] for f in series["frames"]], noise.registration_sigma)):
                        frame.update(track)
                    series["timings_s"]["translation_diagnostics"] = time.perf_counter() - started_registration
                if not contours_only:
                    series.pop("analysis_reused_from", None)
                    analyze_series(root, name, series, noise, raw_frames=raw,
                                   difference_limit=record["settings"].get("difference_limit_dn", 32.0), **native_options)
                observations, contours = measure_series(root, name, series, template, site["match_gate_px"], segment,
                                                        **contour_options, **execution_options)
                site["observations"].extend(observations)
                site["contours"].extend(contours)
    finish_comparison(root, record, reuse_contours=render_only)
    return record


def _assignments(values: list[str] | None) -> dict[str, Path]:
    result = {}
    for value in values or []:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError("Use a unique NAME=PATH for each override")
        result[name] = (ROOT / Path(path).expanduser()).resolve()
    return result


def load_otsu_settings(path: Path) -> OtsuSettings:
    """Read the same three detector settings used by the standalone preview."""
    return OtsuSettings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def configure_run(args: argparse.Namespace) -> ComparisonSettings:
    config = load_settings(args.config or ROOT / "edge_denoise/configs/sem_real_compare.yml")
    if args.site_dir:
        name = args.site or args.site_dir.name
        config.sites = {name: SiteSettings(source_dir=(ROOT / args.site_dir.expanduser()).resolve())}
    elif args.site:
        if args.site not in config.sites:
            raise ValueError(f"Unknown site: {args.site}")
        config.sites = {args.site: config.sites[args.site]}
    if args.model:
        unknown = set(args.model) - config.checkpoints.keys()
        if unknown:
            raise ValueError(f"Unknown models: {sorted(unknown)}")
        config.checkpoints = {name: arm for name, arm in config.checkpoints.items() if name in args.model}
    if args.experiment_prefix:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.experiment_prefix):
            raise ValueError("Experiment prefix must be a folder name")
        if Path(args.checkpoint_name).name != args.checkpoint_name:
            raise ValueError("Checkpoint name must be a filename")
        for arm in config.checkpoints.values():
            arm.checkpoint = (ROOT / args.runs_dir / f"{args.experiment_prefix}_{arm.registration}_{arm.brightness}" / args.checkpoint_name).resolve()
            arm.prepared_manifest = None  # Resolve from the selected checkpoint, not an old example path.
    for option, field in ((args.checkpoint, "checkpoint"), (args.prepared_manifest, "prepared_manifest")):
        for name, path in _assignments(option).items():
            if name not in config.checkpoints:
                raise ValueError(f"Unknown model override: {name}")
            setattr(config.checkpoints[name], field, path)
    for field in ("output_dir", "segmentation_config"):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, (ROOT / value.expanduser()).resolve())
    for field in ("metrology_device", "device", "tile_batch", "difference_limit_dn", "contour_method",
                  "analysis_batch", "analysis_memory_mb", "io_workers", "tensorboard"):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    if args.otsu_config is not None:
        if config.contour_method != "otsu":
            raise ValueError("--otsu-config requires --contour-method otsu")
        config.otsu = load_otsu_settings(args.otsu_config)
    return ComparisonSettings.model_validate(config.model_dump())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="Base YAML; defaults to edge_denoise/configs/sem_real_compare.yml")
    source.add_argument("--from-comparison", type=Path, help="Rebuild from comparison.json and saved uint8 images; no inference")
    parser.add_argument("--render-only", action="store_true", help="Reuse v3 measurements instead of remeasuring contours")
    parser.add_argument("--contours-only", action="store_true", help="Remeasure contours; reuse saved brightness/noise/drift analysis")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--site-dir", type=Path, help="Override with one remote acquisition folder")
    parser.add_argument("--site", help="Pilot only this named site from the YAML")
    parser.add_argument("--model", action="append", help="Include this arm only; repeat for multiple arms")
    parser.add_argument("--experiment-prefix", help="For example 260921_real_n2n; appends _REGISTRATION_BRIGHTNESS")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/edge_denoise"))
    parser.add_argument("--checkpoint-name", default="ckpt_latest.pt")
    parser.add_argument("--checkpoint", action="append", metavar="NAME=PATH", help="Override one checkpoint, including older baselines")
    parser.add_argument("--prepared-manifest", action="append", metavar="NAME=PATH")
    parser.add_argument("--segmentation-config", type=Path)
    parser.add_argument("--contour-method", choices=("otsu", "current"),
                        help="Use Gaussian + Otsu masks, or the existing segmentation/refinement method")
    parser.add_argument("--otsu-config", type=Path,
                        help="Override Otsu polarity, sigma_px and min_area_px with a detector YAML")
    parser.add_argument("--metrology-device", help="Otsu/native analysis or current-method refinement device: cpu, cuda, or cuda:N")
    parser.add_argument("--device", help="Denoiser device")
    parser.add_argument("--tile-batch", type=int)
    parser.add_argument("--analysis-batch", type=int, help="Maximum images per CUDA Otsu batch (default 16)")
    parser.add_argument("--analysis-memory-mb", type=int, help="Estimated CUDA batch working-set budget in MiB (default 8192)")
    parser.add_argument("--io-workers", type=int, help="Saved-image decoder workers (default 2)")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=None,
                        help="Write TensorBoard images as well as the HTML report; --no-tensorboard speeds visual iterations")
    parser.add_argument("--difference-limit-dn", type=float)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        if args.from_comparison:
            if not args.output_dir:
                raise ValueError("--from-comparison requires --output-dir (may be the original directory with --render-only)")
            if any((args.site_dir, args.site, args.model, args.experiment_prefix, args.checkpoint,
                    args.prepared_manifest, args.device, args.tile_batch, args.difference_limit_dn)):
                raise ValueError("Rebuild uses saved images and settings; omit inference/site overrides")
            record = rebuild(args.from_comparison, args.output_dir, metrology_device=args.metrology_device,
                             segmentation_config=args.segmentation_config, contour_method=args.contour_method,
                             otsu_config=args.otsu_config, render_only=args.render_only, contours_only=args.contours_only,
                             analysis_batch=args.analysis_batch, analysis_memory_mb=args.analysis_memory_mb,
                             io_workers=args.io_workers, tensorboard=args.tensorboard)
            output = args.output_dir
        else:
            if args.render_only or args.contours_only:
                raise ValueError("--render-only/--contours-only require --from-comparison")
            config = configure_run(args)
            record = run(config)
            output = config.output_dir
        print(f"Comparison: {output / 'index.html'} ({record['status']}); workflow {time.perf_counter() - started:.1f}s", flush=True)
        return 0 if record["status"] == "complete" else 1
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Error: {error} (workflow {time.perf_counter() - started:.1f}s)", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
