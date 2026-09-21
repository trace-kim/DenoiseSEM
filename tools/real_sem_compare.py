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
import sys
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
    device: str = "auto"
    tile_batch: int = Field(default=4, ge=1)
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
    if {name.casefold() for name in config.checkpoints} & {"raw", "average8"}:
        raise ValueError("raw and average8 are reserved series names")
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
    for path in paths:
        pixels = read_uint8(path)
        row = {"mean_dn": float(pixels.mean()), "p10_dn": float(np.percentile(pixels, 10)),
               "p90_dn": float(np.percentile(pixels, 90))}
        try:
            matrix, score = estimate_geometry(reference, pixels, motion="translation", sigma=sigma)
            row.update(dy_px=float(matrix[1, 2]), dx_px=float(matrix[0, 2]),
                       registration_status="registered", registration_score=score, registration_error="")
        except ValueError as error:
            row.update(dy_px=None, dx_px=None, registration_status="failed",
                       registration_score=None, registration_error=str(error))
        tracks.append(row)
    return tracks


def output_registration_diagnostics(root: Path, reference: np.ndarray, series: dict, sigma: float) -> None:
    """Measure each output's drift independently, without changing correspondence."""
    tracks = registration_tracks(reference, [root / f["path"] for f in series["frames"]], sigma)
    for frame, track in zip(series["frames"], tracks):
        for key in ("dy_px", "dx_px", "registration_status", "registration_score", "registration_error"):
            frame[f"output_{key}"] = track[key]
        valid = frame["registration_status"] == "registered" and track["registration_status"] == "registered"
        for component in ("dy_px", "dx_px"):
            frame[f"output_minus_raw_{component}"] = track[component] - frame[component] if valid else None


def measure_series(root: Path, name: str, series: dict, template: np.ndarray,
                   gate: float, segment_config: SegmentConfig) -> tuple[list[dict], list[dict]]:
    from sem_segment.pipeline import segment_image
    from sem_segment.repeatability import match_centroids

    rows, contours = [], []
    factor = segment_config.input.pixel_size_nm or 1.0
    for frame in series["frames"]:
        # Measurements always start by decoding the saved, quantized file.
        pixels = read_uint8(root / frame["path"])
        crop = segment_config.input.crop
        if crop:
            y0, y1, x0, x1 = crop
            pixels = pixels[y0:y1, x0:x1]
        result = segment_image(pixels.astype(np.float64) / 255.0, segment_config)
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
            for method in ("coarse", "refined"):
                shape = getattr(region, method) if region is not None else None
                status = match["match_status"]
                if status == "matched":
                    status = "valid"
                    if region.touches_border and not segment_config.masks.include_border_regions:
                        status = "border"
                    elif shape is None:
                        status = "measurement_failed"
                    elif method == "refined" and region.valid_fraction < segment_config.refine.min_valid_fraction:
                        status = "insufficient_refined_vertices"
                    elif not np.isfinite([shape.equivalent_diameter_px, shape.major_axis_px, shape.minor_axis_px]).all():
                        status = "nonfinite_measurement"
                rows.append({"series": name, "frame": frame["index"], "order": frame["order"],
                             "timestamp_s": frame["timestamp_s"], "filename": frame["path"],
                             "hole": match["hole"], "method": method, "status": status,
                             "unit": "nm" if segment_config.input.pixel_size_nm else "px",
                             "clipped": frame.get("clipped", False), "cd": shape.equivalent_diameter_px * factor if shape else None,
                             "major_axis": shape.major_axis_px * factor if shape else None,
                             "minor_axis": shape.minor_axis_px * factor if shape else None,
                             "valid_fraction": region.valid_fraction if region else None})
            # Coordinates for four fixed overlay crops; dimensions retain all holes.
            if index is not None and match["hole"] <= 4:
                contours.append({"series": name, "frame": frame["index"], "hole": match["hole"],
                                 "shift_yx": shift.tolist(),
                                 "coarse": result.coarse[index].points.tolist(),
                                 "refined": result.refined[index].polygon.tolist() if result.refined else []})
    return rows, contours


def analyze_series(root: Path, name: str, series: dict, noise_config: AnalysisConfig) -> None:
    from sem_noise.pipeline import analyze_dataset, write_csv

    folder = root / Path(series["frames"][0]["path"]).parent
    manifest = folder.parent / f"{name}_manifest.csv"
    write_csv(manifest, [{"site": name, "path": Path(f["path"]).name, "frame_index": i,
                          "timestamp_s": f["timestamp_s"]} for i, f in enumerate(series["frames"])])
    noise = replace(noise_config, expected_frames=len(series["frames"]), exclude_duplicates=False,
                    frame_interval_s=None if series["frames"][0]["timestamp_s"] is not None else noise_config.frame_interval_s)
    report = folder.parent / f"noise_{name}"
    result = analyze_dataset(folder, report, config=noise, manifest=manifest,
                             progress=lambda message: print(message, flush=True))
    series["noise_report"] = (report / "index.html").relative_to(root).as_posix()
    series["noise_settings"] = asdict(noise)
    series["noise"] = result["sites"][0]
    series["noise_status"] = result["status"]


def run(config: ComparisonSettings) -> dict:
    from burst_diffusion.data import content_key
    from edge_denoise.infer import Denoiser
    from sem_noise.config import AnalysisConfig, load_config as load_noise
    from sem_noise.io import file_hash, pixel_hash
    from sem_noise.pipeline import write_csv, write_json
    from sem_noise.comparison_report import render_comparison, write_tensorboard
    from sem_segment.config import Config as SegmentConfig, load_config as load_segment
    from sem_segment.pipeline import segment_image
    from sem_segment.repeatability import correspondence_gate, summarize_observations

    files_by_site = validate_inputs(config)
    noise = load_noise(config.analysis_config) if config.analysis_config else AnalysisConfig()
    if noise.min_frames > 16:
        raise ValueError("Noise min_frames must be <= 16 for the eight-frame averages")
    segment = load_segment(config.segmentation_config) if config.segmentation_config else SegmentConfig(
        segmentation={"backend": "classical", "polarity": "dark"})
    # Segment's measurement loader settings cannot stretch DN in this workflow.
    segment.input.black_level, segment.input.white_level = 0.0, 255.0
    if config.pixel_size_nm is not None:
        segment.input.pixel_size_nm = config.pixel_size_nm
        noise = replace(noise, pixel_size_nm=config.pixel_size_nm)
    if config.frame_interval_s is not None:
        noise = replace(noise, frame_interval_s=config.frame_interval_s)
    root = config.output_dir
    root.mkdir(parents=True)
    record = {"schema_version": 2, "study": "real N2N registration/brightness comparison", "status": "running", "settings": config.model_dump(mode="json"),
              "segmentation_settings": segment.model_dump(mode="json"), "noise_settings": asdict(noise),
              "unit": "nm" if segment.input.pixel_size_nm else "px", "range_warning": RANGE_WARNING,
              "models": {}, "sites": [], "prediction_ranges": [], "artifacts": [], "warnings": []}

    def save() -> None:
        write_json(root / "comparison.json", record)
        write_csv(root / "prediction_ranges.csv", record["prediction_ranges"])

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
            full = Path(name) / "visual_reference_only" / "full_average.png"
            save_rgb(root / full, np.rint(total / 128).astype(np.uint8))
            site["full_average"] = full.as_posix()  # Deliberately outside every quantitative series.
            site["series"] = {"raw": {"frames": raw_frames, "step": 0}, **site["series"]}
            reference = read_uint8(root / site["series"]["average8"]["frames"][0]["path"])
            if segment.input.crop:
                y0, y1, x0, x1 = segment.input.crop
                if y1 > reference.shape[0] or x1 > reference.shape[1]:
                    raise ValueError("Segmentation crop extends outside the saved image")
                template_image = reference[y0:y1, x0:x1]
            else:
                template_image = reference
            template_result = segment_image(template_image.astype(float) / 255, segment)
            template = np.array([(r.coarse.centroid_y, r.coarse.centroid_x) for r in template_result.regions
                                 if r.coarse and (not r.touches_border or segment.masks.include_border_regions)]).reshape(-1, 2)
            gate = correspondence_gate(template, config.match_gate_px)
            site.update(template_centroids=template.tolist(), match_gate_px=gate)
            if not len(template):
                site["warnings"].append("No template holes detected; repeatability is unavailable.")
            elif len(template) == 1:
                site["warnings"].append("One template hole: no neighbour spacing exists; inspect the recorded matching gate.")
            for series_name, series in site["series"].items():
                tracks = registration_tracks(reference, [root / f["path"] for f in series["frames"]], noise.registration_sigma)
                for frame, track in zip(series["frames"], tracks):
                    frame.update(track)
                analyze_series(root, series_name, series, noise)
                observations, contours = measure_series(root, series_name, series, template, gate, segment)
                site["observations"].extend(observations)
                site["contours"].extend(contours)
            save()

        # One model resident at a time, across all requested sites.
        for model_name, arm in config.checkpoints.items():
            print(f"Loading {model_name}", flush=True)
            checkpoint = arm.checkpoint
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
                    frame.update(path=path.as_posix(), **stats, mean_dn=float(quantized.mean()),
                                 p10_dn=float(np.percentile(quantized, 10)), p90_dn=float(np.percentile(quantized, 90)))
                    series["frames"].append(frame)
                    if raw["index"] % 16 == 0:
                        print(f"{site['name']}/{model_name}: {raw['index']}/128", flush=True)
                reference = read_uint8(root / site["series"]["average8"]["frames"][0]["path"])
                output_registration_diagnostics(root, reference, series, noise.registration_sigma)
                analyze_series(root, model_name, series, noise)
                observations, contours = measure_series(root, model_name, series, np.array(site["template_centroids"]),
                                                        site["match_gate_px"], segment)
                site["observations"].extend(observations)
                site["contours"].extend(contours)
                save()
            del denoiser
        for site in record["sites"]:
            holes, summaries = summarize_observations(site["observations"], list(site["series"]))
            site.update(per_hole=holes, repeatability=summaries)
            for filename, rows in (("observations", site["observations"]), ("per_hole", holes), ("repeatability", summaries)):
                write_csv(root / site["name"] / f"{filename}.csv", rows)
            write_json(root / site["name"] / "contours.json", site["contours"])
            write_csv(root / site["name"] / "frames.csv", [dict(series=name, **frame)
                      for name, series in site["series"].items() for frame in series["frames"]])
        record["status"] = "complete" if all(s["noise_status"] == "complete" for site in record["sites"]
                                             for s in site["series"].values()) else "partial_failure"
        render_comparison(root, record)
        if record["status"] == "complete":
            write_tensorboard(root, record)
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        save()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--site", help="Pilot only this named site from the YAML")
    args = parser.parse_args()
    try:
        config = load_settings(args.config)
        if args.site:
            if args.site not in config.sites:
                raise ValueError(f"Unknown site: {args.site}")
            config.sites = {args.site: config.sites[args.site]}
        record = run(config)
        print(f"Comparison: {config.output_dir / 'index.html'} ({record['status']})")
        return 0 if record["status"] == "complete" else 1
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
