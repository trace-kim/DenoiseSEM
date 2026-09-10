"""Reference-based evaluation on real held-out sites and disjoint frame pools.

The reference is a registered average, never called clean ground truth.
Measurement boxes and output coordinate transforms are fixed across models.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from burst_diffusion.data import BurstCache, BurstSource
from burst_diffusion.metrics import psnr, ssim
from burst_diffusion.real_data import cache_path, file_digest, normalize_native
from burst_diffusion.repeatability import find_cd_sites, measure_site

from .config import Config
from .fusion import warp_prediction
from .gradient import sobel
from .infer import Denoiser
from .real_data import LOSS_MARGIN, fixed_windows

logger = logging.getLogger("edge_denoise.real_evaluate")


def frame_pools(count: int, inputs: int, seed: int) -> tuple[list[int], list[int]]:
    if inputs < 2 or count - inputs < 2:
        raise ValueError("need at least two evaluation input frames and two disjoint reference frames per site")
    order = np.random.default_rng(seed).permutation(count)
    return sorted(order[:inputs].tolist()), sorted(order[inputs:].tolist())


def _predictions(
    denoiser: Denoiser, source: BurstSource, indices: list[int], shifts: np.ndarray,
    window: tuple[int, int], black: float, white: float, max_batch: int,
) -> list[np.ndarray]:
    size, outputs = denoiser.image_size, []
    for start in range(0, len(indices), max_batch):
        selected = indices[start:start + max_batch]
        crops, residuals = [], []
        for index in selected:
            origin = np.rint(np.asarray(window) + shifts[index]).astype(int)
            raw = source.frames[index][origin[0]:origin[0] + size, origin[1]:origin[1] + size]
            crops.append(normalize_native(raw, black, white) * 2 - 1)
            residuals.append(origin - shifts[index] - np.asarray(window))
        prediction = denoiser.denoise(torch.from_numpy(np.stack(crops))[:, None])
        # Use the acquisition transforms, not newly fitted output transforms:
        # a model's edge displacement must remain visible in the measurements.
        prediction = warp_prediction(prediction, torch.tensor(np.asarray(residuals), dtype=torch.float32))
        outputs.extend(((prediction[:, 0].numpy() + 1) / 2).clip(0, 1))
    return outputs


def _metrics(images: list[np.ndarray], reference: np.ndarray, locations: list) -> dict:
    stack = np.stack(images)
    window = min(11, min(reference.shape))
    window -= int(window % 2 == 0)
    if window < 3:
        raise ValueError("evaluation crop is too small after removing the registration border")
    widths, biases, failures = [], [], 0
    for location in locations:
        measured = [measure_site(image, location, tolerance=4.0, smooth=3) for image in images]
        valid = [value[0] for value in measured if value is not None]
        failures += len(measured) - len(valid)
        if len(valid) >= 2:
            widths.append(3 * float(np.std(valid, ddof=1)))
        if valid:
            biases.append(float(np.mean(valid) - location.clean_cd))
    reference_tensor = torch.from_numpy(reference.copy()).float()[None, None]
    predictions = torch.from_numpy(stack.copy()).float()[:, None]
    return {
        "realizations": len(images),
        "psnr_vs_reference": float(np.mean([psnr(reference, image) for image in images])),
        "ssim_vs_reference": float(np.mean([ssim(reference, image, window=window) for image in images])),
        "gradient_mse_vs_reference": float(((sobel(predictions) - sobel(reference_tensor)) ** 2).mean()),
        "pixel_sigma": float(np.sqrt(np.var(stack.astype(np.float64), axis=0, ddof=1).mean())),
        "cd_3sigma_px": float(np.median(widths)) if widths else None,
        "cd_bias_vs_reference_px": float(np.mean(biases)) if biases else None,
        "cd_failure_fraction": failures / (len(images) * len(locations)) if locations else None,
        "cd_locations_with_repeatability": len(widths),
    }


def _comparison(path: Path, images: dict[str, np.ndarray]) -> None:
    canvas = Image.new("L", (320 * len(images), 352), 255)
    draw = ImageDraw.Draw(canvas)
    for index, (name, values) in enumerate(images.items()):
        image = Image.fromarray(np.rint(values.clip(0, 1) * 255).astype(np.uint8))
        image.thumbnail((320, 320))
        canvas.paste(image, (320 * index, 32))
        draw.text((320 * index + 4, 8), name, fill=0)
    canvas.save(path)


def evaluate_real(
    config: Config, checkpoints: dict[str, str | Path], *, out_dir: str | Path,
    split: str = "val", input_frames: int = 32, rois: int = 5,
    max_batch: int = 4, device: str | None = None,
) -> dict:
    if not checkpoints or not 1 <= rois <= 5 or max_batch < 1:
        raise ValueError("provide checkpoints, 1..5 ROIs, and a positive max_batch")
    reserved = {"single_frame", "avg_of_4", "avg_of_8", "avg_of_16"}
    if reserved.intersection(checkpoints):
        raise ValueError("checkpoint names collide with classical baselines")
    logger.info("loading and verifying prepared data: %s", config.data.dataset_dir)
    cache = BurstCache(config.data.dataset_dir, min_size=config.data.image_size, min_replicas=4)
    if cache.real_metadata is None:
        raise ValueError("evaluate-real requires a dataset created by prepare-real")
    sources = cache.sources_for_split(split)
    if not sources:
        raise ValueError(f"no sites in prepared {split} split")
    metadata = cache.real_metadata
    sites = {site["source_index"]: site for site in metadata["sites"]}
    black, white = metadata["normalization"]["black"], metadata["normalization"]["white"]
    denoisers = {name: Denoiser.from_checkpoint(path, device=device or config.training.device)
                 for name, path in checkpoints.items()}
    for denoiser in denoisers.values():
        if denoiser.image_size != config.data.image_size:
            raise ValueError("checkpoint and evaluation crop sizes differ")
        if denoiser.config.data.white_level is not None and (
                denoiser.config.data.white_level != white or denoiser.config.data.black_level != black):
            raise ValueError("checkpoint and evaluation normalization differ")
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    details = []
    for source in sources:
        site = sites[source.source_index]
        inputs, references = frame_pools(len(source.frames), input_frames, metadata["split_seed"] + source.source_index)
        aligned = np.load(cache_path(cache.burst_dir, site["aligned"]["path"]), mmap_mode="r")
        shifts = np.asarray(site["shifts"])
        rows = []
        for roi, window in enumerate(fixed_windows(site["bounds"], config.data.image_size)[:rois]):
            logger.info("site %s, ROI %d: %d input frames, %d reference frames", site["name"], roi + 1, len(inputs), len(references))
            y, x = window
            size, margin = config.data.image_size, LOSS_MARGIN
            crop = np.s_[y:y + size, x:x + size]
            reference = np.mean([aligned[index][crop] for index in references], axis=0, dtype=np.float64).astype(np.float32)
            outputs = {"single_frame": [np.array(aligned[index][crop]) for index in inputs]}
            for count in (4, 8, 16):
                if len(inputs) // count >= 2:
                    outputs[f"avg_of_{count}"] = [np.mean(outputs["single_frame"][start:start + count], axis=0)
                                                 for start in range(0, len(inputs) - count + 1, count)]
            timing = {}
            for name, denoiser in denoisers.items():
                started = time.perf_counter()
                outputs[name] = _predictions(denoiser, source, inputs, shifts, window, black, white, max_batch)
                timing[name] = time.perf_counter() - started
            reference = reference[margin:-margin, margin:-margin].clip(0, 1)
            outputs = {name: [image[margin:-margin, margin:-margin].clip(0, 1) for image in images]
                       for name, images in outputs.items()}
            locations = find_cd_sites(reference, band_height=min(16, reference.shape[0]), max_sites=8)
            location_records = []
            for location in locations:
                record = asdict(location)
                record["reference_cd"] = record.pop("clean_cd")
                record["reference_center"] = record.pop("clean_center")
                location_records.append(record)
            rows.append({"origin_yx": list(window), "measurement_locations": location_records,
                         "methods": {name: _metrics(images, reference, locations) for name, images in outputs.items()},
                         "inference_seconds": timing})
            _comparison(destination / f"site{source.source_index:05d}_roi{roi}.png",
                        {"reference average": reference, **{name: images[0] for name, images in outputs.items() if name in denoisers or name == "single_frame"}})
        details.append({"site": site["name"], "source_index": source.source_index,
                        "input_indices": inputs, "reference_indices": references, "rois": rows})
    # Each site gets equal weight; overlapping crops and frames are not
    # presented as independently acquired sites or used to inflate sample n.
    metric_names = ("psnr_vs_reference", "ssim_vs_reference", "gradient_mse_vs_reference", "pixel_sigma",
                    "cd_3sigma_px", "cd_bias_vs_reference_px", "cd_failure_fraction")
    methods = {}
    method_names = details[0]["rois"][0]["methods"]
    for name in method_names:
        methods[name] = {}
        for metric in metric_names:
            per_site = []
            for site in details:
                values = [row["methods"][name][metric] for row in site["rois"] if row["methods"][name][metric] is not None]
                if values:
                    per_site.append(float(np.mean(values)))
            methods[name][metric] = float(np.median(per_site)) if per_site else None
    result = {"kind": "real_sem_reference_evaluation", "split": split, "site_count": len(details),
              "reference": "registered mean of disjoint frame pool; not clean ground truth",
              "summary": "median across sites of each site's mean across fixed ROIs",
              "dataset_fingerprint": cache.real_fingerprint,
              "checkpoints": {name: {"path": str(path), "sha256": file_digest(Path(path))} for name, path in checkpoints.items()},
              "methods": methods, "sites": details}
    (destination / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
