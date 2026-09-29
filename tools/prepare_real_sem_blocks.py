"""Build average2/4 and affine-registered average2/4 image folders and training caches.

Run from the repository root; see edge_denoise/docs/real_sem_block_datasets.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from burst_diffusion.data import BurstCache, content_key
from burst_diffusion.real_data import REAL_FORMAT, file_digest
from edge_denoise.real_data import IMAGE_EXTENSIONS, _natural_key, prepare_real_dataset, read_native
from edge_denoise.uint8_output import average_uint8
from tools.real_sem_averages import BLOCK_MANIFEST, load_block_manifest, registered_average

VARIANTS = {"average2": (2, False), "average4": (4, False),
            "average2_registered": (2, True), "average4_registered": (4, True)}


def image_files(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                  key=_natural_key)


def write_json(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inventory(raw: Path, parent: dict, expected_frames: int) -> dict[str, dict]:
    """Hash decoded content before reusing the existing training-site splits."""
    sites = {}
    for split in ("train", "test", "all"):
        folder = raw / split
        if not folder.is_dir():
            raise ValueError(f"missing raw split directory: {folder}")
        folders = sorted((p for p in folder.iterdir() if p.is_dir()), key=_natural_key)
        if not folders:
            raise ValueError(f"no site folders under {folder}")
        for site in folders:
            files = image_files(site)
            if len(files) != expected_frames:
                raise ValueError(f"{site}: expected {expected_frames} frames, found {len(files)}; no partial blocks are dropped")
            hashes, shape = [], None
            for file in files:
                pixels = read_native(file)
                if pixels.dtype != np.uint8 or (shape is not None and pixels.shape != shape):
                    raise ValueError(f"expected same-sized uint8 images within each site: {file}")
                shape = pixels.shape
                hashes.append(content_key(pixels))
            relative = site.relative_to(raw).as_posix()
            sites[relative] = {"files": files, "hashes": hashes, "shape": shape}
            print(f"Hashed {relative}: {len(files)} frames, {shape[0]} x {shape[1]}", flush=True)
    training = {key.split("/", 1)[1] for key in sites if key.startswith("train/")}
    if training != {site["name"] for site in parent["sites"]}:
        raise ValueError("raw train sites must exactly match the existing prepared manifest's sites")
    for site in parent["sites"]:
        actual = sites[f"train/{site['name']}"]
        if actual["hashes"] != [f["sha256"] for f in site["frames"]]:
            raise ValueError(f"{site['name']}: raw training pixels/order differ from the prepared manifest")
    training_hashes = {h for key, site in sites.items() if key.startswith("train/") for h in site["hashes"]}
    for key, site in sites.items():
        if key.startswith("test/") and training_hashes.intersection(site["hashes"]):
            raise ValueError(f"raw train/test image content overlaps: {key}")
    return sites


def create_site(raw: Path, relative: str, source: dict, destinations: dict[str, Path],
                *, device: str, equivalents: dict[tuple, Path], resume: bool) -> None:
    """Decode groups of four once; publish each variant/site atomically."""
    from contextlib import ExitStack

    active = {}
    with ExitStack() as stack:
        for variant, destination in destinations.items():
            count, registered = VARIANTS[variant]
            identity = (variant, tuple(source["hashes"]))
            if destination.exists():
                if not resume:
                    raise ValueError(f"output exists: {destination}; use --resume to verify and reuse it")
                record = load_block_manifest(destination, image_files(destination))
                if (record is None or record["frames_per_output"] != count or record["registered"] != registered
                        or [h for f in record["frames"] for h in f["source_sha256"]] != source["hashes"]):
                    raise ValueError(f"source/settings changed: {destination}")
                equivalents[identity] = destination
                print(f"Verified/reused {variant}/raw/{relative}", flush=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Stage outside raw/<split>/ so a hard-killed job cannot leave a
            # temporary folder that prepare-real mistakes for an extra site.
            temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix=".blocks-", dir=destination.parents[2]))
            staging = Path(temporary)
            reused = equivalents.get(identity)
            if reused is not None:
                shutil.copytree(reused, staging, dirs_exist_ok=True)
                record = json.loads((staging / BLOCK_MANIFEST).read_text(encoding="utf-8"))
                record.update(source_dir=str(raw / relative), reused_from=str(reused))
                for index, frame in enumerate(record["frames"]):
                    frame["source_files"] = [p.name for p in source["files"][index * count:(index + 1) * count]]
                write_json(staging / BLOCK_MANIFEST, record)
                staging.rename(destination)
                print(f"Reused identical acquisitions for {variant}/raw/{relative}", flush=True)
                continue
            active[variant] = (staging, destination, {"format": 1, "source_dir": str(raw / relative),
                "frames_per_output": count, "registered": registered, "registration": "affine" if registered else "none",
                "brightness": "native; no gain or offset", "grouping": "consecutive, nonoverlapping, natural filename order",
                "source_frame_count": len(source["files"]), "frames": [], "device": device,
                "timings_s": {"decode": 0., "production_and_export": 0.}})
        if not active:
            return
        for start in range(0, len(source["files"]), 4):
            began = time.perf_counter()
            pixels = np.stack([read_native(p) for p in source["files"][start:start + 4]])
            if [content_key(p) for p in pixels] != source["hashes"][start:start + 4]:
                raise ValueError(f"raw inputs changed during preparation: {relative}")
            decode_s = time.perf_counter() - began
            for variant, (staging, _, record) in active.items():
                count, registered = VARIANTS[variant]
                record["timings_s"]["decode"] += decode_s
                began = time.perf_counter()
                for offset in range(0, 4, count):
                    first, last = start + offset, start + offset + count
                    block = pixels[offset:offset + count]
                    name = f"block_{first // count + 1:03d}_{first + 1:03d}-{last:03d}.png"
                    frame = {"name": name, "first_acquisition": first + 1, "last_acquisition": last,
                             "source_files": [p.name for p in source["files"][first:last]],
                             "source_sha256": source["hashes"][first:last]}
                    if registered:
                        values, support, diagnostics = registered_average(block, device=device)
                        frame["registration"] = diagnostics
                        frame["support"] = Path(name).with_suffix(".support.npz").name
                        np.savez_compressed(staging / frame["support"], valid=support)
                        frame["support_sha256"] = file_digest(staging / frame["support"])
                    else:
                        values = average_uint8(block)
                    Image.fromarray(np.repeat(values[..., None], 3, axis=2)).save(staging / name)
                    # Cache preparation/QC will decode these same delivered PNGs.
                    frame.update(sha256=content_key(read_native(staging / name)), file_sha256=file_digest(staging / name))
                    record["frames"].append(frame)
                record["timings_s"]["production_and_export"] += time.perf_counter() - began
            if (start + 4) % 32 == 0 or start + 4 == len(source["files"]):
                print(f"{relative}: averaged {start + 4}/{len(source['files'])} source frames", flush=True)
        for variant, (staging, destination, record) in active.items():
            write_json(staging / BLOCK_MANIFEST, record)
            staging.rename(destination)
            equivalents[(variant, tuple(source["hashes"]))] = destination


def build_datasets(raw_dir: Path, prepared_dir: Path, output_dir: Path, *,
                   device: str = "cpu", image_size: int = 512, expected_frames: int = 128,
                   variants: list[str] | None = None, resume: bool = False) -> Path:
    """Create raw mirrors and align-none caches with the parent's locked splits."""
    started = time.perf_counter()
    raw, prepared, output = (Path(p).resolve() for p in (raw_dir, prepared_dir, output_dir))
    selected = list(VARIANTS) if variants is None else variants
    if not selected or len(set(selected)) != len(selected) or any(v not in VARIANTS for v in selected):
        raise ValueError("select unique known variants")
    if expected_frames < 8 or expected_frames % 4 or image_size < 8:
        raise ValueError("expected frames must be >=8 and divisible by four; image size must be >=8")
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (raw, prepared)):
        raise ValueError("output must be outside and must not contain the raw/prepared input directories")
    manifest = prepared / "real_dataset.json"
    parent = json.loads(manifest.read_text(encoding="utf-8"))
    if (parent.get("format") != REAL_FORMAT or parent.get("kind") != "real_sem"
            or parent["registration"]["mode"] != "none" or not parent["sites"]):
        raise ValueError("prepared-dir must contain an align-none real SEM dataset")
    if parent["normalization"] != {"black": 0, "white": 255}:
        raise ValueError("block datasets require the existing uint8 normalization black=0, white=255")
    if any(s["split"] not in ("train", "val", "test") for s in parent["sites"]):
        raise ValueError("invalid original site split")
    from edge_denoise.train import resolve_device

    device = str(resolve_device(device))
    request = {"format": 1, "raw_dir": str(raw), "prepared_dir": str(prepared),
               "parent_manifest_sha256": file_digest(manifest), "variants": selected,
               "image_size": image_size, "expected_frames": expected_frames, "device": device}
    if output.exists():
        if not resume or not (output / "request.json").is_file():
            raise ValueError("output must be new; use --resume for a previous block-dataset run")
        if json.loads((output / "request.json").read_text(encoding="utf-8")) != request:
            raise ValueError("resume settings or original prepared manifest changed; use a new output directory")
    hashed = time.perf_counter()
    sites = inventory(raw, parent, expected_frames)
    for key, source in sites.items():
        if key.startswith("train/") and min(source["shape"]) < image_size + 8:
            raise ValueError(f"{key}: images cannot fit {image_size}px training crops with preparation margin")
    hash_s = time.perf_counter() - hashed
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "request.json", request)
    equivalents = {}
    for relative, source in sites.items():
        create_site(raw, relative, source, {v: output / v / "raw" / relative for v in selected},
                    device=device, equivalents=equivalents, resume=resume)
    for variant in selected:
        root = output / variant
        destination = root / "train_align_none"
        derivation = {"variant": variant, "frames_per_output": VARIANTS[variant][0],
            "registration": "affine" if VARIANTS[variant][1] else "none", "brightness": "none",
            "parent_manifest_sha256": request["parent_manifest_sha256"], "measurement_source": "decoded saved uint8 PNGs"}
        sources, saved_hashes = {}, {}
        for site in parent["sites"]:
            folder = root / "raw" / "train" / site["name"]
            record = load_block_manifest(folder, image_files(folder))
            for frame in record["frames"]:
                relative = f"{site['name']}/{frame['name']}"
                sources[relative] = frame["source_sha256"]
                saved_hashes[relative] = frame["sha256"]
        if destination.exists():
            # Verify every saved array before reporting a resumed cache complete.
            cache = BurstCache(destination, min_size=image_size)
            metadata = cache.real_metadata
            frames = [f for s in metadata["sites"] for f in s["frames"]]
            if (metadata.get("block_averaging") != derivation
                    or {s["name"]: s["split"] for s in metadata["sites"]} != {s["name"]: s["split"] for s in parent["sites"]}
                    or {f["name"]: f.get("source_sha256") for f in frames} != sources
                    or {f["name"]: f["sha256"] for f in frames} != saved_hashes):
                raise ValueError(f"prepared block provenance/splits changed: {destination}")
            print(f"Verified/reused {destination}", flush=True)
            del cache
            continue
        splits = root / "site_splits.json"
        write_json(splits, {site["name"]: site["split"] for site in parent["sites"]})
        print(f"Preparing training cache for {variant}", flush=True)
        with tempfile.TemporaryDirectory(prefix=".block-cache-", dir=root) as temporary:
            prepared_manifest = prepare_real_dataset(root / "raw" / "train", Path(temporary) / "prepared",
                image_size=image_size, align="none", black_level=0, white_level=255,
                split_file=splits, frame_sources=sources, val_fraction=parent.get("val_fraction", .1),
                test_fraction=parent.get("test_fraction", .1), split_seed=parent.get("split_seed", 2019),
                progress=lambda message: print(message, flush=True))
            metadata = json.loads(prepared_manifest.read_text(encoding="utf-8"))
            metadata["block_averaging"] = derivation
            write_json(prepared_manifest, metadata)
            prepared_manifest.parent.rename(destination)
    write_json(output / "block_datasets.json", {**request, "status": "complete", "sites": len(sites),
        "outputs_per_site": {v: expected_frames // VARIANTS[v][0] for v in selected},
        "timings_s": {"source_decode_and_hash": hash_s, "total": time.perf_counter() - started}})
    print(f"Complete: {output / 'block_datasets.json'}", flush=True)
    return output / "block_datasets.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("/data/260904_raw_data"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("/data/260904_prep_data/train_align_none"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda:0 (within scheduler visibility)")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--expected-frames", type=int, default=128)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--resume", action="store_true", help="Verify/reuse completed sites and caches after interruption")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    if any(VARIANTS[v][1] for v in args.variants):
        import cv2

        cv2.setNumThreads(args.cpu_threads)
    build_datasets(args.raw_dir, args.prepared_dir, args.output_dir, device=args.device,
                   image_size=args.image_size, expected_frames=args.expected_frames,
                   variants=args.variants, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
