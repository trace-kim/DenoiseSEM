"""Average consecutive repeats of each site into average2/average4/average8 folders.

Only images directly in --source are used (subfolders are ignored), in natural
filename order; every `--frames-per-site` consecutive files are one site.
No registration. Averages are saved as lossless grayscale PNG.

`sites/site001/ ...` receives byte-identical copies of each site's original
files: the one-folder-per-site layout `python -m edge_denoise prepare-real`
expects for training.

    python tools/average_site_groups.py --source /data/20261002_162547 \
        --output /data/20261002_162547_averages
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image

EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}


def natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
        if image.mode == "RGB":
            if not (np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 0], array[..., 2])):
                raise ValueError(f"RGB channels differ (not a grayscale image): {path}")
            array = array[..., 0]
    if array.ndim != 2 or array.dtype != np.uint8:
        raise ValueError(f"expected grayscale uint8 image: {path} ({array.shape}, {array.dtype})")
    return array


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-site", type=int, default=8)
    parser.add_argument("--averages", type=lambda v: [int(k) for k in v.split(",")], default=[2, 4, 8])
    args = parser.parse_args(argv)

    source, output, n = args.source.resolve(), args.output.resolve(), args.frames_per_site
    if output.exists():
        raise SystemExit(f"Output already exists: {output}")
    if output == source or source in output.parents:
        raise SystemExit("Output must not be inside the source folder")
    if any(n % k for k in args.averages):
        raise SystemExit(f"Every average size must divide {n}")

    # Only images directly inside --source; subfolders are never read.
    files = sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS), key=natural_key)
    if not files:
        raise SystemExit(f"No images found directly in {source}")
    if len(files) % n:
        raise SystemExit(f"{source}: {len(files)} images is not a multiple of {n}")

    for k in args.averages:
        (output / f"average{k}").mkdir(parents=True)
    for s in range(len(files) // n):
        site = files[s * n:(s + 1) * n]
        site_dir = output / "sites" / f"site{s + 1:03d}"
        site_dir.mkdir(parents=True)
        for path in site:
            shutil.copy2(path, site_dir / path.name)
        frames = [read_gray(p) for p in site]
        if len({f.shape for f in frames}) != 1:
            raise SystemExit(f"Different image sizes within site: {site[0]} .. {site[-1]}")
        stack = np.stack(frames).astype(np.float64)
        for k in args.averages:
            for g in range(n // k):
                group = site[g * k:(g + 1) * k]
                mean = stack[g * k:(g + 1) * k].mean(axis=0)
                pixels = np.clip(np.rint(mean), 0, 255).astype(np.uint8)
                name = f"site{s + 1:03d}_avg{k}_{g + 1}_{group[0].stem}-{group[-1].stem}.png"
                Image.fromarray(pixels).save(output / f"average{k}" / name)
    print(f"{len(files)} images, {len(files) // n} sites; wrote "
          f"{', '.join(f'average{k}' for k in args.averages)} under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
