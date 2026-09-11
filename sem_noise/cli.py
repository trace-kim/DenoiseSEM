"""Command line entry point; optional analysis dependencies load only on use."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

from .config import load_config


def _inventory(source: Path, output: Path) -> None:
    from .io import SUFFIXES, _page_count, natural_key

    source = source.resolve()
    if not source.is_dir():
        raise ValueError("inventory input must be a directory")
    paths = sorted((p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES),
                   key=lambda p: natural_key(p.relative_to(source).as_posix()))
    if not paths:
        raise ValueError("no supported images found")
    rows, counters = [], {}
    for path in paths:
        relative = path.relative_to(source)
        site = relative.parent.as_posix() if relative.parent != Path(".") else source.name
        for page in range(_page_count(path)):
            index = counters.get(site, 0)
            rows.append({"site": site, "path": relative.as_posix(), "frame_index": index,
                         "page": page, "timestamp_s": "", "include": "true"})
            counters[site] = index + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} frames to {output}. Edit site and frame_index to match actual acquisition order.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Characterize noise, registration, and stability of repeated grayscale SEM images.")
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="Recursively create an editable site/order manifest CSV")
    inventory.add_argument("--input", required=True, type=Path)
    inventory.add_argument("--output", required=True, type=Path)
    analyze = commands.add_parser("analyze", help="Write an offline HTML report and numerical results")
    analyze.add_argument("--input", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path, help="New directory, outside input")
    analyze.add_argument("--manifest", type=Path)
    analyze.add_argument("--config", type=Path)
    analyze.add_argument("--frame-interval-s", type=float)
    analyze.add_argument("--pixel-size-nm", type=float)
    analyze.add_argument("--roi", nargs=4, type=int, metavar=("Y0", "Y1", "X0", "X1"))
    analyze.add_argument("--registration", choices=["translation", "none"])
    demo = commands.add_parser("demo", help="Generate three synthetic PNG sites with known noise and drift")
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--frames", type=int, default=128)
    demo.add_argument("--size", type=int, default=128)
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            _inventory(args.input, args.output)
            return 0
        if args.command == "demo":
            from .synthetic import create_demo
            create_demo(args.output, frames=args.frames, size=args.size)
            print(f"Synthetic data and truth written to {args.output}")
            return 0
        from .pipeline import analyze_dataset
        config = load_config(args.config)
        overrides = {name: getattr(args, name) for name in ("frame_interval_s", "pixel_size_nm", "roi", "registration") if getattr(args, name) is not None}
        result = analyze_dataset(args.input, args.output, config=replace(config, **overrides),
                                 manifest=args.manifest, progress=lambda message: print(message, flush=True))
        print(f"Report: {args.output / 'index.html'}")
        return 0 if result["status"] == "complete" else 1
    except ModuleNotFoundError as error:
        print(f'Missing dependency {error.name}. Install: python -m pip install -e ".[analysis]"', file=sys.stderr)
        return 2
    except (ValueError, TypeError, OSError, IndexError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
