"""Command-line interface: ``python -m edge_denoise <command>``.

Commands cover the experiment loop: ``train``, ``denoise`` (one measurement),
``evaluate`` (accuracy vs classical baselines), and ``repeatability`` (the
metrology-precision table, optionally holding burst_diffusion arms so every
method lands in one comparison).  Dataset generation is deliberately NOT
duplicated here -- edge_denoise trains on the same burst datasets produced by
``python -m burst_diffusion generate``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="edge_denoise",
    help="Edge-preserving deterministic denoisers for SEM metrology (N2N, gradient, hybrid).",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def _configure_logging(
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Log progress to stderr."),
) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s - %(name)s - %(message)s",
    )


def _parse_arms(entries: list[str], *, default_name: str) -> dict[str, Path]:
    arms: dict[str, Path] = {}
    for entry in entries:
        arm, _, path = entry.partition("=")
        if not path:
            arm, path = default_name, entry
        if arm in arms:
            raise typer.BadParameter(f"duplicate arm {arm!r}")
        arms[arm] = Path(path)
    return arms


@app.command()
def train(
    config: Path = typer.Option(..., help="YAML config path."),
    resume: bool = typer.Option(False, help="Resume from <run_dir>/ckpt_latest.pt."),
    resume_from: Optional[Path] = typer.Option(None, help="Resume from a specific checkpoint."),
) -> None:
    """Train an edge_denoise model as described by a config file."""
    import sys

    from .config import load_config
    from .provenance import write_provenance
    from .train import LATEST_CHECKPOINT_NAME, Trainer

    loaded = load_config(config)
    checkpoint: Path | None = resume_from
    if resume and checkpoint is None:
        checkpoint = Path(loaded.training.run_dir) / LATEST_CHECKPOINT_NAME
    if checkpoint is not None and not checkpoint.is_file():
        raise typer.BadParameter(f"resume checkpoint not found: {checkpoint}")
    trainer = Trainer(loaded, resume_from=checkpoint)
    final = trainer.run()
    provenance = write_provenance(
        loaded.training.run_dir,
        loaded,
        config_path=config,
        checkpoint=final,
        command=" ".join(sys.argv),
    )
    typer.echo(f"finished at step {trainer.step}; latest checkpoint: {final}")
    typer.echo(f"provenance written to {provenance}")


@app.command("distill-targets")
def distill_targets(
    config: Path = typer.Option(..., help="edge_denoise YAML config (dataset section drives the split)."),
    teacher: Path = typer.Option(..., help="Phase-1 teacher checkpoint (burst_diffusion or edge_denoise)."),
    out: Path = typer.Option(..., help="Output directory for per-source .npy targets + manifest."),
    splits: str = typer.Option("train,val", help="Comma-separated splits to cover (train, val)."),
    stride: int = typer.Option(48, min=1, help="Tile stride in pixels (tile size = data.image_size)."),
    tile_batch: int = typer.Option(64, min=1, help="Tiles per forward pass."),
    ema: bool = typer.Option(True, "--ema/--no-ema", help="Use the teacher's EMA weights."),
    device: str = typer.Option("auto", help="auto | cpu | cuda"),
) -> None:
    """Precompute per-scene averaged-denoised images for ``gradient_target: file``.

    Runs the frozen teacher over every replica of every covered source with
    blended overlapping tiles, averages the outputs, and writes one float32
    ``.npy`` per source -- the distillation arm's frozen "answer sheet".
    """
    import sys

    from .config import load_config
    from .distill import build_teacher, write_distill_targets

    loaded = load_config(config)
    denoise_fn, description = build_teacher(teacher, device=device, use_ema=ema)
    typer.echo(f"teacher: {description}", err=True)
    manifest = write_distill_targets(
        loaded,
        denoise_fn=denoise_fn,
        out_dir=out,
        splits=[part.strip() for part in splits.split(",") if part.strip()],
        stride=stride,
        tile_batch=tile_batch,
        teacher_path=teacher,
        teacher_description=description,
        command=" ".join(sys.argv),
        progress=lambda message: typer.echo(message, err=True),
    )
    total = sum(len(indices) for indices in manifest["splits"].values())
    typer.echo(f"wrote {total} target(s) + manifest to {out}")


@app.command()
def denoise(
    checkpoint: Path = typer.Option(..., help="Checkpoint (.pt) to denoise with."),
    out: Path = typer.Option(..., help="Output directory for PNGs."),
    input: Optional[Path] = typer.Option(None, help="A noisy measurement image (center-cropped)."),
    dataset: Optional[Path] = typer.Option(None, help="Burst dataset to pull a frame from."),
    source_index: int = typer.Option(0, min=0, help="Dataset source to denoise."),
    replica: int = typer.Option(0, min=0, help="Which burst frame is the measurement."),
    ema: bool = typer.Option(True, "--ema/--no-ema", help="Use the EMA weights."),
    device: str = typer.Option("auto", help="auto | cpu | cuda"),
) -> None:
    """Denoise one measurement with a single deterministic forward pass."""
    from burst_diffusion.data import resolve_burst_dir
    from burst_diffusion.metrics import psnr
    from burst_diffusion.sample import load_input_image, save_model_image

    from .infer import Denoiser

    if (input is None) == (dataset is None):
        raise typer.BadParameter("provide exactly one of --input or --dataset")
    denoiser = Denoiser.from_checkpoint(checkpoint, device=device, use_ema=ema)
    image_size = denoiser.image_size
    channels = denoiser.config.data.channels

    clean_path: Path | None = None
    if input is not None:
        measurement_path = input
        stem = input.stem
    else:
        burst_dir = resolve_burst_dir(dataset)
        measurement_path = burst_dir / "noisy" / f"{source_index:05d}_{replica:05d}.png"
        if not measurement_path.is_file():
            raise typer.BadParameter(f"no such burst frame: {measurement_path}")
        candidate = burst_dir / "clean" / f"{source_index:05d}.png"
        clean_path = candidate if candidate.is_file() else None
        stem = f"src{source_index:05d}_rep{replica:05d}"

    measurement = load_input_image(measurement_path, image_size=image_size, channels=channels)
    denoised = denoiser.denoise(measurement)
    save_model_image(measurement[0], out / f"{stem}_input.png")
    save_model_image(denoised[0], out / f"{stem}_denoised.png")
    typer.echo(f"wrote 2 PNG(s) to {out}")

    if clean_path is not None:
        clean = load_input_image(clean_path, image_size=image_size, channels=channels)
        clean01 = ((clean[0] + 1) / 2).numpy().transpose(1, 2, 0)
        for label, tensor in (("input", measurement[0]), ("denoised", denoised[0])):
            value01 = ((tensor.clamp(-1, 1) + 1) / 2).numpy().transpose(1, 2, 0)
            typer.echo(f"PSNR vs clean [{label}]: {psnr(clean01, value01):.2f} dB")


@app.command()
def evaluate(
    config: Path = typer.Option(..., help="YAML config path (dataset + model)."),
    checkpoint: Path = typer.Option(..., help="Checkpoint (.pt) to evaluate."),
    out: Path = typer.Option(..., help="Output directory for results.json."),
    split: str = typer.Option("val", help="val | train | test (test is the locked holdout: report once)"),
    limit: Optional[int] = typer.Option(None, help="Evaluate at most this many sources."),
    device: Optional[str] = typer.Option(None, help="auto | cpu | cuda (default: config)."),
) -> None:
    """Compare the model against the classical baselines on held-out sources."""
    from .config import load_config
    from .evaluate import evaluate as run_evaluation

    results = run_evaluation(
        load_config(config),
        checkpoint,
        split=split,
        limit=limit,
        out_dir=out,
        device=device,
    )
    for name, method in results["methods"].items():
        typer.echo(
            f"{name:>15}: PSNR {method['psnr_mean']:6.2f} dB "
            f"(median {method['psnr_median']:6.2f}) | SSIM {method['ssim_mean']:.4f}"
        )
    typer.echo(f"results written to {Path(out) / 'results.json'}")


@app.command()
def repeatability(
    config: Path = typer.Option(..., help="edge_denoise YAML config (dataset section drives the split)."),
    checkpoint: list[str] = typer.Option(
        ...,
        help="edge_denoise checkpoint arm as NAME=PATH (repeatable; a bare PATH is named 'edge').",
    ),
    burst_checkpoint: list[str] = typer.Option(
        [],
        help="burst_diffusion checkpoint arm as NAME=PATH (repeatable) to include in the same table.",
    ),
    out: Path = typer.Option(..., help="Output directory for repeatability.json + summary.md."),
    split: str = typer.Option("val", help="val | train | test (test is the locked holdout: report once)"),
    limit: Optional[int] = typer.Option(None, help="Evaluate at most this many sources."),
    seeds: int = typer.Option(10, min=1, help="Fresh frames (seeds) per source."),
    device: str = typer.Option("auto", help="auto | cpu | cuda"),
) -> None:
    """One metrology-precision table: classical + edge_denoise (+ burst) arms.

    Runs burst_diffusion's repeatability harness (CD sites, subpixel
    crossings, pooled c4-debiased sigmas) with edge arms supplied as
    realization providers, so every method is measured on identical sources,
    seeds, crops, and CD sites.
    """
    from burst_diffusion.config import Config as BurstConfig
    from burst_diffusion.repeatability import repeatability as run_repeatability
    from burst_diffusion.train import load_checkpoint as load_burst_checkpoint

    from .config import load_config
    from .provider import providers_from_checkpoints

    edge_arms = _parse_arms(checkpoint, default_name="edge")
    burst_arms = _parse_arms(burst_checkpoint, default_name="burst")
    overlap = set(edge_arms) & set(burst_arms)
    if overlap:
        raise typer.BadParameter(f"arm name(s) used in both tiers: {sorted(overlap)}")

    loaded = load_config(config)

    # The harness plans its schedule and frame requirements from a burst
    # config; bridge the edge config's data section into one.  Every burst arm
    # must share one num_steps (the harness validates each checkpoint against
    # it); with no burst arms a trivial schedule is used.
    num_steps = 1
    steps_seen: dict[str, int] = {}
    for arm, path in burst_arms.items():
        payload = load_burst_checkpoint(path)
        steps_seen[arm] = int(BurstConfig.model_validate(payload["config"]).schedule.num_steps)
    if steps_seen:
        distinct = sorted(set(steps_seen.values()))
        if len(distinct) > 1:
            raise typer.BadParameter(
                f"burst arms disagree on schedule.num_steps ({steps_seen}); "
                "run arms with different schedules separately"
            )
        num_steps = distinct[0]
    bridge = BurstConfig.model_validate(
        {
            "data": {
                "dataset_dir": str(loaded.data.dataset_dir),
                "image_size": loaded.data.image_size,
                "channels": loaded.data.channels,
                "val_fraction": loaded.data.val_fraction,
                "test_fraction": loaded.data.test_fraction,
                "split_seed": loaded.data.split_seed,
            },
            "schedule": {"num_steps": num_steps},
            "model": {"ch": 8, "ch_mult": [1], "num_res_blocks": 1, "attn_resolutions": []},
            "training": {"run_dir": str(out), "device": device},
            "sampling": {},
        }
    )

    providers = providers_from_checkpoints(edge_arms, device=device)
    results = run_repeatability(
        bridge,
        burst_arms,
        out_dir=out,
        split=split,
        limit=limit,
        num_seeds=seeds,
        device=device,
        extra_providers=providers,
        progress_callback=lambda done, total: typer.echo(f"source {done}/{total}", err=True),
    )
    for name, method in results["methods"].items():
        pixel = method["pixel_repeatability"]
        pixel_text = "-" if pixel is None else f"{pixel['sigma_mean'] * 1e3:6.2f}e-3"
        cd3 = method["cd"]["scene_median_3sigma_px"]
        cd_text = "-" if cd3 is None else f"{cd3:6.3f} px"
        shift = method["registration"]["shift_sigma_px"]
        shift_text = "-" if shift is None else f"{shift:6.3f} px"
        typer.echo(
            f"{name:>26}: PSNR {method['accuracy']['psnr_mean']:6.2f} dB | "
            f"pixel sigma {pixel_text} | CD 3sigma scene {cd_text} | shift sigma {shift_text}"
        )
    typer.echo(f"results written to {Path(out) / 'repeatability.json'}")
