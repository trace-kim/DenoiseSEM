"""Command-line interface: ``python -m edge_denoise <command>``.

Commands cover the experiment loop: ``train``, ``denoise`` (one measurement),
``evaluate`` (accuracy vs classical baselines), and ``repeatability`` (the
metrology-precision table, optionally holding burst_diffusion arms so every
method lands in one comparison). ``prepare-real`` imports measured repeats
and ``evaluate-real`` compares them against disjoint reference averages.
Synthetic datasets still come from ``python -m burst_diffusion generate``.
"""

from __future__ import annotations

import json
import logging
import os
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
        level=logging.INFO if verbose and int(os.environ.get("RANK", "0")) == 0 else logging.WARNING,
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
    overwrite: bool = typer.Option(
        False,
        help=(
            "Start over in a run_dir that already holds a run: its checkpoints, "
            "provenance, config copy and TensorBoard logs are deleted first. Without "
            "this flag (or --resume) an occupied run_dir is refused."
        ),
    ),
    run_dir: Optional[Path] = typer.Option(None, help="Override training.run_dir."),
    max_steps: Optional[int] = typer.Option(None, min=1, help="Total optimizer steps, including resumed steps."),
    batch_size: Optional[int] = typer.Option(None, min=1, help="Samples per GPU per microstep."),
    accumulation_steps: Optional[int] = typer.Option(None, min=1, help="Microsteps per optimizer update."),
    precision: Optional[str] = typer.Option(None, help="fp32 | bf16 (CUDA only)."),
    device: Optional[str] = typer.Option(None, help="auto | cpu | cuda."),
    seed: Optional[int] = typer.Option(None, min=0, help="Base training seed."),
    init_checkpoint: Optional[Path] = typer.Option(None, help="Weights-only initialization for a fresh run."),
) -> None:
    """Train an edge_denoise model as described by a config file."""
    import sys

    from .config import Config, load_config
    from .provenance import write_provenance
    from .train import LATEST_CHECKPOINT_NAME, Trainer

    loaded = load_config(config)
    raw = loaded.model_dump()
    overrides = {"run_dir": run_dir, "max_steps": max_steps, "batch_size": batch_size,
                 "accumulation_steps": accumulation_steps, "precision": precision,
                 "device": device, "seed": seed, "init_checkpoint": init_checkpoint}
    raw["training"].update({name: value for name, value in overrides.items() if value is not None})
    loaded = Config.model_validate(raw)
    checkpoint: Path | None = resume_from
    if resume and checkpoint is None:
        checkpoint = Path(loaded.training.run_dir) / LATEST_CHECKPOINT_NAME
    if checkpoint is not None and not checkpoint.is_file():
        raise typer.BadParameter(f"resume checkpoint not found: {checkpoint}")
    try:
        trainer = Trainer(loaded, resume_from=checkpoint, overwrite=overwrite)
    except FileExistsError as error:
        raise typer.BadParameter(str(error)) from error
    try:
        final = trainer.run()
        provenance = trainer.runtime.on_primary(lambda: write_provenance(
            trainer.config.training.run_dir, trainer.config, config_path=config,
            checkpoint=final, command=" ".join(sys.argv), cache=trainer.cache,
        ))
        if trainer.runtime.primary:
            typer.echo(f"finished at step {trainer.step}; latest checkpoint: {final}")
            typer.echo(f"provenance written to {provenance}")
    finally:
        trainer.runtime.close()


@app.command("prepare-real")
def prepare_real(
    source_dir: Path = typer.Option(..., help="One subfolder per site; grayscale or identical-channel RGB images, including JPEG."),
    out: Path = typer.Option(..., help="New prepared dataset directory, outside source-dir."),
    image_size: int = typer.Option(512, min=8, help="Minimum required training crop size in the registration overlap; full frames are stored."),
    black_level: float = typer.Option(0.0, help="Fixed detector black level; never estimated per frame."),
    white_level: Optional[float] = typer.Option(None, help="Fixed white level; default 255/65535 from storage dtype."),
    val_fraction: float = typer.Option(0.1, min=0, max=0.99),
    test_fraction: float = typer.Option(0.1, min=0, max=0.99),
    split_seed: int = typer.Option(2019, min=0),
    split_file: Optional[Path] = typer.Option(None, help="Optional JSON mapping every site folder to train/val/test."),
    align: str = typer.Option("translation", help="translation | none (only for already aligned data)."),
    sigma: float = typer.Option(2.0, min=0, help="Smoothing used to estimate shifts, not to train inputs."),
    radius: int = typer.Option(6, min=1, help="Shift search radius around the previous frame estimate."),
    max_shift: float = typer.Option(32.0, min=0.01, help="Reject larger absolute shifts in either axis."),
    min_registration_contrast: float = typer.Option(
        0.005, min=0.0, help="Skip frames below this std of 16px block means in [0,1]; 0 disables the check."),
    frame_start: int = typer.Option(0, min=0, help="First frame after natural filename sorting (zero based)."),
    frame_stop: Optional[int] = typer.Option(None, min=1, help="Exclusive final frame after sorting."),
    device: str = typer.Option("cpu", help="cpu | cuda | auto; registration only."),
) -> None:
    """Prepare native real repeats, registration, QC previews, and locked site splits."""
    from .real_data import prepare_real_dataset
    from .train import resolve_device

    manifest = prepare_real_dataset(
        source_dir, out, image_size=image_size, black_level=black_level, white_level=white_level,
        val_fraction=val_fraction, test_fraction=test_fraction, split_seed=split_seed,
        split_file=split_file, align=align, sigma=sigma, radius=radius, max_shift=max_shift,
        frame_start=frame_start, frame_stop=frame_stop, device=str(resolve_device(device)),
        min_registration_contrast=min_registration_contrast,
        progress=lambda message: typer.echo(message),
    )
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        names = [site["name"] for site in metadata["sites"] if site["split"] == split]
        typer.echo(f"{split}: {len(names)} sites: {', '.join(names)}")
    typer.echo(f"Prepared {manifest}; inspect qc.csv and previews/ before training.")


@app.command("evaluate-real")
def evaluate_real_command(
    config: Path = typer.Option(..., help="YAML config selecting the prepared dataset and crop size."),
    checkpoint: list[str] = typer.Option(..., help="Repeat NAME=checkpoint.pt for each model."),
    out: Path = typer.Option(..., help="Results JSON and comparison previews."),
    split: str = typer.Option("val", help="val | test | train; prepared site splits are fixed."),
    input_frames: int = typer.Option(32, min=2, help="Frames used as inference inputs; remaining frames form the reference."),
    rois: int = typer.Option(5, min=1, max=5, help="Fixed center/corner crops per site."),
    max_batch: int = typer.Option(4, min=1, help="Inference frames per forward pass."),
    device: str = typer.Option("auto", help="auto | cpu | cuda."),
) -> None:
    """Compare real repeats using disjoint reference frames and fixed measurement boxes."""
    from .config import load_config
    from .real_evaluate import evaluate_real

    result = evaluate_real(load_config(config), _parse_arms(checkpoint, default_name="model"),
                           out_dir=out, split=split, input_frames=input_frames, rois=rois,
                           max_batch=max_batch, device=device)
    for name, metrics in result["methods"].items():
        typer.echo(f"{name}: PSNR vs reference {metrics['psnr_vs_reference']:.2f} dB; "
                   f"CD 3-sigma {metrics['cd_3sigma_px']} px; CD failure fraction {metrics['cd_failure_fraction']}")
    typer.echo(f"Reference-based results for {result['site_count']} sites written to {out}")


@app.command("train-prior")
def train_prior(
    config: Path = typer.Option(..., help="Prior YAML config (data / diffusion / model / training)."),
    resume: bool = typer.Option(False, help="Resume from <run_dir>/ckpt_latest.pt."),
) -> None:
    """Train the DDPM prior on clean train-split crops (the generative arm)."""
    from .prior import LATEST_CHECKPOINT_NAME, PriorTrainer, load_prior_config

    loaded = load_prior_config(config)
    checkpoint = Path(loaded.training.run_dir) / LATEST_CHECKPOINT_NAME if resume else None
    if checkpoint is not None and not checkpoint.is_file():
        raise typer.BadParameter(f"resume checkpoint not found: {checkpoint}")
    trainer = PriorTrainer(loaded, resume_from=checkpoint)
    final = trainer.run()
    typer.echo(f"finished at step {trainer.step}; latest checkpoint: {final}")


def _posterior_arms(
    prior_checkpoint: Optional[Path], specs: list[str], *, device: str, peak: float
) -> dict:
    """Build posterior samplers from ``--posterior-arm NAME=mode,key=value...``."""
    from .prior import PosteriorSampler, parse_posterior_spec

    if specs and prior_checkpoint is None:
        raise typer.BadParameter("--posterior-arm needs --prior-checkpoint")
    arms = {}
    for spec in specs:
        name, kwargs = parse_posterior_spec(spec)
        if name in arms:
            raise typer.BadParameter(f"duplicate posterior arm {name!r}")
        arms[name] = PosteriorSampler.from_checkpoint(
            prior_checkpoint, device=device, peak=peak, **kwargs
        )
    return arms


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
    out: Path = typer.Option(..., help="Output directory for PNG previews and a float32 TIFF."),
    input: Optional[Path] = typer.Option(
        None, help="A noisy measurement image of any size >= the training crop."
    ),
    dataset: Optional[Path] = typer.Option(None, help="Burst dataset to pull a frame from."),
    source_index: int = typer.Option(0, min=0, help="Dataset source to denoise."),
    replica: int = typer.Option(0, min=0, help="Which burst frame is the measurement."),
    full: bool = typer.Option(
        True,
        "--full/--center-crop",
        help=(
            "Denoise the WHOLE frame with blended overlapping tiles (default); "
            "--center-crop processes a single training-resolution center tile instead."
        ),
    ),
    stride: Optional[int] = typer.Option(
        None, min=1, help="Tile stride (default 48 for small models, half the tile for 512px models)."
    ),
    tile_batch: int = typer.Option(4, min=1, help="Tiles per forward pass for the full frame."),
    ema: bool = typer.Option(True, "--ema/--no-ema", help="Use the EMA weights."),
    device: str = typer.Option("auto", help="auto | cpu | cuda"),
) -> None:
    """Denoise one measurement deterministically (the whole frame by default)."""
    import numpy as np
    import torch
    from PIL import Image

    from burst_diffusion.data import resolve_burst_dir
    from burst_diffusion.metrics import psnr
    from burst_diffusion.sample import save_model_image

    from .infer import Denoiser, load_measurement01

    if (input is None) == (dataset is None):
        raise typer.BadParameter("provide exactly one of --input or --dataset")
    denoiser = Denoiser.from_checkpoint(checkpoint, device=device, use_ema=ema)
    image_size = denoiser.image_size
    stride = stride if stride is not None else (min(48, image_size) if image_size <= 64 else image_size // 2)

    clean_path: Path | None = None
    frame01 = None
    if input is not None:
        measurement_path = input
        stem = input.stem
    else:
        burst_dir = resolve_burst_dir(dataset)
        if (burst_dir / "real_dataset.json").is_file():
            from burst_diffusion.data import BurstCache
            from burst_diffusion.real_data import normalize_native

            cache = BurstCache(burst_dir, min_replicas=1)
            source = next((item for item in cache.all_sources if item.source_index == source_index), None)
            if source is None or replica >= len(source.frames):
                raise typer.BadParameter("no such prepared site/frame")
            levels = cache.real_metadata["normalization"]
            if denoiser.config.data.white_level is not None and (
                    denoiser.config.data.white_level != levels["white"] or denoiser.config.data.black_level != levels["black"]):
                raise typer.BadParameter("checkpoint and prepared dataset normalization differ")
            frame01 = normalize_native(source.frames[replica], levels["black"], levels["white"])
        else:
            measurement_path = burst_dir / "noisy" / f"{source_index:05d}_{replica:05d}.png"
            if not measurement_path.is_file():
                raise typer.BadParameter(f"no such burst frame: {measurement_path}")
            candidate = burst_dir / "clean" / f"{source_index:05d}.png"
            clean_path = candidate if candidate.is_file() else None
        stem = f"src{source_index:05d}_rep{replica:05d}"

    def to_model(frame01: "np.ndarray") -> torch.Tensor:  # [H, W] in [0, 1] -> [1, H, W] in [-1, 1]
        return torch.from_numpy(frame01 * 2.0 - 1.0).to(torch.float32)[None]

    if full:
        frame01 = denoiser.load_measurement(measurement_path) if frame01 is None else frame01
        denoised01 = denoiser.denoise_full(frame01, stride=stride, tile_batch=tile_batch)
        measurement_chw, denoised_chw = to_model(frame01), to_model(denoised01)
        typer.echo(
            f"denoised the full {frame01.shape[0]}x{frame01.shape[1]} frame "
            f"(tile {image_size}, stride {min(stride, image_size)})"
        )
    else:
        frame01 = denoiser.load_measurement(measurement_path) if frame01 is None else frame01
        if min(frame01.shape) < image_size:
            raise typer.BadParameter(f"input must be at least {image_size}x{image_size}")
        y, x = (frame01.shape[0] - image_size) // 2, (frame01.shape[1] - image_size) // 2
        measurement_chw = to_model(frame01[y:y + image_size, x:x + image_size])
        denoised_chw = denoiser.denoise(measurement_chw[None])[0]
    save_model_image(measurement_chw, out / f"{stem}_input.png")
    save_model_image(denoised_chw, out / f"{stem}_denoised.png")
    quantitative = ((denoised_chw[0].clamp(-1, 1).numpy() + 1) / 2).astype(np.float32)
    Image.fromarray(quantitative).save(out / f"{stem}_denoised.tif")
    typer.echo(f"wrote 2 preview PNG(s) and normalized float32 TIFF to {out}")

    if clean_path is not None:
        if full:
            clean_chw = to_model(load_measurement01(clean_path))
        else:
            clean = load_measurement01(clean_path)
            y, x = (clean.shape[0] - image_size) // 2, (clean.shape[1] - image_size) // 2
            clean_chw = to_model(clean[y:y + image_size, x:x + image_size])
        clean01 = ((clean_chw + 1) / 2).numpy().transpose(1, 2, 0)
        for label, tensor in (("input", measurement_chw), ("denoised", denoised_chw)):
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


@app.command("generate-drift")
def generate_drift_command(
    source: Path = typer.Option(..., help="Pixel-aligned burst dataset whose clean images (and split) are reused."),
    out: Path = typer.Option(..., help="Output dataset directory (burst/ + drift.json)."),
    frames: int = typer.Option(16, min=2, help="Frames per burst."),
    retakes: int = typer.Option(10, min=1, help="Independent bursts per held-out (val/test) source."),
    velocity_sigma: float = typer.Option(0.35, help="Stage drift velocity sigma, px/frame per axis."),
    walk_sigma: float = typer.Option(0.25, help="Random-walk drift increment sigma, px/frame per axis."),
    gain_sigma: float = typer.Option(0.02, help="Charging gain drift amplitude."),
    offset_sigma: float = typer.Option(0.005, help="Charging offset drift amplitude ([0, 1] units)."),
    peak: float = typer.Option(10.0, help="Poisson peak (must match the source dataset)."),
    seed: int = typer.Option(0, help="Generation seed."),
    val_fraction: float = typer.Option(0.1),
    test_fraction: float = typer.Option(0.1),
    split_seed: int = typer.Option(2019),
) -> None:
    """Render a drifting-burst dataset (stage drift, intra-frame shear,
    charging) from the clean images of an existing burst dataset, with the
    per-frame truth recorded in drift.json."""
    from .drift import DriftParams, generate_drift_dataset

    truth = generate_drift_dataset(
        source,
        out,
        frames_per_burst=frames,
        holdout_retakes=retakes,
        params=DriftParams(
            velocity_sigma=velocity_sigma,
            walk_sigma=walk_sigma,
            gain_sigma=gain_sigma,
            offset_sigma=offset_sigma,
        ),
        peak=peak,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_seed=split_seed,
        progress=lambda message: typer.echo(message, err=True),
    )
    typer.echo(
        f"wrote {out}: {len(truth['sources'])} sources, val {truth['split']['val_source_indices']}, "
        f"test {truth['split']['test_source_indices']}"
    )


@app.command()
def register(
    dataset: Path = typer.Option(..., help="Burst dataset (drifting or not) to register burst by burst."),
    out: Path = typer.Option(..., help="Output registration table (JSON)."),
    checkpoint: Optional[Path] = typer.Option(
        None, help="Single-frame checkpoint: register its denoised outputs instead of the raw frames."
    ),
    sigma: Optional[float] = typer.Option(None, help="Gaussian smoothing before registration (default 2 raw / 1 pre-denoised)."),
    radius: int = typer.Option(6, min=1, help="Coarse search radius around the previous frame's shift (px)."),
    frames_per_burst: Optional[int] = typer.Option(None, help="Burst length (default: drift.json, else all frames of a source)."),
    stride: int = typer.Option(48, min=1, help="Tile stride of the pre-denoiser."),
    device: str = typer.Option("auto", help="auto | cpu | cuda"),
) -> None:
    """Register every burst of a dataset to its first frame from the noisy
    frames alone (bounded-search cross-correlation + Gauss-Newton refinement
    + constant-velocity smoothing); reports the accuracy against drift.json
    when the dataset carries one."""
    import time

    import numpy as np
    from burst_diffusion.data import BurstCache

    from .distill import build_teacher, denoise_full_frame
    from .drift import burst_truths, load_drift_truth
    from .register import RegistrationTable, register_burst, registration_errors
    from .train import resolve_device

    resolved = resolve_device(device)
    truth = load_drift_truth(dataset)
    if frames_per_burst is None:
        frames_per_burst = int(truth["frames_per_burst"]) if truth is not None else 0
    predenoise = None
    method = "raw"
    if checkpoint is not None:
        import torch

        fn, description = build_teacher(checkpoint, device=device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        tile = int(payload["config"]["data"]["image_size"])
        predenoise = lambda frame01: denoise_full_frame(fn, frame01, tile=tile, stride=stride)  # noqa: E731
        method = f"predenoised:{checkpoint}"
        typer.echo(f"registering pre-denoised frames ({description})", err=True)
    effective_sigma = sigma if sigma is not None else (2.0 if predenoise is None else 1.0)
    cache = BurstCache(dataset, channels=1, min_replicas=1, min_size=1, val_fraction=0.0, test_fraction=0.0)
    bursts: dict[int, list] = {}
    errors: dict[str, list] = {"train": [], "holdout": []}
    started = time.time()
    for source in sorted(cache.all_sources, key=lambda item: item.source_index):
        length = frames_per_burst if frames_per_burst > 0 else len(source.frames)
        count = len(source.frames) // length
        trajectories = []
        truths = burst_truths(truth, source.source_index) if truth is not None else None
        split = truth["sources"][str(source.source_index)]["split"] if truth is not None else "train"
        for burst in range(count):
            frames01 = [f.astype(np.float64) / 255.0 for f in source.frames[burst * length : (burst + 1) * length]]
            trajectory = register_burst(
                frames01, sigma=effective_sigma, radius=radius, device=resolved, predenoise=predenoise
            )
            trajectories.append(trajectory)
            if truths is not None:
                errors[split].append(
                    {
                        "source_index": source.source_index,
                        "burst": burst,
                        **registration_errors(trajectory, truths[burst].position, truths[burst].velocity),
                    }
                )
        bursts[source.source_index] = trajectories
        typer.echo(
            f"source {source.source_index}: {count} burst(s) registered ({time.time() - started:.0f} s)",
            err=True,
        )
    table = RegistrationTable(
        method=method,
        sigma=effective_sigma,
        radius=radius,
        frames_per_burst=frames_per_burst if frames_per_burst > 0 else max(len(s.frames) for s in cache.all_sources),
        bursts=bursts,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.save(out)
    typer.echo(f"registration table written to {out}")
    if truth is not None:
        report = {}
        for split, items in errors.items():
            if not items:
                continue
            per_frame = np.concatenate([np.asarray(item["position_errors"]) for item in items])
            report[split] = {
                "bursts": len(items),
                "position_rms_px": np.sqrt((per_frame**2).mean(axis=0)).tolist(),
                "position_p95_px": np.percentile(np.abs(per_frame), 95.0, axis=0).tolist(),
                "position_max_px": np.abs(per_frame).max(axis=0).tolist(),
                "frames_worse_than_0.25px": float((np.abs(per_frame) > 0.25).any(axis=1).mean()),
                "per_burst": items,
            }
            typer.echo(
                f"{split}: {len(items)} bursts | position error rms (dy, dx) "
                f"{report[split]['position_rms_px'][0]:.3f}, {report[split]['position_rms_px'][1]:.3f} px | "
                f"p95 {report[split]['position_p95_px'][0]:.3f}, {report[split]['position_p95_px'][1]:.3f} | "
                f"max {report[split]['position_max_px'][0]:.2f}, {report[split]['position_max_px'][1]:.2f} | "
                f"frames > 0.25 px: {100 * report[split]['frames_worse_than_0.25px']:.1f}%"
            )
        report_path = out.with_suffix(".accuracy.json")
        report_path.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
        typer.echo(f"accuracy vs drift truth written to {report_path}")


def _fusion_burst_arms(
    fusion_checkpoint: list[str],
    *,
    fusion_frames: str,
    fusion_align: str,
    fusion_predenoise: Optional[Path],
    regavg: bool,
    frames_per_retake: int,
    dataset_dir: Path,
    device: str,
    stride: int,
    debias_peak: Optional[float] = None,
) -> dict:
    """Build a :class:`~edge_denoise.fusion.BurstFusionArms` per fusion checkpoint."""
    from .data import ClipDebiaser
    from .distill import build_teacher, denoise_full_frame
    from .drift import load_drift_truth
    from .fusion import BurstFusionArms, FusionDenoiser

    arms = _parse_arms(fusion_checkpoint, default_name="fuse")
    if not arms:
        return {}
    counts = [int(part) for part in fusion_frames.split(",") if part.strip()]
    truth = load_drift_truth(dataset_dir) if fusion_align == "truth" else None
    result = {}
    for name, path in arms.items():
        denoiser = FusionDenoiser.from_checkpoint(path, device=device)
        if debias_peak is not None:
            # Inference-time inverse of the clipped-Poisson response on the OUTPUT
            # (the network estimates the clipped mean g(x); g is monotone).
            denoiser._debias = ClipDebiaser(debias_peak)
        predenoise = None
        if fusion_predenoise is not None:
            fn, _ = build_teacher(fusion_predenoise, device=device)
            tile = denoiser.image_size
            predenoise = lambda frame01, fn=fn, tile=tile: denoise_full_frame(fn, frame01, tile=tile, stride=stride)  # noqa: E731
        result[name] = BurstFusionArms(
            denoiser,
            frame_counts=counts,
            frames_per_retake=frames_per_retake,
            align=fusion_align,
            regavg=regavg,
            predenoise=predenoise,
            truth=truth,
            stride=stride,
        )
    return result


@app.command()
def fuse(
    checkpoint: Path = typer.Option(..., help="Burst-fusion checkpoint."),
    dataset: Path = typer.Option(..., help="Burst dataset directory."),
    source_index: int = typer.Option(..., help="Source to fuse."),
    retake: int = typer.Option(0, min=0, help="Which burst (retake) of the source."),
    frames: str = typer.Option("1,4,16", help="Frame counts K to fuse."),
    out: Path = typer.Option(..., help="Output directory for the PNGs."),
    align: str = typer.Option("registered", help="registered | none | truth"),
    device: str = typer.Option("auto", help="auto | cpu | cuda"),
) -> None:
    """Fuse one burst at several frame counts and write the images next to the
    classical references (single frame, drifting average, registered average, clean)."""
    import numpy as np
    from burst_diffusion.data import BurstCache
    from PIL import Image

    from .fusion import frames_per_retake_of

    per_retake = frames_per_retake_of(dataset, default=0)
    cache = BurstCache(dataset, channels=1, min_replicas=1, min_size=1, val_fraction=0.0, test_fraction=0.0)
    matches = [s for s in cache.all_sources if s.source_index == source_index]
    if not matches:
        raise typer.BadParameter(f"source {source_index} not in {dataset}")
    source = matches[0]
    per_retake = per_retake or len(source.frames)
    arms = _fusion_burst_arms(
        [f"fuse={checkpoint}"],
        fusion_frames=frames,
        fusion_align=align,
        fusion_predenoise=None,
        regavg=True,
        frames_per_retake=per_retake,
        dataset_dir=dataset,
        device=device,
        stride=48,
    )["fuse"]
    outputs = arms.outputs(source, retake)
    burst = arms.retake_frames(source, retake)
    outputs["single_frame"] = burst[0]
    for count in arms.counts:
        outputs[f"avg{count}"] = np.mean(burst[:count], axis=0)
    outputs["clean"] = source.clean.astype(np.float64) / 255.0
    out.mkdir(parents=True, exist_ok=True)
    for name, image in outputs.items():
        Image.fromarray(np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)).save(out / f"{name}.png")
    typer.echo(f"wrote {len(outputs)} images to {out}")


@app.command("fine-features")
def fine_features_command(
    config: Path = typer.Option(..., help="edge_denoise YAML config (dataset section drives the split)."),
    checkpoint: list[str] = typer.Option(
        [], help="edge_denoise checkpoint arm as NAME=PATH (repeatable; a bare PATH is named 'edge')."
    ),
    burst_checkpoint: list[str] = typer.Option(
        [], help="burst_diffusion checkpoint arm as NAME=PATH (repeatable), one-shot prediction."
    ),
    out: Path = typer.Option(..., help="Output directory for fine_features.json, summary.md, feature_plate.png."),
    split: str = typer.Option("val", help="val | train | test (test is the locked holdout: report once)"),
    sources: Optional[str] = typer.Option(None, help="Comma-separated source indices (default: whole split)."),
    seeds: int = typer.Option(4, min=1, help="Retakes (frames) denoised per source for the retention spread."),
    stride: int = typer.Option(48, min=1, help="Tile stride for full-frame blended denoising."),
    peak: float = typer.Option(10.0, help="Effective Poisson peak of the dataset (feature SNR scale)."),
    prior_checkpoint: Optional[Path] = typer.Option(None, help="Diffusion prior checkpoint for --posterior-arm."),
    posterior_arm: list[str] = typer.Option(
        [], help="Posterior-sampling arm as NAME=mode[,steps=..,samples=..,guidance=..,clip=..,seed=..] (repeatable)."
    ),
    fusion_checkpoint: list[str] = typer.Option(
        [], help="Burst-fusion checkpoint arm as NAME=PATH (repeatable): rows fuse{K}@NAME (+ regavg{K}@NAME)."
    ),
    fusion_frames: str = typer.Option("1,2,4,8,16", help="Frame counts K of the fusion arms."),
    fusion_align: str = typer.Option("registered", help="registered | none | truth: how evaluated bursts are aligned."),
    fusion_predenoise: Optional[Path] = typer.Option(
        None, help="Single-frame checkpoint whose outputs the registration runs on instead of the raw frames."
    ),
    regavg: bool = typer.Option(True, "--regavg/--no-regavg", help="Also report the registered K-frame average."),
    retake_frames: Optional[int] = typer.Option(
        None, help="Frames per retake (default: the dataset's drift.json frames_per_burst, else 1)."
    ),
    fusion_debias_peak: Optional[float] = typer.Option(
        None, help="Apply the inverse clipped-Poisson response (this peak) to the fusion outputs."
    ),
    device: str = typer.Option("auto", help="auto | cpu | cuda"),
) -> None:
    """Scale-resolved fine-feature retention of every arm on full frames.

    Measures, in flat regions, the transfer gain/correlation of each
    difference-of-Gaussian band against the linear (Wiener) bound, and the
    retention of structured clean features (blemishes, scratches) per
    single-frame SNR, per retake, plus false-feature rates.
    """
    import sys

    from .config import load_config
    from .distill import build_teacher
    from .finefeat import fine_features as run_fine_features
    from .fusion import frames_per_retake_of
    from .provider import checkpoint_records

    edge_arms = _parse_arms(checkpoint, default_name="edge")
    burst_arms = _parse_arms(burst_checkpoint, default_name="burst")
    overlap = set(edge_arms) & set(burst_arms)
    if overlap:
        raise typer.BadParameter(f"arm name(s) used in both tiers: {sorted(overlap)}")
    if not edge_arms and not burst_arms and not posterior_arm and not fusion_checkpoint:
        raise typer.BadParameter(
            "provide at least one --checkpoint, --burst-checkpoint, --posterior-arm or --fusion-checkpoint"
        )
    arms = {}
    for name, path in {**burst_arms, **edge_arms}.items():
        arms[name], _ = build_teacher(path, device=device)
    for name, sampler in _posterior_arms(prior_checkpoint, posterior_arm, device=device, peak=peak).items():
        if name in arms:
            raise typer.BadParameter(f"arm name {name!r} used twice")
        arms[name] = sampler.denoise
    loaded = load_config(config)
    per_retake = retake_frames if retake_frames is not None else frames_per_retake_of(loaded.data.dataset_dir)
    fusion_arms = _fusion_burst_arms(
        fusion_checkpoint,
        fusion_frames=fusion_frames,
        fusion_align=fusion_align,
        fusion_predenoise=fusion_predenoise,
        regavg=regavg,
        frames_per_retake=per_retake,
        dataset_dir=loaded.data.dataset_dir,
        device=device,
        stride=stride,
        debias_peak=fusion_debias_peak,
    )
    burst_callables = {}
    for name, fusion in fusion_arms.items():
        if name in arms:
            raise typer.BadParameter(f"arm name {name!r} used twice")
        for method, fn in fusion.burst_arms().items():
            burst_callables[f"{method}@{name}"] = fn
    wanted = None if sources is None else [int(part) for part in sources.split(",") if part.strip()]
    results = run_fine_features(
        loaded,
        arms,
        out_dir=out,
        split=split,
        sources=wanted,
        num_seeds=seeds,
        stride=stride,
        peak=peak,
        extra_metadata={
            "provider_checkpoints": checkpoint_records(edge_arms),
            "burst_checkpoints": {name: str(path) for name, path in burst_arms.items()},
            "fusion_checkpoints": checkpoint_records(_parse_arms(fusion_checkpoint, default_name="fuse")),
            "fusion_frames": fusion_frames,
            "fusion_align": fusion_align,
            "fusion_predenoise": None if fusion_predenoise is None else str(fusion_predenoise),
            "command": " ".join(sys.argv),
        },
        progress=lambda message: typer.echo(message, err=True),
        burst_arms=burst_callables,
        frames_per_retake=per_retake,
    )
    for name, method in results["summary"]["methods"].items():
        gains = " ".join(f"{g:.2f}" if g is not None else "-" for g in method["band_gain"])
        retention = method["feature_retention_median"]
        typer.echo(
            f"{name:>22}: band gain [{gains}] | feature retention median "
            f"{'-' if retention is None else f'{retention:.3f}'} | false/1000px "
            f"{method['false_features_per_1000_flat_px']:.2f}"
        )
    typer.echo(f"results written to {Path(out) / 'fine_features.json'}")


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
    prior_checkpoint: Optional[Path] = typer.Option(None, help="Diffusion prior checkpoint for --posterior-arm."),
    posterior_arm: list[str] = typer.Option(
        [], help="Posterior-sampling arm as NAME=mode[,steps=..,samples=..,guidance=..,clip=..,seed=..] (repeatable)."
    ),
    peak: float = typer.Option(10.0, help="Effective Poisson peak (posterior likelihood)."),
    fusion_checkpoint: list[str] = typer.Option(
        [], help="Burst-fusion checkpoint arm as NAME=PATH (repeatable): rows fuse{K}@NAME (+ regavg{K}@NAME)."
    ),
    fusion_frames: str = typer.Option("1,2,4,8,16", help="Frame counts K of the fusion arms."),
    fusion_align: str = typer.Option("registered", help="registered | none | truth: how evaluated bursts are aligned."),
    fusion_predenoise: Optional[Path] = typer.Option(
        None, help="Single-frame checkpoint whose outputs the registration runs on instead of the raw frames."
    ),
    regavg: bool = typer.Option(True, "--regavg/--no-regavg", help="Also report the registered K-frame average."),
    retake_frames: Optional[int] = typer.Option(
        None, help="Frames per retake = seed stride (default: the dataset's drift.json frames_per_burst, else 1)."
    ),
    fusion_debias_peak: Optional[float] = typer.Option(
        None, help="Apply the inverse clipped-Poisson response (this peak) to the fusion outputs."
    ),
    device: str = typer.Option("auto", help="auto | cpu | cuda"),
) -> None:
    """One metrology-precision table: classical + edge_denoise (+ burst) arms.

    Runs burst_diffusion's repeatability harness (CD sites, subpixel
    crossings, pooled c4-debiased sigmas) with edge arms supplied as
    realization providers, so every method is measured on identical sources,
    seeds, crops, and CD sites.  On a drifting dataset the seeds are the
    first frames of consecutive retakes and burst-fusion arms fuse each
    retake's first K frames.
    """
    from burst_diffusion.config import Config as BurstConfig
    from burst_diffusion.repeatability import repeatability as run_repeatability
    from burst_diffusion.train import load_checkpoint as load_burst_checkpoint

    import sys

    from .config import load_config
    from .fusion import frames_per_retake_of
    from .provider import callable_provider, checkpoint_records, providers_from_checkpoints

    edge_arms = _parse_arms(checkpoint, default_name="edge")
    burst_arms = _parse_arms(burst_checkpoint, default_name="burst")
    overlap = set(edge_arms) & set(burst_arms)
    if overlap:
        raise typer.BadParameter(f"arm name(s) used in both tiers: {sorted(overlap)}")
    posterior = _posterior_arms(prior_checkpoint, posterior_arm, device=device, peak=peak)
    overlap = set(posterior) & (set(edge_arms) | set(burst_arms))
    if overlap:
        raise typer.BadParameter(f"posterior arm name(s) already used: {sorted(overlap)}")

    loaded = load_config(config)
    per_retake = retake_frames if retake_frames is not None else frames_per_retake_of(loaded.data.dataset_dir)
    fusion_arms = _fusion_burst_arms(
        fusion_checkpoint,
        fusion_frames=fusion_frames,
        fusion_align=fusion_align,
        fusion_predenoise=fusion_predenoise,
        regavg=regavg,
        frames_per_retake=per_retake,
        dataset_dir=loaded.data.dataset_dir,
        device=device,
        stride=48,
        debias_peak=fusion_debias_peak,
    )
    overlap = set(fusion_arms) & (set(edge_arms) | set(burst_arms) | set(posterior))
    if overlap:
        raise typer.BadParameter(f"fusion arm name(s) already used: {sorted(overlap)}")

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
    for name, sampler in posterior.items():
        providers[name] = callable_provider(sampler.denoise01)
    for name, fusion in fusion_arms.items():
        providers[name] = fusion.provider()
    results = run_repeatability(
        bridge,
        burst_arms,
        out_dir=out,
        split=split,
        limit=limit,
        num_seeds=seeds,
        device=device,
        extra_providers=providers,
        seed_stride=per_retake,
        # Bind every edge arm to the exact checkpoint evaluated (path + hash +
        # step) and keep the invocation: the harness itself knows providers
        # by name only.
        extra_metadata={
            "provider_checkpoints": checkpoint_records(edge_arms),
            "posterior_arms": {
                "prior_checkpoint": None if prior_checkpoint is None else str(prior_checkpoint),
                "specs": list(posterior_arm),
            },
            "fusion_checkpoints": checkpoint_records(_parse_arms(fusion_checkpoint, default_name="fuse")),
            "fusion_frames": fusion_frames,
            "fusion_align": fusion_align,
            "fusion_predenoise": None if fusion_predenoise is None else str(fusion_predenoise),
            "command": " ".join(sys.argv),
        },
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
