"""Command line interface.

A thin shell over :func:`sem_segment.pipeline.segment_image`: resolve the
config, load the image, measure, write.  Anything a caller might want to do
differently - aggregate across images, feed arrays from memory, keep results in
a notebook - is available from the API without going through here.

Heavy imports are function-local throughout, so ``--help`` and the config check
never import torch, transformers, or matplotlib.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import typer

app = typer.Typer(
    name="sem_segment",
    help="Segment SEM images with SAM 3, extract contours, and measure them.",
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


def _parse_crop(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    parts = [p for p in text.replace(" ", "").split(",") if p]
    if len(parts) != 4:
        raise typer.BadParameter(f"crop must be y0,y1,x0,x1 (four integers), got {text!r}")
    try:
        values = tuple(int(p) for p in parts)
    except ValueError as error:
        raise typer.BadParameter(f"crop values must be integers, got {text!r}") from error
    return values  # type: ignore[return-value]


@app.command()
def segment(
    config: Path = typer.Option(..., help="YAML config path."),
    input: Path = typer.Option(..., help="Image file, or a folder of images."),
    out: Path = typer.Option(..., help="Output directory; one subdirectory per image."),
    backend: str | None = typer.Option(None, help="Override segmentation.backend."),
    crop: str | None = typer.Option(None, help="y0,y1,x0,x1 applied before the grayscale check."),
    pixel_size_nm: float | None = typer.Option(None, help="Add nanometre columns to the output."),
    device: str | None = typer.Option(None, help="Override segmentation.device."),
    estimator: str | None = typer.Option(None, help="Override refine.estimator."),
    overwrite: bool = typer.Option(False, help="Allow writing into existing result directories."),
) -> None:
    """Segment, contour, refine and measure one image or a folder of images."""
    from pydantic import ValidationError

    from .config import Config, load_config
    from .image_io import discover_images, load_image
    from .pipeline import segment_image
    from .writers import REPORT_NAME, write_result

    if not config.is_file():
        raise typer.BadParameter(f"config not found: {config}")
    loaded = load_config(config)

    raw = loaded.model_dump()
    if backend is not None:
        raw["segmentation"]["backend"] = backend
    if device is not None:
        raw["segmentation"]["device"] = device
    if estimator is not None:
        raw["refine"]["estimator"] = estimator
    if crop is not None:
        raw["input"]["crop"] = _parse_crop(crop)
    if pixel_size_nm is not None:
        raw["input"]["pixel_size_nm"] = pixel_size_nm
    try:
        # Round-trip through the model so an override can never bypass validation.
        loaded = Config.model_validate(raw)
    except ValidationError as error:
        raise typer.BadParameter(str(error)) from error

    try:
        paths = discover_images(input)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    failures: list[tuple[Path, str]] = []
    for index, path in enumerate(paths, start=1):
        typer.echo(f"[{index}/{len(paths)}] {path.name}", err=True)
        destination = Path(out) / path.stem
        try:
            image = load_image(
                path,
                crop=loaded.input.crop,
                black_level=loaded.input.black_level,
                white_level=loaded.input.white_level,
            )
            result = segment_image(image.measure01, loaded)
            write_result(
                destination,
                result,
                pixel_size_nm=loaded.input.pixel_size_nm,
                source=path,
                extra_provenance={
                    "source": str(path),
                    "file_sha256": image.file_sha256,
                    "pixel_sha256": image.pixel_sha256,
                    "crop": list(image.crop) if image.crop else None,
                },
                exist_ok=overwrite,
            )
            if loaded.report.enabled:
                from .report import write_report

                write_report(destination / REPORT_NAME, image.measure01, result, loaded,
                             title=f"SEM segmentation - {path.name}")
            typer.echo(
                f"    {result.region_count} regions -> {destination}"
                + (f" ({REPORT_NAME})" if loaded.report.enabled else ""),
                err=True,
            )
            for warning in result.diagnostics.warnings:
                typer.echo(f"    warning: {warning}", err=True)
        except FileExistsError:
            failures.append((path, f"result directory already exists: {destination} (use --overwrite)"))
        except (ValueError, OSError, RuntimeError) as error:
            failures.append((path, str(error)))
        if failures and failures[-1][0] == path:
            typer.echo(f"    failed: {failures[-1][1]}", err=True)

    if failures:
        typer.echo(f"\n{len(failures)} of {len(paths)} image(s) failed.", err=True)
        raise typer.Exit(code=1)


@app.command()
def backends() -> None:
    """Report which backends can run here, and why the others cannot.

    Written for the remote machine: it turns an authentication failure buried in
    the middle of a batch run into one line of output before the run starts.
    """
    from .backends import available_backends
    from .weights import backend_readiness

    typer.echo(f"{'backend':<16}{'status':<12}{'detail'}")
    typer.echo("-" * 78)
    for name in available_backends():
        ready, detail = backend_readiness(name)
        typer.echo(f"{name:<16}{'ready' if ready else 'unavailable':<12}{detail}")

    from .weights import _token, _token_source

    signed_in = "yes" if _token() else "no"
    offline = os.environ.get("HF_HUB_OFFLINE", "0") not in ("0", "", "false", "False")
    typer.echo(
        f"\nsigned in: {signed_in} (via {_token_source()})   HF_HUB_OFFLINE: {offline}   "
        f"HF_HOME: {os.environ.get('HF_HOME', '(default)')}"
    )


@app.command("download-weights")
def download_weights(
    model_id: str = typer.Option("facebook/sam3", help="Hugging Face repository id."),
    dest: Path | None = typer.Option(None, help="Download into this directory instead of the cache."),
    revision: str | None = typer.Option(None, help="Pin a specific revision."),
    all_files: bool = typer.Option(
        False, help="Mirror the whole repo, including the original-format checkpoint."
    ),
) -> None:
    """Fetch model weights explicitly.

    Nothing in this package downloads implicitly. A run either finds the weights
    already present or tells you exactly how to get them, so a pipeline never
    silently pulls gigabytes in the middle of a batch.

    The original-format checkpoint is skipped by default: facebook/sam3 ships
    the model twice and transformers only reads the safetensors copy, so this
    halves the transfer from 6.9 GB to 3.4 GB.
    """
    from .weights import GatedRepositoryError, MissingDependency, fetch_weights

    try:
        path = fetch_weights(model_id, dest=dest, revision=revision, all_files=all_files)
    except (GatedRepositoryError, MissingDependency) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"weights ready at: {path}")
    typer.echo("Point a config at it with segmentation.model_path, or leave model_id and let the cache serve it.")
