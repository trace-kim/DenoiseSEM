"""Ordered real-SEM training with per-pipeline failure continuation and resume.

The suite launches the ordinary training CLI in separate processes. Checkpoints
and a durable manifest, rather than log messages or directory existence, decide
which work has completed. No scheduler GPU visibility is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import yaml

from burst_diffusion.data import BurstCache, resolve_burst_dir
from burst_diffusion.provenance import file_sha256, git_state
from runctl.bundles import atomic_write_text
from runctl.control import StopController
from runctl.run_logging import atomic_write_json, utc_now

from .config import Config, load_config
from .train import LATEST_CHECKPOINT_NAME, existing_run_artifacts, load_checkpoint

ROOT = Path(__file__).resolve().parent.parent
PIPELINES = ("ft_noisy", "ft_consist", "grad", "hybrid", "ft_grad_consist")
DEFAULT_TEST_SITE = Path("/data/260904_raw_data/test/260904_0947-13")


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _content_splits(metadata: dict) -> dict:
    return {split: sorted(sorted(frame["sha256"] for frame in site["frames"])
                          for site in metadata["sites"] if site["split"] == split)
            for split in ("train", "val", "test")}


def _step_overrides(entries: list[str]) -> dict[str, int]:
    values = {}
    for entry in entries:
        name, separator, number = entry.partition("=")
        if not separator or name not in PIPELINES or name in values:
            raise ValueError("--steps requires a unique PIPELINE=POSITIVE_INTEGER")
        try:
            value = int(number)
        except ValueError as error:
            raise ValueError("--steps requires a unique PIPELINE=POSITIVE_INTEGER") from error
        if value < 1:
            raise ValueError("--steps must be positive")
        values[name] = value
    return values


def build_plan(args: argparse.Namespace) -> dict:
    """Read-only preflight: verify prepared content, teacher and all recipes."""
    if not re.fullmatch(r"\d{6}", args.date):
        raise ValueError("--date must be YYMMDD")
    datetime.strptime(args.date, "%y%m%d")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("the sequential suite uses one worker; launch without torchrun")
    steps = _step_overrides(args.steps)
    teacher_path = args.n2n_checkpoint.resolve()
    teacher = load_checkpoint(teacher_path)
    teacher_config = Config.model_validate(teacher["config"])
    objective = teacher_config.objective
    if (objective.representation != "image" or objective.target != "noisy"
            or objective.lambda_image <= 0 or objective.lambda_gradient != 0
            or objective.lambda_consistency != 0 or objective.fusion is not None
            or not teacher.get("dataset_fingerprint")):
        raise ValueError("--n2n-checkpoint must be a real, single-frame image Noise2Noise checkpoint")
    dataset = args.dataset_dir.resolve()
    cache = BurstCache(dataset, min_replicas=3, min_size=teacher_config.data.image_size)
    if cache.real_metadata is None or cache.real_metadata["registration"]["mode"] != "none":
        raise ValueError("suite requires a prepared real dataset from prepare-real --align none")
    if not cache.val_sources:
        raise ValueError("suite requires held-out validation sites")
    if any(source.frames.dtype != np.uint8 for source in cache.all_sources):
        raise ValueError("this real-SEM suite requires the user's native uint8 acquisitions")
    if teacher["dataset_fingerprint"] == cache.real_fingerprint:
        teacher_manifest_path = cache.burst_dir / "real_dataset.json"
    else:
        teacher_manifest_path = args.teacher_manifest
        if teacher_manifest_path is None:
            teacher_manifest_path = resolve_burst_dir(teacher_config.data.dataset_dir) / "real_dataset.json"
    teacher_manifest_path = teacher_manifest_path.resolve()
    if file_sha256(teacher_manifest_path) != teacher["dataset_fingerprint"]:
        raise ValueError("teacher manifest differs from its checkpoint fingerprint")
    teacher_metadata = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    if (teacher_metadata.get("kind") != "real_sem"
            or _content_splits(teacher_metadata) != _content_splits(cache.real_metadata)):
        raise ValueError("teacher and suite must use the same acquisition content and site splits")
    levels = cache.real_metadata["normalization"]
    if (teacher_metadata["normalization"] != levels
            or (teacher_config.data.black_level, teacher_config.data.white_level) != (levels["black"], levels["white"])):
        raise ValueError("teacher and suite normalization differ")
    run_root = args.run_root.resolve()
    suite_dir = run_root / f"{args.date}_real_suite"
    matching_cache = suite_dir / "real_matching.json"
    pipelines = []
    for name in PIPELINES:
        template = load_config(args.config_dir / f"sem_real_{name}.yml")
        raw = template.model_dump(mode="json")
        # All continuations use the supplied real teacher's actual backbone,
        # including attention settings; scratch gradient uses equal capacity.
        raw["model"] = teacher_config.model.model_dump(mode="json")
        raw["data"].update(dataset_dir=str(dataset), image_size=teacher_config.data.image_size,
                           black_level=levels["black"], white_level=levels["white"],
                           real_matching_cache=str(matching_cache))
        if args.registration_failure is not None:
            raw["data"]["real_matching"]["registration_failure"] = args.registration_failure
        raw["training"].update(
            run_dir=str(run_root / f"{args.date}_real_{name}_affine_percentile"),
            init_checkpoint=None if name == "grad" else str(teacher_path),
            device=args.device, cpu_threads=args.cpu_threads, profile=args.profile,
        )
        for field in ("max_steps", "batch_size", "accumulation_steps", "precision", "lr", "checkpoint_every"):
            value = getattr(args, field)
            if value is not None:
                raw["training"][field] = value
        if name in steps:
            raw["training"]["max_steps"] = steps[name]
        config = Config.model_validate(raw)
        if (config.data.real_matching.registration, config.data.real_matching.brightness) != ("affine", "percentile"):
            raise ValueError("all suite recipes must retain affine/percentile matching")
        pipelines.append({"name": name, "config": config.model_dump(mode="json"),
                          "config_path": str(suite_dir / "configs" / f"{name}.yml")})
    matching = teacher_config.data.real_matching
    return {
        "version": 1, "date": args.date, "suite_dir": str(suite_dir),
        "dataset_fingerprint": cache.real_fingerprint,
        "teacher": {"path": str(teacher_path), "sha256": file_sha256(teacher_path),
                    "step": int(teacher["step"]), "prepared_manifest": str(teacher_manifest_path),
                    "registration": matching.registration if matching else teacher_metadata["registration"]["mode"],
                    "brightness": matching.brightness if matching else "none"},
        "pipelines": pipelines,
        "site_dir": str(args.site_dir.resolve()),
        "output_dir": str((args.output_dir or ROOT / "output" / f"{args.date}_real_models_comparison").resolve()),
        "metrology_device": args.metrology_device,
    }


def training_command(pipeline: dict, *, resume: bool = False) -> list[str]:
    command = [sys.executable, "-u", "-m", "edge_denoise", "train", "--config", pipeline["config_path"]]
    return command + (["--resume"] if resume else [])


@contextmanager
def suite_lock(path: Path) -> Iterator[None]:
    """OS lock automatically released on process exit, including crashes."""
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("this training suite is already running") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _checkpoint_step(path: Path, pipeline: dict, fingerprint: str) -> int:
    payload = load_checkpoint(path)
    if payload.get("dataset_fingerprint") != fingerprint:
        raise ValueError("saved checkpoint dataset differs from suite")
    if Config.model_validate(payload["config"]).model_dump(mode="json") != pipeline["config"]:
        raise ValueError("saved checkpoint configuration differs from suite")
    return int(payload["step"])


def _archive_uncheckpointed(run_dir: Path, attempt: int) -> None:
    """Retain all previous training artifacts when startup failed before a save."""
    artifacts = existing_run_artifacts(run_dir)
    if not artifacts:
        return
    archive = run_dir / "failed_attempts" / f"before_attempt_{attempt:03d}"
    if not archive.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("failed-attempt archive escapes its run directory")
    archive.mkdir(parents=True, exist_ok=False)
    for path in artifacts:
        if not path.resolve().is_relative_to(run_dir.resolve()):
            raise ValueError("run artifact escapes its directory")
        path.rename(archive / path.name)


def _launch(command: list[str], log_path: Path, run_dir: Path, stop: StopController,
            cpu_threads: int) -> int:
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS=str(cpu_threads), MKL_NUM_THREADS=str(cpu_threads),
                       OPENBLAS_NUM_THREADS=str(cpu_threads), PYTHONUNBUFFERED="1")
    with log_path.open("w", encoding="utf-8") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT) as process:
            while True:
                try:
                    return process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    if stop.is_requested():
                        # Keep the marker present even if initialization has
                        # not yet reached Trainer's normal marker cleanup.
                        (run_dir / "stop").touch()


def _write_comparison(plan: dict) -> Path:
    recipe = yaml.safe_load((ROOT / "edge_denoise/configs/sem_real_compare.yml").read_text(encoding="utf-8"))
    recipe["checkpoints"] = {"n2n": {"checkpoint": plan["teacher"]["path"],
                                     "prepared_manifest": plan["teacher"]["prepared_manifest"]}}
    for pipeline in plan["pipelines"]:
        recipe["checkpoints"][pipeline["name"]] = {
            "checkpoint": str(Path(pipeline["config"]["training"]["run_dir"]) / LATEST_CHECKPOINT_NAME)}
    recipe["sites"] = {Path(plan["site_dir"]).name: plan["site_dir"]}
    recipe["output_dir"] = plan["output_dir"]
    recipe["device"] = plan["pipelines"][0]["config"]["training"]["device"]
    recipe["metrology_device"] = plan["metrology_device"]
    recipe["tensorboard"] = False
    path = Path(plan["suite_dir"]) / "comparison.yml"
    atomic_write_text(path, yaml.safe_dump(recipe, sort_keys=False))
    return path


def run_suite(plan: dict, *, resume: bool = False) -> int:
    suite_dir = Path(plan["suite_dir"])
    suite_dir.mkdir(parents=True, exist_ok=True)
    with suite_lock(suite_dir / "suite.lock"):
        return _run_locked(plan, resume=resume)


def _run_locked(plan: dict, *, resume: bool) -> int:
    suite_dir = Path(plan["suite_dir"])
    manifest_path = suite_dir / "suite.json"
    identity = _digest(plan)
    if manifest_path.exists():
        if not resume:
            raise ValueError("suite already exists; repeat the command with --resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("plan_sha256") != identity or _digest(manifest.get("plan")) != identity:
            raise ValueError("resume plan changed (dataset, teacher, recipes, paths or budgets); restore original arguments")
        for record in manifest["pipelines"].values():
            for attempt in record["attempts"]:
                if attempt.get("status") == "running":
                    attempt.update(status="interrupted", error="Previous suite exited without recording a result")
    else:
        if resume:
            raise ValueError("no suite manifest to resume; omit --resume for a new suite")
        manifest = {"plan": plan, "plan_sha256": identity, "created_at": utc_now(),
                    "status": "pending", "pipelines": {p["name"]: {"status": "pending", "owned": False, "attempts": []}
                                                          for p in plan["pipelines"]}}
    atomic_write_json(manifest_path, manifest)
    comparison_path = _write_comparison(plan)
    stop = StopController(suite_dir / "stop")
    if resume:
        stop.stop_file.unlink(missing_ok=True)
    handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        handlers[signum] = signal.signal(signum, lambda number, frame: stop.request(f"signal {number}"))
    try:
        manifest["status"] = "running"
        for pipeline in plan["pipelines"]:
            if stop.is_requested():
                break
            name = pipeline["name"]
            record = manifest["pipelines"][name]
            run_dir = Path(pipeline["config"]["training"]["run_dir"])
            checkpoint = run_dir / LATEST_CHECKPOINT_NAME
            attempt = {"started_at": utc_now(), "git": git_state(ROOT)}
            started = time.perf_counter()
            try:
                if record["status"] == "complete":
                    step = _checkpoint_step(checkpoint, pipeline, plan["dataset_fingerprint"])
                    if step < pipeline["config"]["training"]["max_steps"] or file_sha256(checkpoint) != record["checkpoint_sha256"]:
                        raise ValueError("completed checkpoint changed or no longer reaches its requested budget")
                    print(f"{name}: verified complete at step {step}", flush=True)
                    continue
                if file_sha256(Path(plan["teacher"]["path"])) != plan["teacher"]["sha256"]:
                    raise ValueError("N2N teacher changed since preflight")
                previous_attempts = len(record["attempts"])
                if not record.get("owned") and existing_run_artifacts(run_dir):
                    raise ValueError("run directory already belongs to another run; choose a new date/run root")
                if record.get("checkpoint_sha256") is not None and file_sha256(checkpoint) != record["checkpoint_sha256"]:
                    raise ValueError("previously completed checkpoint changed; restore it before resuming")
                continuing = checkpoint.exists()
                if continuing:
                    attempt["start_step"] = _checkpoint_step(checkpoint, pipeline, plan["dataset_fingerprint"])
                elif previous_attempts:
                    _archive_uncheckpointed(run_dir, previous_attempts + 1)
                    attempt["restart_reason"] = "no checkpoint; preserved previous artifacts and restarted initialization"
                run_dir.mkdir(parents=True, exist_ok=True)
                record["owned"] = True
                config_path = Path(pipeline["config_path"])
                atomic_write_text(config_path, yaml.safe_dump(pipeline["config"], sort_keys=False))
                command = training_command(pipeline, resume=continuing)
                log_path = run_dir / f"attempt_{previous_attempts + 1:03d}.log"
                attempt.update(command=command, log=str(log_path), status="running")
                record["attempts"].append(attempt)
                record["status"] = "running"
                status_path = run_dir / "training_status.json"
                if status_path.exists():
                    attempt["previous_training_status"] = json.loads(status_path.read_text(encoding="utf-8"))
                    status_path.unlink()
                atomic_write_json(manifest_path, manifest)
                print(f"{name}: {'resume' if continuing else 'start'}; log {log_path}", flush=True)
                returncode = _launch(command, log_path, run_dir, stop,
                                     pipeline["config"]["training"]["cpu_threads"])
                attempt["returncode"] = returncode
                status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
                if stop.is_requested() or status.get("status") == "stopped":
                    stop.request(status.get("stop_reason") or "suite interrupted")
                    record["status"] = attempt["status"] = "stopped"
                else:
                    if returncode != 0:
                        raise RuntimeError(f"training exited with code {returncode}; see {log_path}")
                    step = _checkpoint_step(checkpoint, pipeline, plan["dataset_fingerprint"])
                    if step < pipeline["config"]["training"]["max_steps"] or status.get("status") != "complete":
                        raise RuntimeError(f"training unfinished at step {step}; requested {pipeline['config']['training']['max_steps']}")
                    record.update(status="complete", step=step, checkpoint_sha256=file_sha256(checkpoint))
                    attempt["status"] = "complete"
                    record.pop("error", None)
                timings_path = run_dir / "timings.json"
                if timings_path.exists():
                    attempt["timings"] = json.loads(timings_path.read_text(encoding="utf-8"))
            except Exception as error:
                record["status"] = attempt["status"] = "failed"
                record["error"] = attempt["error"] = f"{type(error).__name__}: {error}"
                if not any(item is attempt for item in record["attempts"]):
                    record["attempts"].append(attempt)
                print(f"{name}: FAILED: {error}; continuing to next pipeline", flush=True)
            finally:
                attempt.update(finished_at=utc_now(), wall_seconds=time.perf_counter() - started)
                record["total_wall_seconds"] = sum(item.get("wall_seconds", 0) for item in record["attempts"])
                record["attempts_without_wall_time"] = sum("wall_seconds" not in item for item in record["attempts"])
                atomic_write_json(manifest_path, manifest)
        complete = all(row["status"] == "complete" for row in manifest["pipelines"].values())
        manifest.update(status="complete" if complete else "stopped" if stop.is_requested() else "incomplete",
                        updated_at=utc_now())
        atomic_write_json(manifest_path, manifest)
        for name, record in manifest["pipelines"].items():
            print(f"{name}: {record['status']}", flush=True)
        print(f"Suite {manifest['status']}: {manifest_path}", flush=True)
        if complete:
            print("Compare saved uint8 outputs: " + shlex.join([
                sys.executable, "tools/real_sem_compare.py", "--config", str(comparison_path)]), flush=True)
        return 0 if complete else 130 if stop.is_requested() else 1
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--n2n-checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--date", required=True, help="YYMMDD; repeat the same date on resume")
    parser.add_argument("--teacher-manifest", type=Path)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "edge_denoise/configs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-steps", type=int, help="Override all budgets, useful for short pilots")
    parser.add_argument("--steps", action="append", default=[], metavar="PIPELINE=N")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--lr", type=float)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--registration-failure", choices=("error", "skip"))
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_TEST_SITE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metrology-device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="Verify inputs and print resolved plans without writing")
    parser.add_argument("--resume", action="store_true", help="Verify completed runs and retry/resume unfinished pipelines")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            for pipeline in plan["pipelines"]:
                print(shlex.join(training_command(pipeline)))
            return 0
        return run_suite(plan, resume=args.resume)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"Suite preflight failed: {error}", file=sys.stderr)
        return 1
