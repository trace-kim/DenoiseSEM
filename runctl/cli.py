"""Human-facing command line for reliable, reproducible experiments.

The command surface is flow-agnostic.  Settings that every flow shares are
typed options (``--batch-size``, ``--max-steps``, ...); settings that belong to
one pipeline are supplied as ``--set NAME=VALUE`` and validated against that
flow's spec model, so a misspelled or wrongly typed setting still fails loudly
instead of being silently ignored.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from .registry import Flow, FlowError, available_flows, get_flow
from .schemas import (
    BaseTrainingSpec,
    ExecutorType,
    MachineProfile,
    ReproducibilityMode,
    RunManifest,
    SlurmResources,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLOW = "ddim"
console = Console()


def resolve_flow(name: str | None = None) -> Flow:
    """Look up a flow, turning registry errors into CLI-shaped failures."""

    try:
        return get_flow(name or DEFAULT_FLOW)
    except FlowError as error:
        raise typer.BadParameter(str(error)) from error


def flow_default_config(flow: Flow) -> Path:
    """The flow's single active recipe, preferring an installed copy."""

    configured = flow.default_config
    if configured is not None and Path(configured).is_file():
        return Path(configured)
    installed = Path(sys.prefix) / "share" / "runctl" / flow.name / "sem.yml"
    if installed.is_file():
        return installed
    if configured is not None:
        return Path(configured)
    raise typer.BadParameter(f"flow {flow.name!r} does not define a default config")


def _default_config_for_help() -> Path | None:
    """Best-effort default shown in ``--help`` without importing torch."""

    try:
        return flow_default_config(get_flow(DEFAULT_FLOW))
    except Exception:
        return None


DEFAULT_CONFIG = _default_config_for_help()

app = typer.Typer(
    name="runctl",
    help="Plan, launch, inspect, and publish reproducible training runs.",
    no_args_is_help=True,
    suggest_commands=True,
)
machine_app = typer.Typer(help="Configure operational machine profiles.", no_args_is_help=True)
train_app = typer.Typer(help="Plan and launch SEM training.", no_args_is_help=True)
run_app = typer.Typer(help="Inspect and control run bundles.", no_args_is_help=True)
track_app = typer.Typer(help="Publish portable run bundles to local tracking.", no_args_is_help=True)
environment_app = typer.Typer(
    help="Build and verify target-native offline dependency bundles.", no_args_is_help=True
)
app.add_typer(machine_app, name="machine")
app.add_typer(train_app, name="train")
app.add_typer(run_app, name="run")
app.add_typer(track_app, name="track")
app.add_typer(environment_app, name="environment")


def _reject_duplicate_scalar_options(argv: list[str]) -> None:
    """Click normally accepts the rightmost duplicate; training must not."""

    seen: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        # Positive/negative boolean spellings are one scalar setting.  Click
        # otherwise accepts both and silently lets the rightmost value win.
        if option == "--no-cache-in-memory":
            option = "--cache-in-memory"
        if option in {"--help", "--version"}:
            continue
        # --set is the repeatable escape hatch for flow-specific fields; each
        # NAME is checked for duplication separately in _parse_set_overrides.
        if option == "--set":
            continue
        if option in seen:
            raise typer.BadParameter(f"duplicate scalar option {option!r} is not allowed")
        seen.add(option)


@app.callback()
def root_callback() -> None:
    """Reliable experiment management for the SEM DDIM workflow."""

    _reject_duplicate_scalar_options(sys.argv[1:])


def _bundle_api() -> Any:
    from . import bundles

    return bundles


def _profile_api() -> Any:
    from . import profiles

    return profiles


def _parse_int_tuple(value: Optional[str], option: str) -> Optional[tuple[int, ...]]:
    if value is None:
        return None
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise typer.BadParameter(f"{option} must be comma-separated integers") from error
    if not parsed:
        raise typer.BadParameter(f"{option} must not be empty")
    return parsed


def _parse_set_overrides(
    values: Optional[list[str]],
    spec_model: type[BaseTrainingSpec],
) -> dict[str, Any]:
    """Parse repeatable ``--set NAME=VALUE`` pairs into spec keyword arguments.

    ``VALUE`` uses YAML syntax so lists (``ch_mult=[1,2,2,2]``) and booleans
    read naturally.  Unknown names and repeated names are rejected here; type
    errors surface later from the spec model's strict validation.
    """

    if not values:
        return {}
    import yaml

    parsed: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects NAME=VALUE, got {item!r}")
        name, _, raw = item.partition("=")
        name = name.strip()
        if name not in spec_model.model_fields:
            known = ", ".join(sorted(spec_model.model_fields))
            raise typer.BadParameter(f"--set {name!r} is not a setting; choose from: {known}")
        if name in parsed:
            raise typer.BadParameter(f"duplicate --set for {name!r}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as error:
            raise typer.BadParameter(f"--set {name}: invalid YAML value {raw!r}") from error
        if isinstance(value, list):
            value = tuple(value)
        parsed[name] = value
    return parsed


def _render_set_value(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(str(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _spec_overrides(
    *,
    label: Optional[str] = None,
    dataset: Optional[str] = None,
    max_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    checkpoint_every: Optional[int] = None,
    checkpoint_minutes: Optional[int] = None,
    validation_every: Optional[int] = None,
    sample_every: Optional[int] = None,
    seed: Optional[int] = None,
    reproducibility: Optional[ReproducibilityMode] = None,
    num_workers: Optional[int] = None,
    cache_in_memory: Optional[bool] = None,
) -> dict[str, Any]:
    """Collect the settings every flow understands (see BaseTrainingSpec)."""

    values: dict[str, Any] = {
        "label": label,
        "dataset_alias": dataset,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "lr": learning_rate,
        "checkpoint_every": checkpoint_every,
        "checkpoint_minutes": checkpoint_minutes,
        "validation_every": validation_every,
        "sample_every": sample_every,
        "seed": seed,
        "reproducibility": reproducibility,
        "num_workers": num_workers,
        "cache_in_memory": cache_in_memory,
    }
    return {key: value for key, value in values.items() if value is not None}


#: General options restated on every canonical command line, paired with the
#: spec field they set.
GENERAL_ARGV_OPTIONS: tuple[tuple[str, str], ...] = (
    ("--label", "label"),
    ("--dataset", "dataset_alias"),
    ("--max-steps", "max_steps"),
    ("--batch-size", "batch_size"),
    ("--learning-rate", "lr"),
    ("--checkpoint-every", "checkpoint_every"),
    ("--checkpoint-minutes", "checkpoint_minutes"),
    ("--validation-every", "validation_every"),
    ("--sample-every", "sample_every"),
    ("--seed", "seed"),
    ("--reproducibility", "reproducibility"),
    ("--num-workers", "num_workers"),
)


def _canonical_argv(
    machine_id: str,
    config: Path,
    spec: BaseTrainingSpec,
    flow: Flow | None = None,
) -> tuple[str, ...]:
    """Render the exact command that reproduces this run.

    Every experiment-varying setting is restated explicitly: shared settings as
    typed flags, flow-specific settings as ``--set``.  Nothing is left implicit
    in the config file.
    """

    resolved = flow if flow is not None else resolve_flow(DEFAULT_FLOW)
    argv: list[str] = [
        "runctl", "train", "launch",
        "--flow", resolved.name,
        "--machine", machine_id,
        "--config", str(Path(config).resolve()),
    ]
    for option, field_name in GENERAL_ARGV_OPTIONS:
        value = getattr(spec, field_name)
        if isinstance(value, ReproducibilityMode):
            value = value.value
        elif isinstance(value, float):
            value = repr(value)
        argv.extend([option, str(value)])
    argv.append("--cache-in-memory" if spec.cache_in_memory else "--no-cache-in-memory")
    for option in resolved.options:
        argv.extend(["--set", f"{option.name}={_render_set_value(getattr(spec, option.name))}"])
    argv.append("--yes")
    return tuple(argv)


def _load_plan(
    machine_id: str,
    config: Path,
    overrides: dict[str, Any],
    flow: Flow | None = None,
) -> tuple[MachineProfile, BaseTrainingSpec, tuple[str, ...]]:
    profiles = _profile_api()
    bundles = _bundle_api()
    resolved = flow if flow is not None else resolve_flow(DEFAULT_FLOW)
    active_config = flow_default_config(resolved)
    if config.expanduser().resolve() != active_config.resolve():
        raise typer.BadParameter(
            f"new {resolved.name} runs use the single active config {active_config}; "
            "put experiment-varying choices in typed flags or --set"
        )
    profile = profiles.load_profile(machine_id)
    spec = bundles.load_training_spec(config, overrides=overrides, flow=resolved)
    if spec.dataset_alias not in profile.datasets:
        choices = ", ".join(sorted(profile.datasets))
        raise typer.BadParameter(
            f"dataset alias {spec.dataset_alias!r} is not in machine {machine_id!r}; available: {choices}"
        )
    dataset_path = profile.dataset_path(spec.dataset_alias).expanduser()
    if not dataset_path.is_dir():
        raise typer.BadParameter(f"dataset directory does not exist: {dataset_path}")
    argv = _canonical_argv(machine_id, config, spec, resolved)
    return profile, spec, argv


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return str(value)


def _show_plan(
    profile: MachineProfile,
    spec: BaseTrainingSpec,
    argv: tuple[str, ...],
    flow: Flow | None = None,
) -> None:
    table = Table(
        title=f"Resolved {flow.name if flow else DEFAULT_FLOW} training plan",
        show_header=False,
    )
    table.add_column("Setting", style="cyan")
    table.add_column("Value", overflow="fold")
    rows = (
        ("Machine", profile.machine_id),
        ("Executor", profile.executor.value),
        (
            "GPU",
            "visible index {}{}".format(
                profile.gpu_index,
                f", expected {profile.expected_gpu!r}" if profile.expected_gpu else "",
            ),
        ),
        ("Dataset", f"{spec.dataset_alias} -> {profile.dataset_path(spec.dataset_alias)}"),
        *(flow.rows_for(spec) if flow is not None else ()),
        ("Optimization", f"lr={spec.lr}, batch={spec.batch_size}, max_steps={spec.max_steps}"),
        ("Intervals", f"checkpoint={spec.checkpoint_every}/{spec.checkpoint_minutes}m, "
                      f"validation={spec.validation_every}, samples={spec.sample_every}"),
        ("Reproducibility", f"{spec.reproducibility.value}, seed={spec.seed}"),
        ("Runs root", str(profile.runs_root)),
        ("Config SHA-256", spec.config_sha256),
    )
    estimated = spec.estimate_checkpoint_bytes()
    for key, value in rows:
        table.add_row(key, str(value))
    if estimated is not None:
        table.add_row("Approx. checkpoint", _format_bytes(estimated))
    console.print(table)
    bundles = _bundle_api()
    posix = bundles.render_posix_command(argv)
    powershell = bundles.render_powershell_command(argv)
    console.print(Panel(Text(posix), title="Canonical POSIX command", border_style="green"))
    console.print(Panel(Text(powershell), title="Canonical PowerShell command", border_style="green"))


def _launch_bundle(
    profile: MachineProfile,
    spec: BaseTrainingSpec,
    argv: tuple[str, ...],
    *,
    confirmed: bool,
    parent_run_id: Optional[str] = None,
    flow: Flow | None = None,
) -> Path:
    if not confirmed:
        if not sys.stdin.isatty():
            raise typer.BadParameter("noninteractive launch requires --yes")
        if not Confirm.ask("Create and launch this immutable run?", default=False):
            raise typer.Abort()

    _launch_preflight(profile)

    bundles = _bundle_api()
    with console.status(
        "Fingerprinting the dataset and capturing an immutable source/environment snapshot..."
    ):
        run_dir, _manifest = bundles.create_run_bundle(
            REPOSITORY_ROOT,
            profile,
            spec,
            argv,
            parent_run_id=parent_run_id,
            flow=(flow.name if flow is not None else DEFAULT_FLOW),
        )
    console.print(f"Created run bundle: [bold]{run_dir}[/bold]")

    from .backends import get_backend

    source_dir = bundles.materialized_source_path(run_dir)
    attempt = 1
    attempt_dir = run_dir / "attempts" / f"{attempt:03d}"
    worker = (
        profile.python_executable,
        str(bundles.worker_bootstrap_path(run_dir)),
        "--run",
        str(run_dir.resolve()),
        "--attempt",
        str(attempt),
    )
    backend = get_backend(profile.executor.value)
    request = _make_launch_request(profile, run_dir, source_dir, attempt_dir, worker)
    _record_worker_command(run_dir, attempt, request)
    console.print(
        f"Starting through [bold]{profile.executor.value}[/bold]; logs: {attempt_dir}"
    )
    try:
        if profile.executor is ExecutorType.SLURM:
            submission = backend.submit(request, profile=profile)
        else:
            submission = backend.submit(request)
    except Exception as error:
        _record_submission_failure(run_dir, attempt, profile.executor.value, error)
        raise
    _record_submission(run_dir, attempt, submission)
    console.print(_submission_message(submission))
    if submission.state == "failed":
        raise RuntimeError(
            f"foreground worker failed with exit code {submission.metadata.get('returncode')}"
        )
    return run_dir


def _launch_preflight(profile: MachineProfile) -> None:
    """Repeat cheap safety checks immediately before an irreversible submission."""

    if not (Path(profile.python_executable).is_file() or shutil.which(profile.python_executable)):
        raise FileNotFoundError(f"configured Python executable was not found: {profile.python_executable}")
    from .backends import get_backend

    backend = get_backend(profile.executor.value)
    if not backend.available():
        raise RuntimeError(f"execution backend is unavailable: {profile.executor.value}")
    if profile.executor not in {ExecutorType.FOREGROUND, ExecutorType.WINDOWS_TASK}:
        return
    gpu_report = _probe_configured_gpu(profile)
    error = _gpu_validation_error(profile, gpu_report)
    if error is not None:
        raise RuntimeError(error)


def _probe_configured_gpu(profile: MachineProfile) -> dict[str, Any]:
    source = """
import json
import os
import sys

import torch

index = int(sys.argv[1])
available = bool(torch.cuda.is_available())
count = int(torch.cuda.device_count())
names = [torch.cuda.get_device_name(i) for i in range(count)]
selection_error = None
selected_name = None
kernel = None
if not available or count == 0:
    selection_error = "CUDA is unavailable; refusing to train on CPU"
elif index >= count:
    selection_error = (
        "configured gpu_index {} is out of range for {} visible CUDA GPU(s)"
        .format(index, count)
    )
else:
    device = torch.device("cuda", index)
    x = torch.ones((16, 16), device=device)
    kernel = float((x @ x).sum().item())
    selected_name = names[index]
print(json.dumps({
    "available": available,
    "count": count,
    "names": names,
    "configured_index": index,
    "selected_index": index if selection_error is None else None,
    "selected_name": selected_name,
    "selection_error": selection_error,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "kernel": kernel,
    "torch": torch.__version__,
}))
"""
    result = subprocess.run(
        [profile.python_executable, "-c", source, str(profile.gpu_index)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "configured Python GPU probe failed")
    try:
        report = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("configured Python returned an invalid GPU probe result") from error
    return report


def _gpu_validation_error(
    profile: MachineProfile, gpu_report: dict[str, Any]
) -> str | None:
    if not bool(gpu_report.get("available")) or int(gpu_report.get("count", 0)) < 1:
        return "CUDA is unavailable; refusing to submit an accidentally CPU-bound run"
    selection_error = gpu_report.get("selection_error")
    if selection_error:
        return str(selection_error)
    selected_name = gpu_report.get("selected_name")
    if not selected_name:
        return "configured Python GPU probe did not identify the selected CUDA GPU"
    if (
        profile.expected_gpu
        and profile.expected_gpu.casefold() not in str(selected_name).casefold()
    ):
        return (
            f"selected GPU {str(selected_name)!r} does not match profile expectation "
            f"{profile.expected_gpu!r}"
        )
    return None


def _gpu_report_detail(profile: MachineProfile, gpu_report: dict[str, Any]) -> str:
    count = int(gpu_report.get("count", 0))
    names = [str(name) for name in gpu_report.get("names", [])]
    details = [f"{count} visible CUDA GPU(s)"]
    selected_name = gpu_report.get("selected_name")
    selected_index = gpu_report.get("selected_index")
    if selected_name is not None and selected_index is not None:
        details.append(f"selected visible index {selected_index}: {selected_name}")
    elif names:
        details.append(", ".join(names))
    visible_devices = gpu_report.get("cuda_visible_devices")
    if visible_devices is not None:
        details.append(f"CUDA_VISIBLE_DEVICES={visible_devices!r}")
    error = _gpu_validation_error(profile, gpu_report)
    if error is not None:
        details.append(error)
    else:
        details.append("worker will isolate this GPU as logical cuda:0")
    return "; ".join(details)


def _make_launch_request(
    profile: MachineProfile,
    run_dir: Path,
    source_dir: Path,
    attempt_dir: Path,
    worker: tuple[str, ...],
) -> Any:
    from .backends import LaunchRequest

    resources: dict[str, Any] = {}
    if profile.slurm is not None:
        resources = {
            "partition": profile.slurm.partition,
            "account": profile.slurm.account,
            "qos": profile.slurm.qos,
            "time_limit": profile.slurm.time_limit,
            "cpus_per_task": profile.slurm.cpus_per_task,
            "memory": f"{profile.slurm.memory_gb}G" if profile.slurm.memory_gb else "32G",
            "gpus": profile.slurm.gpus,
        }
        resources = {key: value for key, value in resources.items() if value is not None}
    return LaunchRequest(
        argv=worker,
        cwd=source_dir,
        run_dir=run_dir,
        name=f"{profile.machine_id}-{run_dir.name}",
        stdout_path=attempt_dir / "stdout.log",
        stderr_path=attempt_dir / "stderr.log",
        resources=resources,
        expected_gpu=profile.expected_gpu,
    )


def _record_submission(run_dir: Path, attempt: int, submission: Any) -> None:
    bundles = _bundle_api()
    payload = submission.to_dict()
    bundles.atomic_write_json(run_dir / "backend.json", payload)
    bundles.atomic_write_json(run_dir / "attempts" / f"{attempt:03d}" / "backend.json", payload)

    # Do not overwrite a worker that won the race and already entered running.
    state_path = run_dir / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if state.get("status") == "prepared" and submission.state in {"submitted", "queued"}:
        state["status"] = submission.state
        state["backend_job_id"] = submission.job_id
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        bundles.atomic_write_json(state_path, state)


def _record_worker_command(run_dir: Path, attempt: int, request: Any) -> None:
    bundles = _bundle_api()
    attempt_dir = run_dir / "attempts" / f"{attempt:03d}"
    bundles.atomic_write_json(attempt_dir / "worker-argv.json", list(request.argv))
    bundles.atomic_write_json(
        attempt_dir / "launch-request.json",
        {
            "argv": list(request.argv),
            "cwd": str(request.cwd),
            "stdout": str(request.stdout_path),
            "stderr": str(request.stderr_path),
            "resources": dict(request.resources),
            "expected_gpu": request.expected_gpu,
        },
    )
    bundles.atomic_write_text(
        attempt_dir / "worker-command.sh",
        "#!/usr/bin/env sh\nset -eu\nexec "
        + bundles.render_posix_command(request.argv)
        + "\n",
    )
    bundles.atomic_write_text(
        attempt_dir / "worker-command.ps1",
        "$ErrorActionPreference = 'Stop'\n"
        + bundles.render_powershell_command(request.argv)
        + "\n",
    )


def _record_submission_failure(
    run_dir: Path, attempt: int, backend_name: str, error: Exception
) -> None:
    bundles = _bundle_api()
    timestamp = datetime.now(timezone.utc).isoformat()
    message = f"submission failed: {type(error).__name__}: {error}"
    backend = {
        "backend": backend_name,
        "state": "submission_failed",
        "job_id": None,
        "submitted_at": timestamp,
        "metadata": {"error": message},
    }
    bundles.atomic_write_json(run_dir / "backend.json", backend)
    bundles.atomic_write_json(
        run_dir / "attempts" / f"{attempt:03d}" / "backend.json", backend
    )
    bundles.atomic_write_json(
        run_dir / "state.json",
        {
            "schema_version": 1,
            "attempt": attempt,
            "status": "failed",
            "updated_at": timestamp,
            "started_at": None,
            "ended_at": timestamp,
            "heartbeat_at": None,
            "pid": None,
            "backend_job_id": None,
            "exit_code": None,
            "message": message,
        },
    )


def _submission_message(submission: Any) -> str:
    message = f"Backend {submission.backend}: {submission.state} ({submission.job_id})"
    instruction = submission.metadata.get("instruction") if submission.metadata else None
    return message + (f"\n{instruction}" if instruction else "")


def _common_plan(
    machine: str,
    config: Optional[Path],
    *,
    launch: bool,
    yes: bool,
    flow: str = DEFAULT_FLOW,
    set_values: Optional[list[str]] = None,
    **values: Any,
) -> None:
    try:
        resolved = resolve_flow(flow)
        active_config = Path(config) if config is not None else flow_default_config(resolved)
        overrides = _spec_overrides(**values)
        overrides.update(_parse_set_overrides(set_values, resolved.spec_model))
        profile, spec, argv = _load_plan(machine, active_config, overrides, resolved)
        _show_plan(profile, spec, argv, resolved)
        if launch:
            _launch_bundle(profile, spec, argv, confirmed=yes, flow=resolved)
    except (ValidationError, ValueError, KeyError, FileNotFoundError, RuntimeError, OSError) as error:
        console.print(f"[bold red]Training request failed:[/bold red] {error}")
        raise typer.Exit(2) from error


@machine_app.command("configure")
def configure_machine(
    machine_id: str = typer.Option(..., "--id", help="Stable machine profile identifier."),
) -> None:
    """Create or replace a secret-free operational machine profile."""

    default_executor = "windows-task" if os.name == "nt" else "external-hpc"
    executor_text = Prompt.ask(
        "Executor",
        choices=[item.value for item in ExecutorType],
        default=default_executor,
    )
    runs_root = Path(Prompt.ask("Runs directory", default=str(REPOSITORY_ROOT / "runs"))).expanduser()
    dataset_alias = Prompt.ask("SEM dataset alias", default="sem")
    dataset_path = Path(Prompt.ask("SEM image directory")).expanduser()
    python_executable = Prompt.ask("Python executable", default=sys.executable)
    timezone = Prompt.ask("Output timezone", default="Asia/Seoul")
    gpu_index = IntPrompt.ask("Visible GPU index for this run", default=0)
    expected_gpu = Prompt.ask(
        "Expected GPU name substring (optional, e.g. H100 or RTX A6000)", default=""
    )
    mlflow_tracking_uri = Prompt.ask(
        "Default MLflow tracking URI (optional; no embedded credentials)", default=""
    )
    executor = ExecutorType(executor_text)
    slurm = None
    if executor is ExecutorType.SLURM:
        slurm = SlurmResources(
            partition=Prompt.ask("Slurm partition", default="") or None,
            account=Prompt.ask("Slurm account", default="") or None,
            qos=Prompt.ask("Slurm QoS", default="") or None,
            time_limit=Prompt.ask("Time limit", default="24:00:00"),
            cpus_per_task=IntPrompt.ask("CPUs per task", default=4),
            memory_gb=IntPrompt.ask("Memory (GiB)", default=64),
            gpus=1,
        )
    try:
        profile = MachineProfile(
            machine_id=machine_id,
            executor=executor,
            runs_root=runs_root,
            datasets={dataset_alias: dataset_path},
            timezone=timezone,
            python_executable=python_executable,
            gpu_index=gpu_index,
            expected_gpu=expected_gpu or None,
            mlflow_tracking_uri=mlflow_tracking_uri or None,
            slurm=slurm,
        )
        profiles = _profile_api()
        destination = profiles.profile_path(machine_id)
        if destination.exists() and not Confirm.ask(
            f"Replace existing profile {destination}?", default=False
        ):
            raise typer.Abort()
        path = profiles.save_profile(profile, overwrite=True)
    except (ValidationError, ValueError, OSError) as error:
        console.print(f"[bold red]Profile was not saved:[/bold red] {error}")
        raise typer.Exit(2) from error
    console.print(f"Saved machine profile: [bold]{path}[/bold]")
    console.print("Run [bold]runctl doctor --machine {}[/bold] next.".format(machine_id))


@machine_app.command("list")
def list_machines() -> None:
    """List configured machine profiles."""

    profiles = _profile_api().list_profiles()
    if not profiles:
        console.print(
            "No machine profiles configured. Run "
            "[bold]runctl machine configure --id MACHINE_ID[/bold]."
        )
        return
    for machine_id in profiles:
        console.print(machine_id)


@machine_app.command("show")
def show_machine(machine_id: str = typer.Argument(..., metavar="MACHINE")) -> None:
    """Show one operational profile (never experiment hyperparameters)."""

    try:
        profile = _profile_api().load_profile(machine_id)
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[bold red]Cannot load profile:[/bold red] {error}")
        raise typer.Exit(2) from error
    console.print_json(profile.model_dump_json(indent=2))


@app.command("doctor")
def doctor(
    machine: str = typer.Option(..., "--machine"),
    flow: str = typer.Option(DEFAULT_FLOW, "--flow", help="Training flow to preflight."),
    export_hpc_probe: Optional[Path] = typer.Option(None, "--export-hpc-probe"),
    exercise_executor: bool = typer.Option(
        False,
        "--exercise-executor",
        help="Run a harmless durable-executor acceptance task.",
    ),
) -> None:
    """Verify that a target can safely execute the workflow."""

    try:
        profile = _profile_api().load_profile(machine)
    except Exception as error:
        console.print(f"[bold red]Cannot load profile:[/bold red] {error}")
        raise typer.Exit(2) from error

    checks: list[tuple[str, bool, str]] = []
    python_available = Path(profile.python_executable).is_file() or shutil.which(
        profile.python_executable
    ) is not None
    checks.append(("Python", python_available, profile.python_executable))
    for alias, path in profile.datasets.items():
        checks.append((f"Dataset {alias}", path.is_dir(), str(path)))
    try:
        profile.runs_root.mkdir(parents=True, exist_ok=True)
        probe = profile.runs_root / ".runctl-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(("Runs directory", True, str(profile.runs_root)))
    except OSError as error:
        checks.append(("Runs directory", False, str(error)))
    try:
        gpu_report = _probe_configured_gpu(profile)
        error = _gpu_validation_error(profile, gpu_report)
        checks.append(("CUDA", error is None, _gpu_report_detail(profile, gpu_report)))
    except Exception as error:
        checks.append(("CUDA", False, str(error)))

    try:
        resolved_flow = resolve_flow(flow)
        spec = _bundle_api().load_training_spec(
            flow_default_config(resolved_flow), flow=resolved_flow
        )
        dataset = _bundle_api().fingerprint_dataset(
            profile.dataset_path(spec.dataset_alias),
            spec.extensions,
            recursive=spec.recursive,
            hash_contents=False,
        )
        checks.append(
            ("Dataset scan", True, f"{dataset.file_count:,} images, {dataset.total_bytes:,} bytes")
        )
        free = shutil.disk_usage(profile.runs_root).free
        estimated = spec.estimate_checkpoint_bytes() or 0
        required = max(1 << 30, estimated * 3)
        checks.append(
            (
                "Free disk",
                free >= required,
                f"{_format_bytes(free)} free; {_format_bytes(required)} minimum preflight",
            )
        )
    except Exception as error:
        checks.append(("Dataset/disk preflight", False, str(error)))

    from .hpc_probe import collect_hpc_probe, write_hpc_probe

    hpc = collect_hpc_probe()
    if export_hpc_probe is not None:
        write_hpc_probe(hpc, export_hpc_probe)
        console.print(f"Wrote HPC probe: {export_hpc_probe}")

    try:
        from .backends import get_backend

        backend = get_backend(profile.executor.value)
        if profile.executor in {ExecutorType.WINDOWS_TASK, ExecutorType.SLURM}:
            backend_report = backend.probe(profile, exercise=exercise_executor)
        else:
            backend_report = backend.probe(profile)
        backend_ok = bool(backend_report.get("available", True))
        if exercise_executor and "exercise" in backend_report:
            backend_ok = backend_ok and bool(backend_report["exercise"].get("success"))
        checks.append(
            (
                f"Executor {profile.executor.value}",
                backend_ok,
                json.dumps(backend_report, default=str, sort_keys=True),
            )
        )
    except Exception as error:
        checks.append((f"Executor {profile.executor.value}", False, str(error)))
    table = Table(title=f"Machine doctor: {machine}")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Details", overflow="fold")
    for name, ok, detail in checks:
        table.add_row(name, "[green]PASS[/green]" if ok else "[red]FAIL[/red]", detail)
    console.print(table)
    if not all(ok for _, ok, _ in checks):
        raise typer.Exit(2)


@train_app.command("plan")
def train_plan(
    machine: str = typer.Option(..., "--machine"),
    flow: str = typer.Option(DEFAULT_FLOW, "--flow", help="Training flow to run."),
    config: Optional[Path] = typer.Option(None, "--config", help="Defaults to the flow's active recipe."),
    label: Optional[str] = typer.Option(None, "--label"), dataset: Optional[str] = typer.Option(None, "--dataset"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"), learning_rate: Optional[float] = typer.Option(None, "--learning-rate"),
    checkpoint_every: Optional[int] = typer.Option(None, "--checkpoint-every"), checkpoint_minutes: Optional[int] = typer.Option(None, "--checkpoint-minutes"),
    validation_every: Optional[int] = typer.Option(None, "--validation-every"), sample_every: Optional[int] = typer.Option(None, "--sample-every"),
    seed: Optional[int] = typer.Option(None, "--seed"), reproducibility: Optional[ReproducibilityMode] = typer.Option(None, "--reproducibility"),
    num_workers: Optional[int] = typer.Option(None, "--num-workers"), cache_in_memory: Optional[bool] = typer.Option(None, "--cache-in-memory/--no-cache-in-memory"),
    set_values: Optional[list[str]] = typer.Option(None, "--set", metavar="NAME=VALUE", help="Flow-specific setting; repeatable."),
) -> None:
    """Validate and preview a run without writing or launching anything."""

    _common_plan(machine, config, launch=False, yes=False,
                 flow=flow, set_values=set_values, label=label, dataset=dataset,
                 max_steps=max_steps, batch_size=batch_size,
                 learning_rate=learning_rate, checkpoint_every=checkpoint_every,
                 checkpoint_minutes=checkpoint_minutes, validation_every=validation_every,
                 sample_every=sample_every, seed=seed, reproducibility=reproducibility,
                 num_workers=num_workers, cache_in_memory=cache_in_memory)


@train_app.command("launch")
def train_launch(
    machine: str = typer.Option(..., "--machine"),
    flow: str = typer.Option(DEFAULT_FLOW, "--flow", help="Training flow to run."),
    config: Optional[Path] = typer.Option(None, "--config", help="Defaults to the flow's active recipe."),
    label: Optional[str] = typer.Option(None, "--label"), dataset: Optional[str] = typer.Option(None, "--dataset"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"), learning_rate: Optional[float] = typer.Option(None, "--learning-rate"),
    checkpoint_every: Optional[int] = typer.Option(None, "--checkpoint-every"), checkpoint_minutes: Optional[int] = typer.Option(None, "--checkpoint-minutes"),
    validation_every: Optional[int] = typer.Option(None, "--validation-every"), sample_every: Optional[int] = typer.Option(None, "--sample-every"),
    seed: Optional[int] = typer.Option(None, "--seed"), reproducibility: Optional[ReproducibilityMode] = typer.Option(None, "--reproducibility"),
    num_workers: Optional[int] = typer.Option(None, "--num-workers"), cache_in_memory: Optional[bool] = typer.Option(None, "--cache-in-memory/--no-cache-in-memory"),
    set_values: Optional[list[str]] = typer.Option(None, "--set", metavar="NAME=VALUE", help="Flow-specific setting; repeatable."),
    yes: bool = typer.Option(False, "--yes", help="Confirm a noninteractive launch."),
) -> None:
    """Create an immutable run and launch it through the profile executor."""

    _common_plan(machine, config, launch=True, yes=yes,
                 flow=flow, set_values=set_values, label=label, dataset=dataset,
                 max_steps=max_steps, batch_size=batch_size,
                 learning_rate=learning_rate, checkpoint_every=checkpoint_every,
                 checkpoint_minutes=checkpoint_minutes, validation_every=validation_every,
                 sample_every=sample_every, seed=seed, reproducibility=reproducibility,
                 num_workers=num_workers, cache_in_memory=cache_in_memory)


@train_app.command("wizard")
def train_wizard(
    machine: str = typer.Option(..., "--machine"),
    flow: str = typer.Option(DEFAULT_FLOW, "--flow", help="Training flow to run."),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Guide a user through the high-impact experiment settings."""

    try:
        resolved_flow = resolve_flow(flow)
        active_config = Path(config) if config is not None else flow_default_config(resolved_flow)
        base = _bundle_api().load_training_spec(active_config, flow=resolved_flow)
        profile = _profile_api().load_profile(machine)
    except Exception as error:
        console.print(f"[bold red]Cannot start wizard:[/bold red] {error}")
        raise typer.Exit(2) from error
    console.print(
        Panel(
            "Every experimental choice is validated and rendered into the final command.",
            title=f"{resolved_flow.name} training wizard",
        )
    )
    aliases = sorted(profile.datasets)
    if not aliases:
        console.print("[bold red]Machine profile defines no datasets.[/bold red]")
        raise typer.Exit(2)
    dataset = Prompt.ask(
        "Dataset alias",
        choices=aliases,
        default=base.dataset_alias if base.dataset_alias in aliases else aliases[0],
    )
    values = {
        "label": Prompt.ask("Run label", default=base.label),
        "dataset": dataset,
        "max_steps": IntPrompt.ask("Maximum optimizer steps", default=base.max_steps),
        "batch_size": IntPrompt.ask("Batch size", default=base.batch_size),
        "learning_rate": FloatPrompt.ask("Learning rate", default=base.lr),
        "checkpoint_every": IntPrompt.ask("Checkpoint every N steps", default=base.checkpoint_every),
        "checkpoint_minutes": IntPrompt.ask(
            "Checkpoint at least every N minutes", default=base.checkpoint_minutes
        ),
        "validation_every": IntPrompt.ask("Validate every N steps", default=base.validation_every),
        "sample_every": IntPrompt.ask("Generate samples every N steps", default=base.sample_every),
        "seed": IntPrompt.ask("Seed", default=base.seed),
        "reproducibility": ReproducibilityMode(
            Prompt.ask(
                "Reproducibility mode",
                choices=[x.value for x in ReproducibilityMode],
                default=base.reproducibility.value,
            )
        ),
        "num_workers": IntPrompt.ask("DataLoader workers", default=base.num_workers),
        "cache_in_memory": Confirm.ask("Cache dataset in memory?", default=base.cache_in_memory),
    }
    set_values = _prompt_flow_options(resolved_flow, base)
    try:
        overrides = _spec_overrides(**values)
        overrides.update(_parse_set_overrides(set_values, resolved_flow.spec_model))
        profile, spec, argv = _load_plan(machine, active_config, overrides, resolved_flow)
        _show_plan(profile, spec, argv, resolved_flow)
        _launch_bundle(profile, spec, argv, confirmed=False, flow=resolved_flow)
    except (ValidationError, ValueError, KeyError, FileNotFoundError, RuntimeError, OSError) as error:
        console.print(f"[bold red]Invalid training plan:[/bold red] {error}")
        raise typer.Exit(2) from error


def _prompt_flow_options(flow: Flow, base: BaseTrainingSpec) -> list[str]:
    """Prompt for each flow-specific setting, returning ``--set`` strings.

    Driving the prompts off the flow's declared options is what keeps the
    wizard usable for a pipeline this module has never heard of.
    """

    answers: list[str] = []
    for option in flow.options:
        current = getattr(base, option.name)
        if option.kind == "int":
            value: Any = IntPrompt.ask(option.prompt, default=int(current))
        elif option.kind == "float":
            value = FloatPrompt.ask(option.prompt, default=float(current))
        elif option.kind == "bool":
            value = Confirm.ask(option.prompt, default=bool(current))
        elif option.kind == "int_list":
            raw = Prompt.ask(option.prompt, default=",".join(str(item) for item in current))
            parsed = _parse_int_tuple(raw, f"--set {option.name}")
            value = list(parsed) if parsed else []
        else:
            value = Prompt.ask(option.prompt, default=str(current))
        answers.append(f"{option.name}={_render_set_value(value)}")
    return answers


def _resolve_run(run: Path) -> Path:
    path = run.expanduser().resolve()
    if not (path / "manifest.json").is_file():
        raise typer.BadParameter(f"not a run bundle: {path}")
    return path


@run_app.command("status")
def run_status(run: Path) -> None:
    """Show canonical and backend state for a run."""

    from .tracking import format_run_summary, summarize_run

    path = _resolve_run(run)
    summary = summarize_run(path)
    console.print(format_run_summary(summary))
    if summary.state == "running":
        try:
            state_payload = json.loads((path / "state.json").read_text(encoding="utf-8"))
            heartbeat = datetime.fromisoformat(str(state_payload["heartbeat_at"]))
            if heartbeat.tzinfo is None:
                heartbeat = heartbeat.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - heartbeat.astimezone(timezone.utc)).total_seconds()
            if age > 120:
                console.print(
                    f"[yellow]Heartbeat is stale ({age:.0f}s); check the scheduler before deciding the run is lost.[/yellow]"
                )
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            console.print("[yellow]Running state has no readable heartbeat; check the scheduler.[/yellow]")
    backend_path = path / "backend.json"
    if not backend_path.is_file():
        return
    try:
        backend_data = json.loads(backend_path.read_text(encoding="utf-8"))
        from .backends import get_backend

        backend_name = str(backend_data.get("backend", ""))
        if backend_name == ExecutorType.FOREGROUND.value:
            return
        if backend_name != ExecutorType.EXTERNAL_HPC.value and not backend_data.get("job_id"):
            return
        backend = get_backend(backend_name)
        backend_status = (
            backend.status(path)
            if backend_name == ExecutorType.EXTERNAL_HPC.value
            else backend.status(str(backend_data["job_id"]))
        )
        scheduler_line = f"Scheduler state: {backend_status.state}"
        if backend_status.detail:
            scheduler_line += f" | raw={backend_status.detail}"
        raw_exit = backend_status.metadata.get("slurm_exit_code")
        if raw_exit:
            scheduler_line += f" | exit={raw_exit}"
        elif backend_status.exit_code is not None:
            scheduler_line += f" | exit={backend_status.exit_code}"
        console.print(scheduler_line)
    except Exception as error:
        console.print(f"[yellow]Scheduler status unavailable:[/yellow] {error}")


@run_app.command("logs")
def run_logs(
    run: Path,
    lines: int = typer.Option(80, "--lines", min=1),
    stream: str = typer.Option("both", "--stream", help="stdout, stderr, or both."),
    follow: bool = typer.Option(False, "--follow", help="Follow one selected stream."),
) -> None:
    """Print the newest worker log lines."""

    from .tracking import follow_log, latest_log_paths, tail_logs

    path = _resolve_run(run)
    normalized = stream.casefold()
    if normalized not in {"stdout", "stderr", "both"}:
        raise typer.BadParameter("--stream must be stdout, stderr, or both")
    if follow and normalized == "both":
        raise typer.BadParameter("--follow requires --stream stdout or --stream stderr")
    console.print(tail_logs(path, lines=lines, stream=normalized), markup=False)
    if not follow:
        return
    selected = latest_log_paths(path)[normalized]
    console.print(f"Following {selected}; press Ctrl+C to stop.")
    try:
        for line in follow_log(selected, start_at_end=True):
            console.print(line, markup=False)
    except KeyboardInterrupt:
        return


@run_app.command("stop")
def run_stop(run: Path, force: bool = typer.Option(False, "--force")) -> None:
    """Request checkpointed stop, or invoke backend cancellation with --force."""

    path = _resolve_run(run)
    state_path = path / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        status = str(state.get("status") or state.get("state") or "")
    except (OSError, json.JSONDecodeError):
        status = ""
    if status in {"completed", "failed", "cancelled", "timed_out", "interrupted", "lost"}:
        raise typer.BadParameter(f"run is already terminal ({status})")
    if not force:
        request = path / "stop.request"
        _bundle_api().atomic_write_text(
            request, f"{datetime.now(timezone.utc).isoformat()} user_requested\n"
        )
        console.print(f"Graceful stop requested: {request}")
        return
    manifest = RunManifest.model_validate_json((path / "manifest.json").read_text(encoding="utf-8"), strict=False)
    from .backends import get_backend

    backend = get_backend(manifest.machine.executor.value)
    if manifest.machine.executor is ExecutorType.FOREGROUND:
        raise typer.BadParameter(
            "foreground force-cancel is owned by its terminal; use the graceful stop request or Ctrl+C there"
        )
    if manifest.machine.executor is ExecutorType.EXTERNAL_HPC:
        status = backend.stop(path)
        console.print(
            "The graceful worker stop was requested, but the external scheduler must be "
            f"cancelled through the approved portal if it does not respond. {status.detail}"
        )
        return
    else:
        backend_data_path = path / "backend.json"
        if not backend_data_path.is_file():
            raise typer.BadParameter("run has no recorded backend job to cancel")
        backend_data = json.loads(backend_data_path.read_text(encoding="utf-8"))
        backend.stop(str(backend_data["job_id"]))
    console.print("Force-cancel request sent to the execution backend.")


@run_app.command("resume")
def run_resume(run: Path, yes: bool = typer.Option(False, "--yes")) -> None:
    """Resume the exact manifest from its newest valid checkpoint."""

    path = _resolve_run(run)
    if not yes and (not sys.stdin.isatty() or not Confirm.ask("Resume this exact run?", default=False)):
        raise typer.Abort()
    manifest = RunManifest.model_validate_json((path / "manifest.json").read_text(encoding="utf-8"), strict=False)
    latest = path / "checkpoints" / "latest.json"
    if not latest.is_file():
        raise typer.BadParameter(f"no resumable checkpoint index exists: {latest}")
    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
    if state.get("status") in {"submitted", "queued", "running"}:
        raise typer.BadParameter(f"cannot resume while run state is {state['status']!r}")
    if state.get("status") == "completed":
        raise typer.BadParameter("completed runs cannot be resumed; create a new run for new work")

    try:
        _launch_preflight(manifest.machine)
    except (RuntimeError, FileNotFoundError, OSError) as error:
        raise typer.BadParameter(str(error)) from error

    existing = [int(item.name) for item in (path / "attempts").iterdir() if item.is_dir() and item.name.isdigit()]
    attempt = max(existing, default=0) + 1
    attempt_dir = path / "attempts" / f"{attempt:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    (attempt_dir / "stdout.log").touch(exist_ok=False)
    (attempt_dir / "stderr.log").touch(exist_ok=False)
    bundles = _bundle_api()
    source_dir = bundles.materialized_source_path(path)
    worker = (
        manifest.machine.python_executable,
        str(bundles.worker_bootstrap_path(path)),
        "--run",
        str(path),
        "--attempt",
        str(attempt),
        "--resume",
    )
    from .backends import get_backend

    backend = get_backend(manifest.machine.executor.value)
    request = _make_launch_request(manifest.machine, path, source_dir, attempt_dir, worker)
    _record_worker_command(path, attempt, request)
    try:
        if manifest.machine.executor is ExecutorType.SLURM:
            submission = backend.submit(request, profile=manifest.machine)
        else:
            submission = backend.submit(request)
    except Exception as error:
        _record_submission_failure(
            path, attempt, manifest.machine.executor.value, error
        )
        raise typer.BadParameter(f"resume submission failed: {error}") from error
    _record_submission(path, attempt, submission)
    console.print(_submission_message(submission))


@track_app.command("publish")
def track_publish(
    run: Path,
    tracking_uri: Optional[str] = typer.Option(None, "--tracking-uri"),
    experiment: str = typer.Option("ddim-sem", "--experiment"),
) -> None:
    """Idempotently publish a portable bundle to MLflow."""

    from .tracking import TrackingError, publish_run

    path = _resolve_run(run)
    if tracking_uri is None:
        manifest = RunManifest.model_validate_json(
            (path / "manifest.json").read_text(encoding="utf-8"), strict=False
        )
        tracking_uri = manifest.machine.mlflow_tracking_uri
    try:
        result = publish_run(
            path, tracking_uri=tracking_uri, experiment_name=experiment
        )
    except (TrackingError, OSError, ValueError) as error:
        console.print(f"[bold red]Publication failed:[/bold red] {error}")
        raise typer.Exit(2) from error
    console.print_json(data=result.to_dict())


@track_app.command("serve")
def track_serve(
    data_dir: Path = typer.Option(Path.home() / ".runctl" / "mlflow", "--data-dir"),
    port: int = typer.Option(5000, "--port", min=1, max=65535),
) -> None:
    """Start a loopback-only local MLflow UI."""

    from .tracking import TrackingError, serve_mlflow

    try:
        process = serve_mlflow(data_dir, port=port)
    except (TrackingError, OSError, ValueError) as error:
        console.print(f"[bold red]MLflow server could not start:[/bold red] {error}")
        raise typer.Exit(2) from error
    console.print(f"MLflow is starting at [bold]{process.url}[/bold] (PID {process.pid}).")
    console.print(f"Logs: {process.stdout_path} | {process.stderr_path}")


@environment_app.command("bundle")
def environment_bundle(
    output: Path = typer.Option(..., "--output", help="New wheelhouse directory; never overwritten."),
    with_tracking: bool = typer.Option(
        False, "--with-tracking", help="Include optional MLflow publication dependencies."
    ),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    """Build a checksummed wheelhouse on a connected target-compatible machine."""

    if not yes:
        if not sys.stdin.isatty():
            raise typer.BadParameter("noninteractive wheelhouse build requires --yes")
        if not Confirm.ask(
            "Download/build all target-native dependency wheels now?", default=False
        ):
            raise typer.Abort()
    from .offline import OfflineBundleError, build_wheelhouse, verify_wheelhouse

    try:
        console.print(
            "Build this on the same OS, architecture, Python minor version, and CUDA/PyTorch "
            "target as the offline machine."
        )
        with console.status("Resolving and hashing the target-native wheel set..."):
            path = build_wheelhouse(
                REPOSITORY_ROOT, output, include_tracking=with_tracking
            )
        report = verify_wheelhouse(path)
    except (OfflineBundleError, FileNotFoundError, FileExistsError, OSError) as error:
        console.print(f"[bold red]Wheelhouse build failed:[/bold red] {error}")
        raise typer.Exit(2) from error
    console.print_json(data=report)


@environment_app.command("verify")
def environment_verify(bundle: Path) -> None:
    """Verify every wheel before installation on an offline target."""

    from .offline import OfflineBundleError, verify_wheelhouse

    try:
        report = verify_wheelhouse(bundle)
    except (OfflineBundleError, FileNotFoundError, OSError) as error:
        console.print(f"[bold red]Wheelhouse verification failed:[/bold red] {error}")
        raise typer.Exit(2) from error
    console.print_json(data=report)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
