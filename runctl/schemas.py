"""Strict, versioned contracts shared by the launcher and training worker.

The models in this module deliberately do not accept unknown fields or implicit
Python-side coercion.  Configuration-file loaders are responsible for parsing
human-facing syntax before constructing these contracts.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = 1
_SLUG_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$"


class ExecutorType(str, Enum):
    FOREGROUND = "foreground"
    WINDOWS_TASK = "windows-task"
    SLURM = "slurm"
    EXTERNAL_HPC = "external-hpc"


class ReproducibilityMode(str, Enum):
    SEEDED = "seeded"
    DETERMINISTIC = "deterministic"
    PERFORMANCE = "performance"


class RunStatus(str, Enum):
    PREPARED = "prepared"
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"
    LOST = "lost"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SlurmResources(FrozenStrictModel):
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    time_limit: str = Field(default="24:00:00", pattern=r"^(?:\d+-)?\d{1,2}:\d{2}:\d{2}$")
    cpus_per_task: int = Field(default=4, ge=1)
    memory_gb: int | None = Field(default=None, ge=1)
    gpus: Literal[1] = 1
    extra_sbatch: tuple[str, ...] = ()

    @field_validator("extra_sbatch")
    @classmethod
    def validate_extra_sbatch(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if not item.startswith("--") or any(c in item for c in "\r\n\0"):
                raise ValueError("extra_sbatch entries must be single safe --options")
        return value


class MachineProfile(FrozenStrictModel):
    """Operational machine settings; training hyperparameters do not belong here."""

    schema_version: Literal[1] = SCHEMA_VERSION
    machine_id: str = Field(pattern=_SLUG_PATTERN)
    executor: ExecutorType
    runs_root: Path
    datasets: dict[str, Path]
    timezone: str = "Asia/Seoul"
    python_executable: str = "python"
    gpu_index: int = Field(default=0, ge=0)
    expected_gpu: str | None = None
    # Reserved for a site-approved Apptainer adapter; v1 wheelhouse execution
    # leaves this unset and rejects container policy assumptions in the CLI.
    container_image: Path | None = None
    mlflow_tracking_uri: str | None = None
    slurm: SlurmResources | None = None

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, value: dict[str, Path]) -> dict[str, Path]:
        if not value:
            raise ValueError("at least one dataset alias is required")
        for alias, path in value.items():
            if not alias or not __import__("re").fullmatch(_SLUG_PATTERN, alias):
                raise ValueError(f"invalid dataset alias: {alias!r}")
            if not path:
                raise ValueError(f"dataset path for {alias!r} is empty")
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value

    @field_validator("python_executable")
    @classmethod
    def validate_python_executable(cls, value: str) -> str:
        if not value.strip() or any(c in value for c in "\r\n\0"):
            raise ValueError("python_executable must be a nonempty single-line value")
        return value

    @field_validator("expected_gpu")
    @classmethod
    def validate_expected_gpu(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or any(c in normalized for c in "\r\n\0"):
            raise ValueError("expected_gpu must be a nonempty single-line substring")
        return normalized

    @field_validator("mlflow_tracking_uri")
    @classmethod
    def validate_tracking_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or any(c in normalized for c in "\r\n\0"):
            raise ValueError("mlflow_tracking_uri must be a nonempty single-line URI")
        parsed = urlsplit(normalized)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("mlflow_tracking_uri must not embed credentials")
        return normalized

    @model_validator(mode="after")
    def validate_executor_settings(self) -> "MachineProfile":
        if self.container_image is not None:
            raise ValueError(
                "container_image is reserved until the site's Apptainer contract is configured; "
                "use a verified target-native wheelhouse in v1"
            )
        if self.executor is ExecutorType.SLURM and self.slurm is None:
            raise ValueError("slurm resources are required for the slurm executor")
        if self.executor is not ExecutorType.SLURM and self.slurm is not None:
            raise ValueError("slurm resources are only valid for the slurm executor")
        return self

    def dataset_path(self, alias: str) -> Path:
        try:
            return self.datasets[alias]
        except KeyError as exc:
            available = ", ".join(sorted(self.datasets))
            raise KeyError(f"unknown dataset alias {alias!r}; available: {available}") from exc


class BaseTrainingSpec(FrozenStrictModel):
    """Flow-agnostic training settings understood by every ``runctl`` command.

    A flow subclasses this with the model/objective fields its trainer needs.
    Only the settings below are relied upon by the orchestration layer: the run
    label, the dataset alias resolved through the machine profile, the dataset
    scan rules used for fingerprinting, and the loop/interval bookkeeping used
    by checkpointing and status reporting.

    Values are intentionally flat so the canonical launch command can restate
    every experiment condition without ambiguous dotted-key overrides.
    """

    schema_version: Literal[1] = SCHEMA_VERSION
    label: str = Field(default="run", pattern=_SLUG_PATTERN)
    dataset_alias: str = Field(default="default", pattern=_SLUG_PATTERN)

    num_workers: int = Field(default=0, ge=0)
    cache_in_memory: bool = False
    recursive: bool = False
    validation_split: float = Field(default=0.1, gt=0.0, lt=1.0)
    split_seed: int = Field(default=2019, ge=0, le=2**63 - 1)
    extensions: tuple[str, ...] = (".png", ".tif", ".tiff", ".jpg", ".jpeg")

    batch_size: int = Field(default=8, ge=1)
    max_steps: int = Field(default=20_000, ge=1)
    checkpoint_every: int = Field(default=2_500, ge=1)
    validation_every: int = Field(default=2_500, ge=1)
    sample_every: int = Field(default=2_500, ge=1)
    checkpoint_minutes: int = Field(default=30, ge=1)

    lr: float = Field(default=0.0002, gt=0.0)

    seed: int = Field(default=1234, ge=0, le=2**63 - 1)
    reproducibility: ReproducibilityMode = ReproducibilityMode.SEEDED

    @field_validator("extensions")
    @classmethod
    def validate_extensions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one dataset extension is required")
        normalized: list[str] = []
        for extension in value:
            item = extension.lower()
            if not item.startswith(".") or len(item) < 2 or any(c in item for c in "/\\\0"):
                raise ValueError(f"invalid dataset extension: {extension!r}")
            if item in normalized:
                raise ValueError(f"duplicate dataset extension: {item}")
            normalized.append(item)
        return tuple(normalized)

    @model_validator(mode="after")
    def validate_interval_relationships(self) -> "BaseTrainingSpec":
        for field_name in ("checkpoint_every", "validation_every", "sample_every"):
            if getattr(self, field_name) > self.max_steps:
                raise ValueError(f"{field_name} cannot exceed max_steps")
        return self

    @property
    def config_sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def estimate_checkpoint_bytes(self) -> int | None:
        """Rough on-disk checkpoint size for launch previews.

        Flows that can estimate their parameter count should override this;
        returning ``None`` simply omits the row from the plan table.
        """

        return None


class DatasetFingerprint(FrozenStrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    root: Path
    file_count: int = Field(ge=1)
    total_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    method: Literal["sha256-metadata-v1", "sha256-content-v1"]


class SourceSnapshot(FrozenStrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    archive: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    git_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool

    @field_validator("archive")
    @classmethod
    def validate_archive_name(cls, value: str) -> str:
        if not value or Path(value).name != value or any(c in value for c in "\r\n\0"):
            raise ValueError("source archive must be a safe filename within the run bundle")
        return value


class OutputLayout(FrozenStrictModel):
    state: str = "state.json"
    metrics: str = "metrics.jsonl"
    tensorboard: str = "tensorboard"
    samples: str = "samples"
    checkpoints: str = "checkpoints"
    attempts: str = "attempts"


class RunManifest(FrozenStrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    flow: str = Field(default="ddim", pattern=_SLUG_PATTERN)
    run_id: str = Field(pattern=r"^[0-9]{8}T[0-9]{6}[+-][0-9]{4}__[A-Za-z0-9][A-Za-z0-9._-]{0,79}__[0-9a-f]{8,16}$")
    created_at: datetime
    canonical_argv: tuple[str, ...]
    training: BaseTrainingSpec
    machine: MachineProfile
    dataset: DatasetFingerprint
    source: SourceSnapshot
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_layout: OutputLayout = OutputLayout()
    parent_run_id: str | None = None

    @field_validator("training", mode="before")
    @classmethod
    def coerce_training_spec(cls, value: Any, info: Any) -> Any:
        """Re-type the nested training payload using the flow's spec model.

        A persisted manifest is plain JSON, so ``training`` arrives as a dict.
        Validating it against the declared base model would silently discard
        every flow-specific field, so the concrete model is looked up from the
        already-validated ``flow`` field (declared above ``training``).
        """

        if isinstance(value, BaseTrainingSpec) or not isinstance(value, dict):
            return value
        from .registry import FlowError, get_flow

        flow_name = (info.data or {}).get("flow", "ddim")
        try:
            spec_model = get_flow(str(flow_name)).spec_model
        except FlowError:
            return value
        return spec_model.model_validate(value, strict=False)

    @model_validator(mode="after")
    def validate_manifest_consistency(self) -> "RunManifest":
        if self.config_sha256 != self.training.config_sha256:
            raise ValueError("config_sha256 does not match the embedded training spec")
        if self.training.dataset_alias not in self.machine.datasets:
            raise ValueError("training dataset_alias is not defined by the machine profile")
        if not self.canonical_argv:
            raise ValueError("canonical_argv cannot be empty")
        if any(any(c in item for c in "\r\n\0") for item in self.canonical_argv):
            raise ValueError("canonical_argv entries must not contain control characters")
        return self


class AttemptState(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    attempt: int = Field(default=1, ge=1)
    status: RunStatus = RunStatus.PREPARED
    updated_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    heartbeat_at: datetime | None = None
    pid: int | None = Field(default=None, ge=1)
    backend_job_id: str | None = None
    exit_code: int | None = None
    message: str | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "AttemptState":
        terminal = {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.TIMED_OUT,
            RunStatus.INTERRUPTED,
            RunStatus.LOST,
        }
        if self.ended_at is not None and self.status not in terminal:
            raise ValueError("ended_at is only valid for terminal states")
        if self.status in terminal and self.ended_at is None:
            raise ValueError("terminal states require ended_at")
        if self.status is RunStatus.COMPLETED and self.exit_code not in (None, 0):
            raise ValueError("completed attempts cannot have a nonzero exit code")
        if self.started_at and self.ended_at and self.ended_at < self.started_at:
            raise ValueError("ended_at cannot precede started_at")
        if self.started_at and self.updated_at < self.started_at:
            raise ValueError("updated_at cannot precede started_at")
        return self
