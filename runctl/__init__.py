"""Reliable, portable, flow-agnostic experiment orchestration.

``runctl`` owns everything that is true of *any* training run: immutable run
bundles, dataset fingerprinting, source snapshotting, executors, checkpoint and
metric logging, machine profiles, and optional tracking publication.  The parts
that are specific to one research pipeline live in a *flow* plugin -- see
:mod:`runctl.registry`.
"""

from .registry import Flow, FlowError, SpecOption, available_flows, get_flow, register_flow
from .schemas import (
    AttemptState,
    BaseTrainingSpec,
    DatasetFingerprint,
    ExecutorType,
    MachineProfile,
    ReproducibilityMode,
    RunManifest,
    RunStatus,
    SourceSnapshot,
)

__all__ = [
    "AttemptState",
    "BaseTrainingSpec",
    "DatasetFingerprint",
    "ExecutorType",
    "Flow",
    "FlowError",
    "MachineProfile",
    "ReproducibilityMode",
    "RunManifest",
    "RunStatus",
    "SourceSnapshot",
    "SpecOption",
    "available_flows",
    "get_flow",
    "register_flow",
]

__version__ = "0.1.0"
