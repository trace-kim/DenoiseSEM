"""Flow plugin registry.

``runctl`` owns reproducible *orchestration* — run bundles, executors, machine
profiles, logging, checkpoints and tracking.  It deliberately knows nothing
about any particular model or training objective.  A *flow* supplies the parts
that are specific to one research pipeline:

* a :class:`~runctl.schemas.BaseTrainingSpec` subclass describing its settings,
* a parser that turns a YAML recipe into that spec's field names,
* a trainer entry point invoked with an immutable ``RunManifest``,
* optional metadata used to render plans and the interactive wizard.

Flows are discovered through the ``runctl.flows`` entry-point group so that a
new pipeline can register itself without editing this package.  ``ddim`` ships
in-tree and is also listed in :data:`BUILTIN_FLOWS` as a fallback for source
checkouts that were never ``pip install``-ed.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .schemas import BaseTrainingSpec


ENTRY_POINT_GROUP = "runctl.flows"

#: Import paths tried when an entry point is unavailable (e.g. a plain source
#: checkout).  Each module must expose ``FLOW``.
BUILTIN_FLOWS: dict[str, str] = {
    "ddim": "ddim.flow",
}


class FlowError(RuntimeError):
    """Raised when a flow cannot be resolved or is invalid."""


@dataclass(frozen=True)
class SpecOption:
    """One flow-specific setting exposed to the wizard and canonical command.

    ``name`` must be a field on the flow's spec model.  ``prompt`` is shown by
    ``runctl train wizard``; ``kind`` selects the prompt widget and the way the
    value is rendered into ``--set`` on the canonical command line.
    """

    name: str
    prompt: str
    kind: str = "str"  # one of: str, int, float, bool, int_list

    def __post_init__(self) -> None:
        if self.kind not in {"str", "int", "float", "bool", "int_list"}:
            raise ValueError(f"unsupported SpecOption kind: {self.kind!r}")


@dataclass(frozen=True)
class Flow:
    """Everything ``runctl`` needs in order to plan and execute one pipeline."""

    name: str
    spec_model: type[BaseTrainingSpec]
    parse_config: Callable[[Path | str], dict[str, Any]]
    trainer: Callable[..., Any]
    default_config: Path | None = None
    description: str = ""
    options: tuple[SpecOption, ...] = ()
    plan_rows: Callable[[BaseTrainingSpec], Sequence[tuple[str, str]]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec_model, type) or not issubclass(
            self.spec_model, BaseTrainingSpec
        ):
            raise FlowError(
                f"flow {self.name!r} spec_model must subclass BaseTrainingSpec"
            )
        unknown = [
            option.name
            for option in self.options
            if option.name not in self.spec_model.model_fields
        ]
        if unknown:
            raise FlowError(
                "flow {!r} declares option(s) that are not spec fields: {}".format(
                    self.name, ", ".join(sorted(unknown))
                )
            )

    def option_names(self) -> tuple[str, ...]:
        return tuple(option.name for option in self.options)

    def rows_for(self, spec: BaseTrainingSpec) -> tuple[tuple[str, str], ...]:
        if self.plan_rows is None:
            return ()
        return tuple((str(key), str(value)) for key, value in self.plan_rows(spec))


_REGISTRY: dict[str, Flow] = {}


def register_flow(flow: Flow, *, replace: bool = False) -> Flow:
    """Add ``flow`` to the process-wide registry."""

    existing = _REGISTRY.get(flow.name)
    if existing is not None and not replace and existing is not flow:
        raise FlowError(f"flow {flow.name!r} is already registered")
    _REGISTRY[flow.name] = flow
    return flow


def _load_entry_point_flow(name: str) -> Flow | None:
    try:
        entry_points = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - very old importlib.metadata
        entry_points = importlib.metadata.entry_points().get(ENTRY_POINT_GROUP, [])
    for entry_point in entry_points:
        if entry_point.name != name:
            continue
        loaded = entry_point.load()
        if isinstance(loaded, Flow):
            return loaded
        flow = getattr(loaded, "FLOW", None)
        if isinstance(flow, Flow):
            return flow
        raise FlowError(
            f"entry point {name!r} did not resolve to a Flow or a module exposing FLOW"
        )
    return None


def _load_builtin_flow(name: str) -> Flow | None:
    module_path = BUILTIN_FLOWS.get(name)
    if module_path is None:
        return None
    try:
        module = importlib.import_module(module_path)
    except ImportError as error:
        raise FlowError(
            f"flow {name!r} is known but its package could not be imported: {error}"
        ) from error
    flow = getattr(module, "FLOW", None)
    if not isinstance(flow, Flow):
        raise FlowError(f"module {module_path!r} does not expose a Flow named FLOW")
    return flow


def get_flow(name: str) -> Flow:
    """Resolve a flow by name, importing its package on first use."""

    if name in _REGISTRY:
        return _REGISTRY[name]
    for loader in (_load_entry_point_flow, _load_builtin_flow):
        flow = loader(name)
        if flow is not None:
            return register_flow(flow, replace=True)
    known = ", ".join(available_flows()) or "none"
    raise FlowError(f"unknown flow {name!r}; available flows: {known}")


def available_flows() -> tuple[str, ...]:
    """Names of every flow that is registered or discoverable."""

    names = set(_REGISTRY) | set(BUILTIN_FLOWS)
    try:
        entry_points = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover
        entry_points = importlib.metadata.entry_points().get(ENTRY_POINT_GROUP, [])
    names.update(entry_point.name for entry_point in entry_points)
    return tuple(sorted(names))


def spec_model_for(name: str) -> type[BaseTrainingSpec]:
    return get_flow(name).spec_model


def reset_registry(flows: Mapping[str, Flow] | None = None) -> None:
    """Replace the registry contents.  Intended for tests."""

    _REGISTRY.clear()
    if flows:
        _REGISTRY.update(flows)
