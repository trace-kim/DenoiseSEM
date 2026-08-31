from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from runctl.bundles import DuplicateOptionError, reject_duplicate_scalar_options
from runctl.schemas import AttemptState, BaseTrainingSpec, RunStatus


def test_repeated_scalar_flags_are_rejected_before_cli_parsing() -> None:
    argv = ["train", "--max-steps=10", "--label", "test", "--max-steps", "20"]
    with pytest.raises(DuplicateOptionError, match="--max-steps"):
        reject_duplicate_scalar_options(argv, {"--max-steps", "--label"})


def test_attempt_state_enforces_terminal_timestamps() -> None:
    now = datetime.now().astimezone()
    with pytest.raises(ValidationError, match="require ended_at"):
        AttemptState(updated_at=now, status=RunStatus.FAILED)
    completed = AttemptState(
        updated_at=now,
        started_at=now,
        ended_at=now,
        status=RunStatus.COMPLETED,
        exit_code=0,
    )
    assert completed.status is RunStatus.COMPLETED


def test_base_spec_is_strict_and_flow_agnostic() -> None:
    """The shared spec forbids unknown fields and never coerces types."""

    with pytest.raises(ValidationError):
        BaseTrainingSpec.model_validate({"max_steps": "20000"})
    with pytest.raises(ValidationError):
        BaseTrainingSpec.model_validate({"not_a_setting": 1})
    # Model geometry belongs to a flow, not to the shared contract.
    with pytest.raises(ValidationError):
        BaseTrainingSpec.model_validate({"ch_mult": (1, 2)})


def test_base_spec_rejects_intervals_beyond_max_steps() -> None:
    with pytest.raises(ValidationError, match="checkpoint_every cannot exceed"):
        BaseTrainingSpec(
            max_steps=10, checkpoint_every=11, validation_every=10, sample_every=10
        )


def test_base_spec_validates_dataset_extensions() -> None:
    with pytest.raises(ValidationError, match="invalid dataset extension"):
        BaseTrainingSpec(extensions=("png",))
    with pytest.raises(ValidationError, match="duplicate dataset extension"):
        BaseTrainingSpec(extensions=(".png", ".PNG"))
    assert BaseTrainingSpec(extensions=(".PNG",)).extensions == (".png",)


def test_base_spec_has_no_checkpoint_estimate_by_default() -> None:
    assert BaseTrainingSpec().estimate_checkpoint_bytes() is None
