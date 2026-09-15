"""Locating and fetching model weights, explicitly.

``facebook/sam3`` is a gated repository: it needs an accepted licence agreement
and a token, and the machine that eventually runs this pipeline may have no
internet access at all.  Weights also cannot travel in the repository, because
``.gitignore`` excludes ``*.safetensors`` globally.

So nothing here downloads implicitly.  A run either finds weights already
present or fails with a message naming the exact remedy - which beats a batch
job that pulls a gigabyte halfway through, or one that dies on an opaque HTTP
401 after an hour of other work.

Resolution order is ``model_path`` (a local directory), then the Hugging Face
cache, then the hub.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Where a user accepts the licence for the default checkpoint.
LICENCE_URL = "https://huggingface.co/facebook/sam3"

INSTALL_HINT = 'python -m pip install -e ".[segment]"'


class MissingDependency(RuntimeError):
    """The optional segmentation stack is not installed."""


class GatedRepositoryError(RuntimeError):
    """Weights exist but this machine is not allowed, or able, to fetch them."""


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "0") not in ("0", "", "false", "False")


def transformers_available() -> tuple[bool, str]:
    """Whether the SAM 3 classes can be imported, and a message if not."""
    try:
        import transformers
    except ImportError as error:
        return False, f"transformers is not installed ({error}). Install with: {INSTALL_HINT}"
    missing = [n for n in ("Sam3Model", "Sam3Processor") if not hasattr(transformers, n)]
    if missing:
        return False, (
            f"transformers {transformers.__version__} does not provide {', '.join(missing)}; "
            f"SAM 3 needs transformers>=5. Upgrade with: {INSTALL_HINT}"
        )
    return True, f"transformers {transformers.__version__}"


def cached_snapshot(model_id: str) -> Path | None:
    """The local cache directory for ``model_id``, when it is already present."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:
        return Path(snapshot_download(model_id, local_files_only=True))
    except Exception:
        return None


def resolve_model_source(model_id: str, model_path: Path | None = None) -> str:
    """Return what ``from_pretrained`` should be given, or explain what is missing."""
    if model_path is not None:
        path = Path(model_path)
        if not path.is_dir():
            raise GatedRepositoryError(
                f"segmentation.model_path does not exist: {path}. It must be a directory holding "
                "the model files (config.json, the safetensors weights, and the processor files)."
            )
        return str(path)

    cached = cached_snapshot(model_id)
    if cached is not None:
        return model_id

    if _offline():
        raise GatedRepositoryError(
            f"HF_HUB_OFFLINE is set and {model_id} is not in the local cache.\n"
            "On a machine without internet access:\n"
            f"  1. On a connected machine, accept the licence at {LICENCE_URL}, then run\n"
            f"     python -m sem_segment download-weights --model-id {model_id}\n"
            "  2. Copy the printed cache directory across, e.g.\n"
            "     scp -r ~/.cache/huggingface/hub/models--facebook--sam3 <host>:~/.cache/huggingface/hub/\n"
            "  3. Set HF_HOME to that cache root here.\n"
            "Or set segmentation.backend to 'classical', which needs no weights at all."
        )
    if _token() is None:
        raise GatedRepositoryError(
            f"{model_id} is a gated repository and no token is configured.\n"
            f"  1. Accept the licence at {LICENCE_URL} with your Hugging Face account.\n"
            "  2. Export a token: HF_TOKEN=hf_...\n"
            f"  3. Fetch once: python -m sem_segment download-weights --model-id {model_id}\n"
            "Or set segmentation.backend to 'classical', which needs no weights at all."
        )
    return model_id


def fetch_weights(
    model_id: str, *, dest: Path | None = None, revision: str | None = None
) -> Path:
    """Download a snapshot explicitly. Never called implicitly by a run."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise MissingDependency(
            f"huggingface_hub is not installed ({error}). Install with: {INSTALL_HINT}"
        ) from error

    kwargs: dict = {"repo_id": model_id, "revision": revision}
    if dest is not None:
        kwargs["local_dir"] = str(Path(dest))
    token = _token()
    if token:
        kwargs["token"] = token
    try:
        return Path(snapshot_download(**{k: v for k, v in kwargs.items() if v is not None}))
    except Exception as error:
        message = str(error)
        if "401" in message or "403" in message or "gated" in message.lower():
            raise GatedRepositoryError(
                f"Access to {model_id} was refused ({type(error).__name__}).\n"
                f"  1. Accept the licence at {LICENCE_URL} while signed in.\n"
                "  2. Export a token with read access: HF_TOKEN=hf_...\n"
                f"Original error: {message}"
            ) from error
        raise GatedRepositoryError(f"could not download {model_id}: {message}") from error


def backend_readiness(name: str) -> tuple[bool, str]:
    """Whether a named backend can run here, for the ``backends`` command."""
    if name == "classical":
        try:
            import skimage  # noqa: F401
            import scipy  # noqa: F401
        except ImportError as error:
            return False, f'needs the analysis extra ({error}); pip install -e ".[analysis]"'
        return True, "no weights required"

    ok, detail = transformers_available()
    if not ok:
        return False, detail
    cached = cached_snapshot("facebook/sam3")
    if cached is None:
        if _token() is None:
            return False, f"{detail}; weights not cached and HF_TOKEN is not set"
        return False, f"{detail}; weights not cached - run: python -m sem_segment download-weights"
    return True, f"{detail}; weights cached at {cached}"
