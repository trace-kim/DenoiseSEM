"""Flow-agnostic training primitives shared by every ``runctl`` flow.

These pieces are about *running a training loop reproducibly*, not about any
particular model: seeding and determinism policy, single-GPU device selection,
epoch-deterministic data ordering, resumable sampling, and cooperative stop
requests.  A flow's trainer composes them; it does not reimplement them.
"""

from __future__ import annotations

import hashlib
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def configure_reproducibility(seed: int, mode: str) -> None:
    mode = str(mode).lower().replace("_", "-")
    aliases = {"seeded": "repeatable", "seeded-repeatable": "repeatable", "deterministic": "strict"}
    mode = aliases.get(mode, mode)
    if mode not in {"repeatable", "strict", "performance"}:
        raise ValueError("reproducibility must be repeatable, strict, or performance")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    if mode == "strict":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    elif mode == "repeatable":
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def select_device(requested: str | torch.device | None = None) -> torch.device:
    if requested is None and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; refusing to start an accidentally CPU-bound training run. "
            "Use an explicit --device cpu only for a deliberate smoke/test invocation."
        )
    device = torch.device(requested) if requested is not None else torch.device("cuda")
    if device.type == "cuda":
        count = torch.cuda.device_count()
        if count != 1:
            raise RuntimeError(
                "The training worker expected one isolated CUDA GPU but found {}. "
                "Launch through runctl or restrict CUDA_VISIBLE_DEVICES before starting "
                "the worker directly.".format(count)
            )
        if device.index is None:
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    return device

class EpochSeededDataset(Dataset[Any]):
    """Make random transforms a pure function of seed, epoch, and sample index."""

    def __init__(self, dataset: Dataset[Any], seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def _sample_seed(self, index: int) -> int:
        digest = hashlib.blake2b(
            "{}:{}:{}".format(self.seed, self.epoch, int(index)).encode("ascii"), digest_size=8
        ).digest()
        return int.from_bytes(digest, "little") % (2**32)

    def __getitem__(self, index: int) -> Any:
        sample_seed = self._sample_seed(index)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        with torch.random.fork_rng(devices=[]):
            try:
                random.seed(sample_seed)
                np.random.seed(sample_seed)
                torch.manual_seed(sample_seed)
                return self.dataset[index]
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)

class ResumableEpochSampler(Sampler[int]):
    def __init__(self, size: int, *, seed: int, epoch: int, start_index: int = 0):
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.start_index = max(0, min(int(start_index), self.size))

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed((self.seed + self.epoch * 1_000_003) % (2**63 - 1))
        permutation = torch.randperm(self.size, generator=generator).tolist()
        return iter(permutation[self.start_index :])

    def __len__(self) -> int:
        return self.size - self.start_index

    def state_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "start_index": self.start_index,
            "dataset_size": self.size,
        }

class StopController:
    def __init__(self, stop_file: os.PathLike[str] | str):
        import threading

        self.stop_file = Path(stop_file)
        self._requested = threading.Event()
        self.reason = "stop requested"

    def request(self, reason: str = "stop requested") -> None:
        self.reason = reason
        self._requested.set()

    def is_requested(self) -> bool:
        return self._requested.is_set() or self.stop_file.exists()

@dataclass(frozen=True)
class TrainingResult:
    status: str
    global_step: int
    epoch: int
    batch_in_epoch: int
    checkpoint: str | None
    best_validation_loss: float | None
