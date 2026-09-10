"""Small torchrun adapter; single-process execution has no collectives."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import TYPE_CHECKING, Callable, TypeVar

import torch
import torch.distributed as dist
from runctl.control import select_device

if TYPE_CHECKING:
    from .data import PairFactory
    from .fusion import FusionFactory

T = TypeVar("T")


class DistributedRuntime:
    def __init__(self, requested: str):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world_size < 1 or not 0 <= self.rank < self.world_size or self.local_rank < 0:
            raise ValueError("invalid torchrun WORLD_SIZE/RANK/LOCAL_RANK")
        self.enabled = self.world_size > 1
        self.primary = self.rank == 0
        cuda = requested != "cpu" and torch.cuda.is_available()
        if requested == "cuda" and not cuda:
            raise RuntimeError("training.device is 'cuda' but CUDA is not available")
        if self.enabled and requested == "auto" and not cuda:
            raise RuntimeError("torchrun requires CUDA; use explicit device: cpu for CPU diagnostics")
        if self.enabled:
            self.device = torch.device("cuda", self.local_rank) if cuda else torch.device("cpu")
        else:
            self.device = select_device("cuda" if cuda else "cpu")
        # runctl's single-worker selector requires one isolated visible GPU.
        # DDP workers instead see the allocation and bind their LOCAL_RANK.
        if self.enabled and self.device.type == "cuda":
            if self.device.index >= torch.cuda.device_count():
                raise ValueError("LOCAL_RANK exceeds the allocated visible GPUs")
            torch.cuda.set_device(self.device)
        self.owns_group = False
        if self.enabled:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl" if cuda else "gloo", init_method="env://", timeout=timedelta(minutes=5))
                self.owns_group = True
            if dist.get_world_size() != self.world_size or dist.get_rank() != self.rank:
                raise ValueError("existing process group differs from torchrun environment")

    def on_primary(self, action: Callable[[], T]) -> T | None:
        """Run one writer, broadcasting failures before peers continue."""
        if not self.enabled:
            return action()
        result, error = None, [None]
        if self.primary:
            try:
                result = action()
            except Exception as failure:
                error[0] = f"{type(failure).__name__}: {failure}"
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(f"rank 0 operation failed: {error[0]}")
        return result

    def should_stop(self, requested: bool) -> bool:
        if not self.enabled:
            return requested
        flag = torch.tensor(int(requested), device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def mean_values(self, values: dict[str, float]) -> dict[str, float]:
        if not self.enabled:
            return values
        names = sorted(values)
        tensor = torch.tensor([values[name] for name in names], dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor)
        return dict(zip(names, (tensor / self.world_size).cpu().tolist()))

    def gather_state(self, factory: PairFactory | FusionFactory, effective_batch: int) -> dict | None:
        if not self.enabled:
            return None
        state = {"factory": factory.state_dict(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state(self.device).cpu() if self.device.type == "cuda" else None}
        states = [None] * self.world_size if self.primary else None
        dist.gather_object(state, states, dst=0)
        return {"world_size": self.world_size, "effective_batch": effective_batch, "ranks": states}

    def close(self) -> None:
        if self.owns_group:
            dist.destroy_process_group()
            self.owns_group = False
