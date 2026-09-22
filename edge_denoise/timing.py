"""Bounded cumulative training stage timings; optional CUDA synchronization."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch


class TrainingTimings:
    def __init__(self, device: torch.device, *, synchronize: bool = False) -> None:
        self.device = device
        self.synchronize = synchronize and device.type == "cuda"
        self.started = time.perf_counter()
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def add(self, name: str, started: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started
        self.calls[name] = self.calls.get(name, 0) + 1

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if self.synchronize:
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            if self.synchronize:
                torch.cuda.synchronize(self.device)
            self.add(name, started)

    def record(self) -> dict:
        return {
            "wall_seconds": time.perf_counter() - self.started,
            "stage_seconds": dict(self.seconds), "stage_calls": dict(self.calls),
            "cuda_synchronized": self.synchronize,
            "note": "Nested stages overlap. Without --profile CUDA stage times include asynchronous dispatch.",
            "device": str(self.device),
            "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated(self.device)
                                          if self.device.type == "cuda" else None),
        }
