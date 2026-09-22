"""Wall-clock progress for long comparison operations, including blocking I/O."""

from __future__ import annotations

import threading
import time


class Progress:
    """Log before work starts, periodically while it runs, and on exit.

    Heartbeats describe the active operation; they do not imply that a blocked
    operation is advancing. Timings accumulate when the same key is used again.
    Nested timings overlap and must not be summed as independent wall time.
    """

    def __init__(self, label: str, *, total: int | None = None,
                 timings: dict[str, float] | None = None, key: str | None = None,
                 interval_s: float = 30.0) -> None:
        self.label = label
        self.total = total
        self.timings = timings
        self.key = key or label
        self.interval_s = interval_s
        self.elapsed_s = 0.0
        self._completed = 0
        self._detail = ""
        self._stop = threading.Event()

    def _emit(self, message: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {self.label}: {message}", flush=True)

    def __enter__(self) -> Progress:
        self._started = self._last_progress = time.perf_counter()
        self._emit("starting" + (f" ({self.total} items)" if self.total is not None else ""))
        self._thread = threading.Thread(target=self._heartbeat, name="sem-progress", daemon=True)
        self._thread.start()
        return self

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval_s):
            count = f"{self._completed}/{self.total} completed; " if self.total is not None else ""
            detail = f"; {self._detail}" if self._detail else ""
            self._emit(f"still running; {count}{time.perf_counter() - self._started:.1f}s elapsed{detail}")

    def update(self, completed: int, detail: str = "") -> None:
        """Report completed items, throttled to the first, every 16, and last."""
        self._completed, self._detail = completed, detail
        now = time.perf_counter()
        if completed == 1 or completed % 16 == 0 or completed == self.total or now - self._last_progress >= self.interval_s:
            count = f"{completed}/{self.total}" if self.total is not None else str(completed)
            self._emit(f"{count}; {now - self._started:.1f}s elapsed" + (f"; {detail}" if detail else ""))
            self._last_progress = now

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        self._thread.join()
        self.elapsed_s = time.perf_counter() - self._started
        if self.timings is not None:
            self.timings[self.key] = self.timings.get(self.key, 0.0) + self.elapsed_s
        if exc_type is not None:
            self._emit(f"failed after {self.elapsed_s:.1f}s ({exc_type.__name__}: {exc_value})")
        else:
            self._emit(f"complete in {self.elapsed_s:.1f}s")
