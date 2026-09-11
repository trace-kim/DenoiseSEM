"""Shared validation for grayscale SEM pixels stored in RGB images."""

from pathlib import Path

import numpy as np


def collapse_grayscale_rgb(array: np.ndarray, *, mode: str, path: str | Path) -> np.ndarray:
    """Extract one decoded RGB channel only when all three match exactly."""
    if mode == "RGB":
        if not (np.array_equal(array[..., 0], array[..., 1])
                and np.array_equal(array[..., 0], array[..., 2])):
            raise ValueError(f"expected identical RGB channels for grayscale SEM image: {path}")
        return array[..., 0]
    return array
