"""Known-truth PNG data for a runnable demonstration, never real training data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def create_demo(output: str | Path, *, frames: int = 128, size: int = 128, seed: int = 23) -> None:
    """Generate white Gaussian, signal-dependent, and drifting/correlated sites."""
    if frames < 8 or size < 64:
        raise ValueError("demo requires at least 8 frames and 64x64 pixels")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    specimen = 600 + 220 * (np.sin(xx / 12) > 0) + 150 * (np.cos(yy / 17 + xx / 45) > 0)
    specimen = ndimage.gaussian_filter(specimen.astype(float), 1.5)
    specimen += 60 * ndimage.gaussian_filter(rng.normal(size=(size, size)), 3)
    rows, truth = [], {}
    for name in ("01_white", "02_signal_dependent", "03_drift_charging"):
        directory = output / name
        directory.mkdir()
        truth[name] = {"noise": "Gaussian sigma=12 DN" if name != "02_signal_dependent" else "0.4 * Poisson(signal/0.4) + Gaussian sigma=5 DN",
                       "variance_slope_dn": 0.4 if name == "02_signal_dependent" else 0,
                       "variance_intercept_dn2": 25 if name == "02_signal_dependent" else 144,
                       "additional_temporal_ar1_rho": 0.8 if name == "03_drift_charging" else 0,
                       "additional_ar1_innovation_sigma_dn": 4 if name == "03_drift_charging" else 0,
                       "additional_row_sigma_dn": 2 if name == "03_drift_charging" else 0,
                       "drift_y_px": [], "drift_x_px": [], "offset_dn": []}
        correlated = np.zeros_like(specimen)
        for i in range(frames):
            fraction = i / max(frames - 1, 1)
            shift = np.array([2.2 * fraction, -3.4 * fraction]) if name == "03_drift_charging" else np.zeros(2)
            offset = 30 * fraction if name == "03_drift_charging" else 0
            clean = ndimage.shift(specimen, shift, order=3, mode="reflect")
            if name == "02_signal_dependent":
                image = 0.4 * rng.poisson(clean / 0.4) + rng.normal(0, 5, clean.shape)
            else:
                image = clean + rng.normal(0, 12, clean.shape)
            if name == "03_drift_charging":
                correlated = 0.8 * correlated + rng.normal(0, 4, clean.shape)
                image += correlated + offset + rng.normal(0, 2, (size, 1))
            relative = f"{name}/frame_{i:04d}.png"
            Image.fromarray(np.clip(np.rint(image), 0, 65535).astype(np.uint16)).save(output / relative)
            rows.append({"site": name, "path": relative, "frame_index": i, "timestamp_s": 2.0 * i,
                         "include": "true", "acquisition": "synthetic; not equipment measurements"})
            truth[name]["drift_y_px"].append(float(shift[0]))
            truth[name]["drift_x_px"].append(float(shift[1]))
            truth[name]["offset_dn"].append(float(offset))
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
