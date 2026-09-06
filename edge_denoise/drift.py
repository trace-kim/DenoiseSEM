"""Drifting-burst generation: the synthetic acquisition every fusion arm trains
and is measured on.

The existing burst datasets are pixel-aligned by construction: every replica
is a fresh Poisson draw of the SAME clean image.  Real SEM bursts are not.
Between frames the stage and beam drift (nm/s against nm-sized pixels, so
several pixels over a 16-frame burst), *within* a frame the raster scans rows
over the frame time so a moving sample shears the frame, and charging slowly
moves brightness and contrast.  A frame average of such a burst is blurred by
the drift, and any training target built by averaging or pairing unregistered
frames blurs the estimator the same way -- the feasibility objection to the
"burst as target" arms.  This module makes that acquisition explicit so the
whole pipeline (registration, fusion, evaluation) can be exercised against it
with a recorded truth.

Model of one burst of ``K`` frames (all in frame units, ``t`` = time):

- drift ``p(t) = v t + w(t)``: a per-burst constant velocity ``v`` (stage
  drift, ``velocity_sigma`` px/frame per axis) plus a random walk ``w`` with
  increments of ``walk_sigma`` px per frame, linearly interpolated inside a
  frame;
- frame ``k`` scans rows ``r = 0..H-1`` at ``t = k + r/(H-1)``, so its
  mid-frame position is ``p(k + 1/2)`` and its drift across the frame
  (velocity) is ``p(k+1) - p(k)``; the row shift is linear in ``r``
  (:func:`edge_denoise.register.warp_scene` conventions);
- the burst is re-anchored so frame 0's mid-frame position is 0: a "retake"
  starts where the reference is, and drifts from there;
- charging: gain ``1 + gain_sigma * g_k`` and offset ``offset_sigma * o_k``
  with AR(1) processes ``g, o`` of unit variance and lag-1 correlation 0.8;
- noise: exactly the stored-frame model of the source dataset,
  ``min(Pois(peak * x), peak) / peak`` with the clean scene warped
  (bicubic) BEFORE the draw, so every frame's noise is independent.

Layout of the output dataset: a burst dataset readable by
:class:`burst_diffusion.data.BurstCache` (``manifest.jsonl``, ``clean/``,
``noisy/``) plus ``drift.json``, the per-frame truth (position, velocity,
gain, offset) organised in bursts of ``frames_per_burst`` replicas.  Training
sources get one burst; held-out sources get ``holdout_retakes`` independent
bursts so that every K-frame method has as many retakes as the single-frame
methods had seeds.  The clean images and source indices are copied verbatim
from the source dataset, so the content-hash split is identical.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from burst_diffusion.data import BurstCache

from .register import warp_scene

DRIFT_FORMAT = 1
DRIFT_FILE_NAME = "drift.json"


@dataclass(frozen=True)
class DriftParams:
    velocity_sigma: float = 0.35  # px/frame per axis, constant within a burst
    walk_sigma: float = 0.25  # px per frame per axis, random-walk increment
    gain_sigma: float = 0.02  # multiplicative brightness drift amplitude
    offset_sigma: float = 0.005  # additive offset drift amplitude ([0, 1] units)
    ar_rho: float = 0.8  # lag-1 correlation of the charging processes

    def __post_init__(self) -> None:
        for name in ("velocity_sigma", "walk_sigma", "gain_sigma", "offset_sigma"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0.0 <= self.ar_rho < 1.0:
            raise ValueError(f"ar_rho must be in [0, 1), got {self.ar_rho}")


@dataclass
class BurstTruth:
    """The recorded acquisition of one burst."""

    replicas: list[int]
    position: np.ndarray  # [K, 2] (dy, dx) mid-frame
    velocity: np.ndarray  # [K, 2] px/frame across the frame
    gain: np.ndarray  # [K]
    offset: np.ndarray  # [K]

    def to_json(self) -> dict:
        return {
            "replicas": [int(r) for r in self.replicas],
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "gain": self.gain.tolist(),
            "offset": self.offset.tolist(),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "BurstTruth":
        return cls(
            replicas=[int(r) for r in payload["replicas"]],
            position=np.asarray(payload["position"], dtype=np.float64),
            velocity=np.asarray(payload["velocity"], dtype=np.float64),
            gain=np.asarray(payload["gain"], dtype=np.float64),
            offset=np.asarray(payload["offset"], dtype=np.float64),
        )


def _ar1(rng: np.random.Generator, count: int, rho: float) -> np.ndarray:
    values = np.zeros(count)
    if count == 0:
        return values
    values[0] = rng.normal()
    scale = math.sqrt(max(1.0 - rho * rho, 0.0))
    for k in range(1, count):
        values[k] = rho * values[k - 1] + scale * rng.normal()
    return values


def sample_burst_truth(
    rng: np.random.Generator, frames: int, params: DriftParams, replicas: list[int]
) -> BurstTruth:
    """Draw one burst's drift and charging trajectory (see the module docstring)."""
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    velocity = rng.normal(0.0, params.velocity_sigma, size=2)
    increments = rng.normal(0.0, params.walk_sigma, size=(frames, 2))
    walk = np.concatenate([np.zeros((1, 2)), np.cumsum(increments, axis=0)], axis=0)  # w(0..K)

    def p(t: float) -> np.ndarray:
        k = min(int(math.floor(t)), frames - 1)
        frac = t - k
        w = walk[k] * (1.0 - frac) + walk[k + 1] * frac
        return velocity * t + w

    position = np.stack([p(k + 0.5) for k in range(frames)])
    per_frame_velocity = np.stack([p(k + 1.0) - p(k) for k in range(frames)])
    position = position - position[0][None, :]
    gain = 1.0 + params.gain_sigma * _ar1(rng, frames, params.ar_rho)
    offset = params.offset_sigma * _ar1(rng, frames, params.ar_rho)
    return BurstTruth(
        replicas=list(replicas),
        position=position,
        velocity=per_frame_velocity,
        gain=gain,
        offset=offset,
    )


def render_frame(
    clean01: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    gain: float,
    offset: float,
    *,
    peak: float,
    rng: np.random.Generator,
    device: str = "cpu",
) -> np.ndarray:
    """One stored frame (uint8) of the drifted, charged, Poisson-sampled scene."""
    scene = warp_scene(clean01, position, velocity, device=device)
    signal = np.clip(gain * scene + offset, 0.0, 1.0)
    counts = np.minimum(rng.poisson(signal * peak), peak) / peak
    return np.rint(np.clip(counts, 0.0, 1.0) * 255.0).astype(np.uint8)


def _write_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def generate_drift_dataset(
    source_dataset_dir: str | Path,
    out_dir: str | Path,
    *,
    frames_per_burst: int = 16,
    holdout_retakes: int = 10,
    params: DriftParams = DriftParams(),
    peak: float = 10.0,
    seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    split_seed: int = 2019,
    device: str = "cpu",
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Write a drifting-burst dataset next to its truth; returns the truth dict.

    Training sources (by the source dataset's content-group split) receive
    one burst of ``frames_per_burst`` frames; validation and test sources
    receive ``holdout_retakes`` bursts.  ``peak`` must match the source
    dataset's Poisson peak so single-frame statistics are unchanged.
    """
    if frames_per_burst < 1:
        raise ValueError(f"frames_per_burst must be >= 1, got {frames_per_burst}")
    if holdout_retakes < 1:
        raise ValueError(f"holdout_retakes must be >= 1, got {holdout_retakes}")
    cache = BurstCache(
        source_dataset_dir,
        channels=1,
        min_replicas=1,
        min_size=1,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_seed=split_seed,
    )
    holdout = {source.source_index for source in cache.val_sources + cache.test_sources}
    destination = Path(out_dir)
    burst_dir = destination / "burst"
    if (burst_dir / "manifest.jsonl").exists():
        raise FileExistsError(f"{burst_dir} already holds a dataset; choose another out_dir")
    (burst_dir / "clean").mkdir(parents=True, exist_ok=True)
    (burst_dir / "noisy").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest_rows: list[str] = []
    truth_sources: dict[str, dict] = {}
    for source in sorted(cache.all_sources, key=lambda item: item.source_index):
        index = source.source_index
        clean_name = f"clean/{index:05d}.png"
        _write_png(source.clean, burst_dir / clean_name)
        clean01 = source.clean.astype(np.float64) / 255.0
        bursts = holdout_retakes if index in holdout else 1
        truths: list[BurstTruth] = []
        replica = 0
        for burst in range(bursts):
            replicas = list(range(replica, replica + frames_per_burst))
            truth = sample_burst_truth(rng, frames_per_burst, params, replicas)
            for k, replica_index in enumerate(replicas):
                frame = render_frame(
                    clean01,
                    truth.position[k],
                    truth.velocity[k],
                    float(truth.gain[k]),
                    float(truth.offset[k]),
                    peak=peak,
                    rng=rng,
                    device=device,
                )
                noisy_name = f"noisy/{index:05d}_{replica_index:05d}.png"
                _write_png(frame, burst_dir / noisy_name)
                manifest_rows.append(
                    json.dumps(
                        {
                            "source_index": index,
                            "replica_index": replica_index,
                            "clean_path": clean_name,
                            "noisy_path": noisy_name,
                            "burst_index": burst,
                            "frame_in_burst": k,
                            "shape": list(source.clean.shape[:2]),
                            "noise_types": ["poisson"],
                            "effective_noise_params": {"poisson": {"peak": peak}},
                        },
                        sort_keys=True,
                    )
                )
            truths.append(truth)
            replica += frames_per_burst
        truth_sources[str(index)] = {
            "split": "holdout" if index in holdout else "train",
            "bursts": [truth.to_json() for truth in truths],
        }
        if progress is not None:
            progress(f"source {index}: {bursts} burst(s) x {frames_per_burst} frames")
    (burst_dir / "manifest.jsonl").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    truth = {
        "format": DRIFT_FORMAT,
        "source_dataset_dir": str(source_dataset_dir),
        "frames_per_burst": frames_per_burst,
        "holdout_retakes": holdout_retakes,
        "peak": peak,
        "seed": seed,
        "params": asdict(params),
        "split": {
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "split_seed": split_seed,
            "val_source_indices": [s.source_index for s in cache.val_sources],
            "test_source_indices": [s.source_index for s in cache.test_sources],
        },
        "sources": truth_sources,
    }
    (destination / DRIFT_FILE_NAME).write_text(json.dumps(truth, sort_keys=True), encoding="utf-8")
    return truth


def load_drift_truth(dataset_dir: str | Path) -> dict | None:
    """The truth sidecar of a drifting dataset, or ``None`` for a plain one."""
    root = Path(dataset_dir)
    for candidate in (root / DRIFT_FILE_NAME, root.parent / DRIFT_FILE_NAME):
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if payload.get("format") != DRIFT_FORMAT:
                raise ValueError(f"{candidate} is not a format-{DRIFT_FORMAT} drift truth")
            return payload
    return None


def burst_truths(truth: dict, source_index: int) -> list[BurstTruth]:
    return [BurstTruth.from_json(item) for item in truth["sources"][str(int(source_index))]["bursts"]]
