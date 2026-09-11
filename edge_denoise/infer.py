"""Inference: load a checkpoint and denoise measurements in one forward pass.

The estimator is deterministic by construction -- no sampler, no injected
noise -- so repeated calls on the same frame return the same image and every
bit of output variation is transmitted input noise.  :meth:`Denoiser.denoise`
takes batches at the training resolution; a measurement of any larger size
(e.g. a full 512x512 frame) goes through :meth:`Denoiser.denoise_full`, which
covers it with overlapping training-resolution tiles and blends them -- the
same full-frame protocol every study evaluation uses.  Nothing is ever
resized: resampling a noisy frame partially denoises it and changes the
statistics the model was trained on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from burst_diffusion.ema import EMAHelper

from .config import Config
from .distill import denoise_full_frame
from .image_io import collapse_grayscale_rgb
from .model import EdgeDenoiser, build_model
from .train import load_checkpoint, resolve_device

_SIXTEEN_BIT_MODES = ("I;16", "I;16L", "I;16B", "I;16N", "I")


def load_measurement01(
    path: str | Path, *, black_level: float | None = None, white_level: float | None = None,
) -> np.ndarray:
    """Load a grayscale measurement as a full ``[H, W]`` float64 array in [0, 1].

    The whole frame, uncropped and unresized. Real checkpoints supply fixed
    detector levels; legacy callers use the storage range (255 or 65535).
    """
    with Image.open(Path(path)) as image:
        if (black_level is None) != (white_level is None):
            raise ValueError("supply both black_level and white_level")
        if white_level is not None:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("export one measurement frame per image file")
            array = np.asarray(image, dtype=np.float64)
            array = collapse_grayscale_rgb(array, mode=image.mode, path=path)
            if array.ndim != 2 or not np.isfinite([black_level, white_level]).all() or white_level <= black_level:
                raise ValueError("expected a grayscale image and finite black_level < white_level")
            return np.clip((array - black_level) / (white_level - black_level), 0, 1)
        if image.mode in _SIXTEEN_BIT_MODES:
            array = np.asarray(image, dtype=np.float64)
            return np.clip(array / 65535.0, 0.0, 1.0)
        return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


class Denoiser:
    def __init__(self, model: EdgeDenoiser, *, config: Config, device: torch.device):
        self.model = model.to(device).eval()
        self.config = config
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, device: str = "auto", use_ema: bool = True
    ) -> "Denoiser":
        resolved = resolve_device(device)
        payload = load_checkpoint(path, map_location=resolved)
        config = Config.model_validate(payload["config"])
        model = build_model(config)
        model.load_state_dict(payload["model"])
        if use_ema:
            if payload["ema"] is None:
                import warnings

                warnings.warn(
                    f"use_ema=True but {path} has no EMA state; using the live weights",
                    stacklevel=2,
                )
            else:
                helper = EMAHelper()
                helper.load_state_dict(payload["ema"])
                named = dict(model.named_parameters())
                for name, value in helper.shadow.items():
                    named[name].data.copy_(value)
        return cls(model, config=config, device=resolved)

    @property
    def image_size(self) -> int:
        return self.config.data.image_size

    def load_measurement(self, path: str | Path) -> np.ndarray:
        """Load native measurements with the training checkpoint's normalization."""
        return load_measurement01(path, black_level=self.config.data.black_level,
                                  white_level=self.config.data.white_level)

    def denoise(self, frames: torch.Tensor) -> torch.Tensor:
        """``[B, 1, S, S]`` noisy frames in [-1, 1] -> denoised images, clamped,
        on CPU.  S must equal the training resolution."""
        if frames.dim() != 4:
            raise ValueError(f"frames must be [B, C, H, W], got shape {tuple(frames.shape)}")
        with torch.no_grad():
            batch = frames.to(device=self.device, dtype=torch.float32)
            denoised = self.model.predict_image(batch)
        return denoised.clamp(-1.0, 1.0).cpu()

    def denoise_full(
        self, frame01: np.ndarray, *, stride: int = 48, tile_batch: int = 64
    ) -> np.ndarray:
        """Denoise a full ``[H, W]`` float frame in [0, 1] of any size >= the
        training resolution; returns the same format.

        The backbone carries attention blocks whose placement is fixed by the
        training resolution, so a larger frame cannot go through in one pass;
        it is covered with overlapping training-resolution tiles blended by
        the shared windowed blender (:func:`edge_denoise.distill.denoise_full_frame`).
        """
        return denoise_full_frame(
            self.denoise,
            np.asarray(frame01, dtype=np.float64),
            tile=self.image_size,
            stride=min(stride, self.image_size),
            tile_batch=tile_batch,
        )

    def denoise01(self, frames01: list[np.ndarray], *, max_batch: int = 10) -> list[np.ndarray]:
        """Denoise a list of ``[H, W, C]`` float arrays in [0, 1] (the exchange
        format of the repeatability evaluation); returns the same format."""
        outputs: list[np.ndarray] = []
        for start in range(0, len(frames01), max_batch):
            chunk = frames01[start : start + max_batch]
            stacked = np.stack(
                [np.ascontiguousarray(frame.transpose(2, 0, 1)) for frame in chunk]
            ).astype(np.float32)
            tensor = torch.from_numpy(stacked * 2.0 - 1.0)
            denoised = self.denoise(tensor)
            array01 = ((denoised + 1.0) / 2.0).numpy()
            outputs.extend(
                np.ascontiguousarray(array01[index].transpose(1, 2, 0)).astype(np.float64)
                for index in range(array01.shape[0])
            )
        return outputs
