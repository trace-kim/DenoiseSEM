"""Inference: load a checkpoint and denoise measurements in one forward pass.

The estimator is deterministic by construction -- no sampler, no injected
noise -- so repeated calls on the same frame return the same image and every
bit of output variation is transmitted input noise.  Inputs must be at the
training resolution (crops only, never resizes: resampling a noisy frame
partially denoises it and changes the statistics the model was trained on).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from burst_diffusion.ema import EMAHelper

from .config import Config
from .model import EdgeDenoiser, build_model
from .train import load_checkpoint, resolve_device


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

    def denoise(self, frames: torch.Tensor) -> torch.Tensor:
        """``[B, 1, S, S]`` noisy frames in [-1, 1] -> denoised images, clamped,
        on CPU.  S must equal the training resolution."""
        if frames.dim() != 4:
            raise ValueError(f"frames must be [B, C, H, W], got shape {tuple(frames.shape)}")
        with torch.no_grad():
            batch = frames.to(device=self.device, dtype=torch.float32)
            denoised = self.model.predict_image(batch)
        return denoised.clamp(-1.0, 1.0).cpu()

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
