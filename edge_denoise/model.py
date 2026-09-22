"""The denoising network: burst_diffusion's U-Net behind a representation layer.

The backbone is imported from ``burst_diffusion.unet`` ON PURPOSE, not copied:
the decisive comparison this package exists for (does gradient-domain training
improve metrology precision over N2N / burst training?) is only clean if every
arm has the identical architecture and capacity -- the same methodology the
matched-N2N ablation used (`schedule.num_steps: 1` on the same U-Net).  Only
``conv_in`` / ``conv_out`` widths differ, driven by the representation.

The network is a deterministic single-pass regressor; there is no diffusion
timestep.  The backbone's ``t`` conditioning is fed the constant 1.0, exactly
the value the existing N2N arm saw (its BatchFactory emits ``t = 1`` for
``num_steps = 1``), so the constant embedding reduces to a learned per-layer
bias and the comparison stays like-for-like.

Determinism is the point: for the same input frame the estimator returns the
same image, so all measurement variance comes from the input noise -- the
quantity the repeatability evaluation isolates.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from burst_diffusion.unet import UNet

from .config import Config
from .gradient import reconstruct_from_sobel, sobel

#: Constant fed to the backbone's timestep conditioning (see module docstring).
CONSTANT_T = 1.0


def make_input(frames: torch.Tensor, representation: str) -> torch.Tensor:
    """Build the network input from raw noisy frames ``[B, 1, H, W]``."""
    if representation == "image":
        return frames
    if representation == "gradient":
        return sobel(frames)
    if representation == "hybrid":
        return torch.cat([frames, sobel(frames)], dim=1)
    raise ValueError(f"unknown representation {representation!r}")


class EdgeDenoiser(nn.Module):
    """Single-pass denoiser: raw noisy frame in, prediction in the configured
    output domain (image for ``image``/``hybrid``, Sobel field for
    ``gradient``)."""

    def __init__(self, config: Config):
        super().__init__()
        self.representation = config.objective.representation
        self.unet = UNet(
            in_channels=config.in_channels,
            out_ch=config.out_channels,
            ch=config.model.ch,
            ch_mult=config.model.ch_mult,
            num_res_blocks=config.model.num_res_blocks,
            attn_resolutions=config.model.attn_resolutions,
            dropout=config.model.dropout,
            resamp_with_conv=config.model.resamp_with_conv,
            resolution=config.data.image_size,
            num_groups=config.model.effective_num_groups,
            attention=config.model.attention,
        )

    def forward(self, frames: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """``t`` (``[B]``) conditions the backbone; ``None`` = the constant
        every single-frame arm uses.  Burst fusion passes the number of frames
        averaged, so the same network serves every dose level."""
        features = make_input(frames, self.representation)
        if t is None:
            t = torch.full((frames.shape[0],), CONSTANT_T, device=frames.device)
        else:
            t = t.to(device=frames.device, dtype=torch.float32)
            if t.shape != (frames.shape[0],):
                raise ValueError(f"t must be [B] = [{frames.shape[0]}], got {tuple(t.shape)}")
        return self.unet(features, t)

    def predict_image(self, frames: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Denoised IMAGE regardless of representation.

        For ``gradient`` the predicted Sobel field is inverted by the exact
        FFT least-squares solve; the unrecoverable DC offset is restored from
        the noisy input's crop mean. This is a defined part of image production,
        not an output correction fitted during measurement. Real acquisitions
        can have brightness drift; quantization, clipping and the detector can
        affect measured contours. Assess the delivered uint8 images.
        """
        return self.image_from_prediction(self.forward(frames, t), frames)

    def image_from_prediction(self, prediction: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """Recover an image from an existing forward pass (also used in validation)."""
        if self.representation != "gradient":
            return prediction
        return reconstruct_from_sobel(prediction, mean=frames.mean(dim=(1, 2, 3)))


def build_model(config: Config) -> EdgeDenoiser:
    return EdgeDenoiser(config)


def initialize_from_image(model: EdgeDenoiser, state: dict[str, torch.Tensor]) -> None:
    """Function-preserving image -> hybrid warm start, with strict remaining keys.

    Extra Sobel input channels initially contribute zero. This is an explicit
    initialization strategy, never an implicit relaxation of checkpoint loading.
    """
    if model.representation != "hybrid":
        raise ValueError("image_to_hybrid requires a hybrid model")
    key = "unet.conv_in.weight"
    source = state.get(key)
    destination = model.state_dict()[key]
    if (source is None or source.ndim != 4 or source.shape[1] != 1
            or source.shape[0] != destination.shape[0] or source.shape[2:] != destination.shape[2:]):
        raise ValueError("image_to_hybrid requires a compatible one-channel image checkpoint")
    expanded = source.new_zeros(destination.shape)
    expanded[:, :1] = source
    model.load_state_dict({**state, key: expanded}, strict=True)
