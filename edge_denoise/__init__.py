"""Edge-preserving deterministic denoisers for SEM metrology.

One single-pass regression U-Net, three input/output representations (plain
image -- which with a fresh-frame target is exactly Noise2Noise -- pure Sobel
gradient with FFT least-squares image recovery, and a hybrid of both), and a
loss that can weigh image fidelity, gradient fidelity, and two-realization
consistency.  The premise, derived and tested in docs/edge_denoise_method.md:
CD and registration live in edge positions, so training should weigh edges --
which an intensity MSE demonstrably does not (the burst report's finding of
edge-concentrated residual variance).

Dependency policy: imports ``burst_diffusion`` for the burst dataset cache
(the audited content-group split), the U-Net backbone (identical capacity =>
clean comparisons), EMA, metrics, and the repeatability harness -- and through
it ``noising_pipeline``.  Never imports ``ddim`` or ``runctl``.
"""

from .config import Config, load_config
from .data import PairBatch, PairFactory, ValPairBatch
from .evaluate import evaluate
from .gradient import SOBEL_NOISE_GAIN, reconstruct_from_sobel, sobel, sobel_kernels
from .infer import Denoiser
from .model import EdgeDenoiser, build_model, make_input
from .provider import providers_from_checkpoints, realization_provider
from .train import Trainer, load_checkpoint, save_checkpoint

__all__ = [
    "Config",
    "Denoiser",
    "EdgeDenoiser",
    "PairBatch",
    "PairFactory",
    "SOBEL_NOISE_GAIN",
    "Trainer",
    "ValPairBatch",
    "build_model",
    "evaluate",
    "load_checkpoint",
    "load_config",
    "make_input",
    "providers_from_checkpoints",
    "realization_provider",
    "reconstruct_from_sobel",
    "save_checkpoint",
    "sobel",
    "sobel_kernels",
]
