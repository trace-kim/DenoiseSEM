"""Naive DDIM denoising of real noisy SEM images: treat each image as x_t.

Each input image is loaded and transformed as in training (grayscale,
resized to image_size x image_size, [0,1] -> [-1,1]), placed at diffusion timestep ``--t`` and
run through the deterministic DDIM reverse process (eta=0) down to t=0.

    python tools/ddim_denoise_from_t.py --run-dir experiments/sem512/logs/run1 \
        --input data/real_sem/test --out output/ddim_t210 --t 210 --timesteps 100

``--t`` is the diffusion timestep (0 .. num_diffusion_timesteps-1). If the
step was picked from ``ddim.main --sample --sequence`` images ``{j}_{i}.png``,
pass ``--sequence-index i`` instead with the same ``--timesteps``; the script
converts it to t the way that sampler built its sequence. Note those images
show the *predicted x0* at each step, not x_t.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as tvu
import yaml
from PIL import Image

from ddim.datasets import data_transform, inverse_data_transform
from ddim.functions.denoising import generalized_steps
from ddim.main import dict2namespace
from ddim.models.diffusion import Model
from ddim.runners.diffusion import get_beta_schedule

EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


def load_model(run_dir, ckpt_name, config, device):
    states = torch.load(os.path.join(run_dir, ckpt_name), map_location=device)
    model = Model(config)
    weights = {k.removeprefix("module."): v for k, v in states[0].items()}
    model.load_state_dict(weights, strict=True)
    if config.model.ema:
        shadow = states[-1]
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(shadow[name].data)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="<exp>/logs/<doc>: holds config.yml and ckpt.pth")
    parser.add_argument("--ckpt", default="ckpt.pth", help="checkpoint file inside --run-dir")
    parser.add_argument("--input", required=True, help="folder of noisy SEM images (or one image)")
    parser.add_argument("--out", required=True, help="output folder")
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--t", type=int, help="diffusion timestep the noisy image is treated as")
    start.add_argument("--sequence-index", type=int, help="index i of a --sequence image {j}_{i}.png")
    parser.add_argument("--timesteps", type=int, default=100, help="DDIM step count over the full schedule (same as ddim.main)")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument(
        "--scale-input",
        action="store_true",
        help="multiply the image by sqrt(alpha_bar_t) before inserting (x_t = sqrt(a)x0 + sqrt(1-a)eps)",
    )
    parser.add_argument("--save-trajectory", action="store_true", help="also save x0 prediction at every step")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("generalized_steps runs on CUDA only")
    device = torch.device("cuda")

    with open(os.path.join(args.run_dir, "config.yml"), "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.device = device

    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)
    num_timesteps = betas.shape[0]
    skip = num_timesteps // args.timesteps
    full_seq = list(range(0, num_timesteps, skip))

    t = args.t if args.t is not None else full_seq[len(full_seq) - 1 - args.sequence_index]
    if not 0 <= t < num_timesteps:
        sys.exit(f"--t must lie in [0, {num_timesteps - 1}], got {t}")
    seq = [s for s in full_seq if s < t] + [t]

    alpha_bar = (1 - betas).cumprod(dim=0)[t].item()
    sigma_equiv = np.sqrt((1 - alpha_bar) / alpha_bar)
    print(
        f"t={t}  steps={len(seq)}  alpha_bar={alpha_bar:.4f}  "
        f"noise std of x_t/sqrt(alpha_bar) in [0,1] intensity units = {sigma_equiv / 2:.4f}"
    )

    model = load_model(args.run_dir, args.ckpt, config, device)
    # The U-Net asserts a square input of exactly image_size; identical to the
    # training Resize(image_size) for square images.
    size = config.data.image_size
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    if os.path.isdir(args.input):
        paths = sorted(
            os.path.join(args.input, name)
            for name in os.listdir(args.input)
            if name.lower().endswith(EXTENSIONS)
        )
    else:
        paths = [args.input]
    os.makedirs(args.out, exist_ok=True)

    for path in paths:
        with Image.open(path) as image:
            x = transform(image.convert("L" if config.data.channels == 1 else "RGB"))
        x = data_transform(config, x[None].to(device))
        if args.scale_input:
            x = x * alpha_bar ** 0.5
        xs, x0_preds = generalized_steps(x, seq, model, betas, eta=args.eta)

        stem = os.path.splitext(os.path.basename(path))[0]
        tvu.save_image(inverse_data_transform(config, xs[-1].to(device)), os.path.join(args.out, f"{stem}.png"))
        if args.save_trajectory:
            for i, x0 in enumerate(x0_preds):
                tvu.save_image(
                    inverse_data_transform(config, x0.to(device)),
                    os.path.join(args.out, f"{stem}_x0_step{i:03d}.png"),
                )
        print(f"{path} -> {stem}.png")


if __name__ == "__main__":
    main()
