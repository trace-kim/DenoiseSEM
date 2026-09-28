"""Conditional single-frame precision with a known saved-uint8 template.

Diagnostic only: a test site's own average is not available to a deployable
single-frame denoiser. Gain/offset are nuisance parameters, never corrections
to exported images. The scale convention is physical diameter:
I(x) = g T(c + (x-c-d)/s) + o, ECD = s * ECD_template.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def fit_template_batch(templates: np.ndarray, images: np.ndarray, annuli: np.ndarray,
                       *, device: str = "cpu", centers_yx: np.ndarray | None = None,
                       initial_shifts_yx: np.ndarray | None = None,
                       max_iterations: int = 40) -> list[dict]:
    """Fit shift(2), log-scale, gain and offset in bounded groups of 64 fits.

    Both image arrays are decoded uint8 crops. The fixed annulus is in the
    observation grid; sampling must retain full cubic support in the template.
    Standard errors use the Gauss-Newton residual sandwich covariance (HC1),
    allowing independent Poisson as well as constant-variance Gaussian noise.
    They are conditional on this template and exclude uncertainty/correlated
    noise in the site's average. No unmeasurable fit becomes a zero deviation.
    """
    if (templates.dtype != np.uint8 or images.dtype != np.uint8 or templates.ndim != 3
            or templates.shape != images.shape or annuli.shape != images.shape or not images.size
            or min(images.shape[1:]) < 8):
        raise ValueError("template fits require matching saved uint8 crops and annuli")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    selected_device = torch.device(device)
    if selected_device.type not in {"cpu", "cuda"}:
        raise ValueError("template device must be cpu or cuda:N")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA template fitting requested but unavailable")
    count, height, width = images.shape
    centers = (np.repeat(np.array([[(height - 1) / 2, (width - 1) / 2]]), count, axis=0)
               if centers_yx is None else np.asarray(centers_yx, dtype=float))
    shifts = np.zeros((count, 2)) if initial_shifts_yx is None else np.asarray(initial_shifts_yx, dtype=float)
    if centers.shape != (count, 2) or shifts.shape != (count, 2):
        raise ValueError("one center and initial shift is required per fit")
    if not np.isfinite(centers).all() or not np.isfinite(shifts).all():
        raise ValueError("centers and initial shifts must be finite")
    results = []
    # In addition to the fit-count bound, cap template storage and profile scratch.
    batch_size = max(1, min(64, (128 * 1024**2) // (height * width * 256)))
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        part_images, part_templates = images[start:stop], templates[start:stop]
        block = min(16, max(1, min(height, width) // 2))
        def structured(values):
            pixels = torch.as_tensor(values, device=selected_device, dtype=torch.float32)[:, None] / 255
            return (F.avg_pool2d(pixels, block, ceil_mode=True).flatten(1).std(1, correction=0) >= .005).cpu().numpy()
        usable = structured(part_images) & structured(part_templates)
        fitted = iter(_fit(part_templates[usable], part_images[usable], annuli[start:stop][usable],
                           centers[start:stop][usable], shifts[start:stop][usable], selected_device, max_iterations)
                      if usable.any() else [])
        for valid in usable:
            results.append(next(fitted) if valid else {"status": "skipped_low_contrast", "scale": None,
                "scale_se": None, "reason": "template or acquisition lacks usable structure"})
    return results


def _fit(templates, images, annuli, centers, shifts, device, max_iterations):
    n, height, width = templates.shape
    locations = [np.column_stack(np.nonzero(mask)) for mask in annuli]
    samples = max(max(map(len, locations)), 6)
    coordinates = np.zeros((n, samples, 2))
    measured = np.zeros((n, samples))
    weights = np.zeros((n, samples))
    for i, points in enumerate(locations):
        coordinates[i, :len(points)] = points
        measured[i, :len(points)] = images[i, points[:, 0], points[:, 1]]
        weights[i, :len(points)] = 1
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device=device)
    template = tensor(templates)[:, None]
    xy, observed, mask, center = map(tensor, (coordinates, measured, weights, centers))
    parameters = torch.zeros((n, 5), device=device, dtype=torch.float64)
    parameters[:, :2] = tensor(shifts)
    parameters[:, 3] = 1
    scale_xy = tensor([2 / (height - 1), 2 / (width - 1)])
    epsilon = .02
    perturbations = tensor([[0, 0], [epsilon, 0], [-epsilon, 0], [0, epsilon], [0, -epsilon]])

    def evaluate(p, *, jacobian=False):
        s = p[:, 2].exp()
        q = (xy - center[:, None] - p[:, None, :2]) / s[:, None, None] + center[:, None]
        offsets = perturbations if jacobian else perturbations[:1]
        points = q[:, None] + offsets[None, :, None]
        grid = (points * scale_xy - 1).flip(-1)
        values = F.grid_sample(template, grid, mode="bicubic", align_corners=True)[:, 0]
        value = values[:, 0]
        residual = observed - (p[:, 3, None] * value + p[:, 4, None])
        if not jacobian:
            return residual, q
        dy = (values[:, 1] - values[:, 2]) / (2 * epsilon)
        dx = (values[:, 3] - values[:, 4]) / (2 * epsilon)
        gain = p[:, 3, None]
        jac = torch.stack((-gain * dy / s[:, None], -gain * dx / s[:, None],
                           -gain * (dy * (q[..., 0] - center[:, 0, None]) + dx * (q[..., 1] - center[:, 1, None])),
                           value, torch.ones_like(value)), dim=-1)
        return residual, q, jac

    converged = torch.zeros(n, dtype=torch.bool, device=device)
    limits = tensor([1., 1., .03, .25, 20.])
    for iteration in range(max_iterations):
        residual, _, jac = evaluate(parameters, jacobian=True)
        norms = (jac.square() * mask[..., None]).sum(1).sqrt().clamp_min(1e-12)
        normalized = jac / norms[:, None]
        hessian = normalized.transpose(1, 2) @ (normalized * mask[..., None])
        rhs = (normalized * (residual * mask)[..., None]).sum(1)
        step = torch.linalg.solve(hessian + torch.eye(5, device=device)[None] * 1e-8, rhs[..., None])[..., 0] / norms
        step /= (step.abs() / limits).amax(1).clamp_min(1)[:, None]
        current = (residual.square() * mask).sum(1)
        accepted = converged.clone()
        update = parameters.clone()
        for fraction in (1., .5, .25, .125, .0625):
            candidate = parameters + step * fraction
            trial, _ = evaluate(candidate)
            loss = (trial.square() * mask).sum(1)
            take = (~accepted) & torch.isfinite(loss) & (loss <= current)
            update[take] = candidate[take]
            accepted |= take
        small = (step.abs() / limits).amax(1) < 1e-5
        # At a least-squares minimum a sub-tolerance line search can stall.
        converged |= small | (~accepted & ((step.abs() / limits).amax(1) < 1e-3))
        parameters = update
        if converged.all():
            break
    residual, coordinates, jac = evaluate(parameters, jacobian=True)
    information = jac.transpose(1, 2) @ (jac * mask[..., None])
    inverse = torch.linalg.pinv(information, hermitian=True)
    meat = jac.transpose(1, 2) @ (jac * (residual.square() * mask)[..., None])
    counts = mask.sum(1)
    covariance = inverse @ meat @ inverse * (counts / (counts - 5).clamp_min(1))[:, None, None]
    norms = information.diagonal(dim1=1, dim2=2).sqrt().clamp_min(1e-12)
    eigenvalues = torch.linalg.eigvalsh(information / norms[:, :, None] / norms[:, None, :])
    support = ((coordinates[..., 0] >= 2) & (coordinates[..., 0] <= height - 3)
               & (coordinates[..., 1] >= 2) & (coordinates[..., 1] <= width - 3)) | (mask == 0)
    p, cov, ranks, covered, done = (value.cpu().numpy() for value in
        (parameters, covariance, eigenvalues[:, 0] > 1e-8, support.all(1), converged))
    output = []
    for i in range(n):
        status = ("insufficient_support" if len(locations[i]) < 20 or not covered[i] else
                  "unmeasurable" if not ranks[i] else "not_converged" if not done[i] else "valid")
        if status == "valid" and (not np.isfinite(p[i]).all() or p[i, 3] <= 0):
            status = "invalid_fit"
        scale = float(np.exp(p[i, 2]))
        output.append({"status": status, "scale": scale, "dy_px": float(p[i, 0]), "dx_px": float(p[i, 1]),
            "gain": float(p[i, 3]), "offset_dn": float(p[i, 4]), "iterations": iteration + 1,
            "scale_se": float(scale * np.sqrt(max(0, cov[i, 2, 2]))) if status == "valid" else None,
            "covariance": cov[i].tolist() if status == "valid" else None,
            "parameter_order": ["dy_px", "dx_px", "log_scale", "gain", "offset_dn"],
            "annulus_pixels": len(locations[i])})
    return output
