from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


MU_WATER = 9.6e-2
OFFICIAL_GAUSSIAN_MODE = "reflect"
OFFICIAL_GAUSSIAN_TRUNCATE = 4.0
GLOBAL_PHYSICS_CACHE_VERSION = 2


def hu_to_mu(ct_hu: torch.Tensor) -> torch.Tensor:
    """Convert CT HU to 511-keV linear attenuation using the BIC-MAC rule."""
    hu1000 = ct_hu + 1000.0
    mu_low = 9.6e-5 * hu1000
    mu_high = 5.10e-5 * hu1000 + 4.71e-2
    return torch.clamp(torch.where(hu1000 < 1047.0, mu_low, mu_high), min=0.0)


def _gaussian_kernel1d(
    sigma: float,
    device,
    dtype,
    truncate: float = OFFICIAL_GAUSSIAN_TRUNCATE,
) -> torch.Tensor:
    if sigma <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    if truncate <= 0:
        raise ValueError(f"Gaussian truncate must be positive, got {truncate}")
    # scipy.ndimage.gaussian_filter rounds the radius this way.
    radius = max(1, int(float(truncate) * sigma + 0.5))
    coordinate = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coordinate / sigma) ** 2)
    return kernel / kernel.sum()


def _pad_for_axis(x: torch.Tensor, axis: int, amount: int, mode: str) -> torch.Tensor:
    # F.pad lists dimensions from W to D for [B, C, D, H, W].
    pads = {
        0: (0, 0, 0, 0, amount, amount),
        1: (0, 0, amount, amount, 0, 0),
        2: (amount, amount, 0, 0, 0, 0),
    }
    if mode == "constant":
        return F.pad(x, pads[axis], mode=mode, value=0.0)
    if mode == "reflect":
        # Match scipy.ndimage mode="reflect" (half-sample symmetric), including
        # repeated edge voxels. torch F.pad(mode="reflect") uses the other
        # reflection convention and cannot pad by >= the input length.
        dimension = axis + 2
        length = int(x.shape[dimension])
        if length <= 0:
            raise ValueError(f"Cannot reflect-pad empty axis {axis}")
        positions = torch.arange(-amount, length + amount, device=x.device)
        period = 2 * length
        reflected = torch.remainder(positions, period)
        reflected = torch.where(reflected < length, reflected, period - 1 - reflected).long()
        return x.index_select(dimension, reflected)
    return F.pad(x, pads[axis], mode=mode)


def gaussian_smooth_3d(
    x: torch.Tensor,
    spacing_mm: Sequence[float],
    fwhm_mm: float = 4.0,
    padding_mode: str = OFFICIAL_GAUSSIAN_MODE,
    truncate: float = OFFICIAL_GAUSSIAN_TRUNCATE,
) -> torch.Tensor:
    """Apply spacing-aware separable Gaussian smoothing to [B,C,X,Y,Z].

    ``replicate`` padding is appropriate for a training patch cut from a larger
    body because it avoids inventing zero attenuation at the patch boundary.
    ``reflect`` with truncate 4 matches the official SciPy reconstruction
    smoothing for a complete scanner field of view.
    """
    if fwhm_mm <= 0:
        return x
    if padding_mode not in {"constant", "replicate", "reflect"}:
        raise ValueError(f"Unsupported Gaussian padding_mode={padding_mode!r}")
    if x.ndim != 5:
        raise ValueError(f"Expected [B,C,X,Y,Z], got {tuple(x.shape)}")

    sigma_mm = float(fwhm_mm) / 2.355
    sx, sy, sz = [float(value) for value in spacing_mm]
    sigmas = [sigma_mm / sx, sigma_mm / sy, sigma_mm / sz]
    out = x
    channels = x.shape[1]

    for axis, sigma in enumerate(sigmas):
        kernel = _gaussian_kernel1d(sigma, x.device, x.dtype, truncate=truncate)
        if kernel.numel() == 1:
            continue
        radius = kernel.numel() // 2
        padded = _pad_for_axis(out, axis, radius, padding_mode)
        if axis == 0:
            weight = kernel.view(1, 1, -1, 1, 1).repeat(channels, 1, 1, 1, 1)
        elif axis == 1:
            weight = kernel.view(1, 1, 1, -1, 1).repeat(channels, 1, 1, 1, 1)
        else:
            weight = kernel.view(1, 1, 1, 1, -1).repeat(channels, 1, 1, 1, 1)
        out = F.conv3d(padded, weight, groups=channels)
    return out


def compute_global_mu_targets(
    ct_hu: torch.Tensor,
    spacing_mm: Sequence[float],
    spatial_size: Sequence[int],
    fwhm_mm: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create conservative low-resolution targets in the official physics order.

    The full-resolution, unclipped CT is converted to mu and smoothed before
    area downsampling. Returned tensors are channel-first [1,X,Y,Z].
    """
    ct = torch.as_tensor(ct_hu, dtype=torch.float32).detach().cpu()
    if ct.ndim == 3:
        ct = ct.unsqueeze(0)
    if ct.ndim != 4 or ct.shape[0] != 1:
        raise ValueError(f"Expected one channel [1,X,Y,Z] CT, got {tuple(ct.shape)}")
    target_size = tuple(int(value) for value in spatial_size)
    if len(target_size) != 3 or any(value <= 0 for value in target_size):
        raise ValueError(f"Invalid global target size: {spatial_size}")

    with torch.inference_mode():
        full_mu = hu_to_mu(ct.unsqueeze(0))
        if fwhm_mm > 0:
            try:
                import numpy as np
                from scipy.ndimage import gaussian_filter

                sigma_mm = float(fwhm_mm) / 2.355
                sigma_voxel = tuple(sigma_mm / float(value) for value in spacing_mm)
                source = full_mu.numpy()[0, 0]
                smoothed = np.empty_like(source, dtype=np.float32)
                gaussian_filter(
                    source,
                    sigma=sigma_voxel,
                    output=smoothed,
                    mode=OFFICIAL_GAUSSIAN_MODE,
                    truncate=OFFICIAL_GAUSSIAN_TRUNCATE,
                )
                full_mu_smooth = torch.from_numpy(smoothed).view_as(full_mu)
            except ImportError:
                full_mu_smooth = gaussian_smooth_3d(
                    full_mu,
                    spacing_mm=spacing_mm,
                    fwhm_mm=fwhm_mm,
                    padding_mode=OFFICIAL_GAUSSIAN_MODE,
                    truncate=OFFICIAL_GAUSSIAN_TRUNCATE,
                )
        else:
            full_mu_smooth = full_mu
        low_mu = F.interpolate(full_mu, size=target_size, mode="area").squeeze(0).contiguous()
        low_mu_smooth = F.interpolate(full_mu_smooth, size=target_size, mode="area").squeeze(0).contiguous()
    return low_mu.float(), low_mu_smooth.float()


def _cache_matches(payload: dict, original_shape, spacing_mm, spatial_size, fwhm_mm: float) -> bool:
    if int(payload.get("version", -1)) != GLOBAL_PHYSICS_CACHE_VERSION:
        return False
    if tuple(payload.get("original_shape", ())) != tuple(int(v) for v in original_shape):
        return False
    if tuple(payload.get("spatial_size", ())) != tuple(int(v) for v in spatial_size):
        return False
    if abs(float(payload.get("fwhm_mm", -1.0)) - float(fwhm_mm)) > 1e-6:
        return False
    cached_spacing = torch.as_tensor(payload.get("spacing_mm", []), dtype=torch.float64)
    expected_spacing = torch.as_tensor(spacing_mm, dtype=torch.float64)
    return cached_spacing.shape == expected_spacing.shape and bool(
        torch.allclose(cached_spacing, expected_spacing, atol=1e-5, rtol=1e-5)
    )


def load_or_compute_global_mu_targets(
    ct_hu: torch.Tensor,
    subject_id: str,
    spacing_mm: Sequence[float],
    spatial_size: Sequence[int],
    fwhm_mm: float,
    cache_dir: str | os.PathLike | None,
    missing_policy: str = "compute",
    overwrite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Load a validated physics cache or compute and atomically store it."""
    ct = torch.as_tensor(ct_hu)
    original_shape = tuple(int(value) for value in ct.shape[-3:])
    cache_path = None
    if cache_dir:
        safe_subject = Path(str(subject_id)).name
        cache_path = Path(cache_dir) / f"{safe_subject}.pt"

    if cache_path is not None and cache_path.exists() and not overwrite:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        if _cache_matches(payload, original_shape, spacing_mm, spatial_size, fwhm_mm):
            return payload["mu_raw"].float(), payload["mu_smooth"].float(), True

    if missing_policy == "error":
        raise FileNotFoundError(
            f"Missing or stale whole-body physics cache for {subject_id}: {cache_path}. "
            "Use physics_cache_missing='compute' or populate the configured cache."
        )
    if missing_policy != "compute":
        raise ValueError(f"physics_cache_missing must be 'compute' or 'error', got {missing_policy!r}")

    mu_raw, mu_smooth = compute_global_mu_targets(ct, spacing_mm, spatial_size, fwhm_mm)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": GLOBAL_PHYSICS_CACHE_VERSION,
            "subject_id": str(subject_id),
            "original_shape": original_shape,
            "spacing_mm": tuple(float(value) for value in spacing_mm),
            "spatial_size": tuple(int(value) for value in spatial_size),
            "fwhm_mm": float(fwhm_mm),
            "mu_raw": mu_raw,
            "mu_smooth": mu_smooth,
        }
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, cache_path)
    return mu_raw, mu_smooth, False
