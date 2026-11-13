from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluator_surrogate import (
    evaluator_aligned_full_fov_patch_proxy_losses,
    evaluator_aligned_pet_proxy_losses,
)
from pet_physics import (
    MU_WATER,
    OFFICIAL_GAUSSIAN_MODE,
    OFFICIAL_GAUSSIAN_TRUNCATE,
    gaussian_smooth_3d,
    hu_to_mu,
)

EPS = 1e-6


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / (mask.sum() + eps)


def masked_mean_metric(x: torch.Tensor, mask: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    if float(mask.sum().detach().item()) <= eps:
        return x.new_tensor(float("nan"))
    return (x * mask).sum() / (mask.sum() + eps)


def masked_weighted_mean(
    x: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    weighted_mask = mask.to(dtype=x.dtype) * weight.to(dtype=x.dtype)
    return (x * weighted_mask).sum() / (weighted_mask.sum() + eps)


def gradient_3d(x: torch.Tensor):
    dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    return dz, dy, dx




def build_tissue_weight(
    target_hu: torch.Tensor,
    body_mask: torch.Tensor,
    air_threshold: float = -300.0,
    bone_threshold: float = 150.0,
    air_weight: float = 1.5,
    bone_weight: float = 2.0,
    soft_weight: float = 1.0,
) -> torch.Tensor:
    weight = torch.full_like(target_hu, soft_weight)
    weight = torch.where(target_hu < air_threshold, torch.full_like(weight, air_weight), weight)
    weight = torch.where(target_hu > bone_threshold, torch.full_like(weight, bone_weight), weight)
    return weight * (body_mask > 0).to(dtype=weight.dtype)


def _projection_weights(values, dtype, device, expected: int, name: str) -> torch.Tensor:
    weights = torch.as_tensor(values, dtype=dtype, device=device).flatten()
    if weights.numel() != expected:
        raise ValueError(f"{name} must contain {expected} values, got {weights.numel()}")
    if torch.any(weights < 0) or float(weights.sum().detach().item()) <= 0.0:
        raise ValueError(f"{name} must be non-negative and sum to a positive value")
    return weights


def _angle_component_name(angle: float) -> str:
    if float(angle).is_integer():
        value = str(abs(int(angle)))
    else:
        value = f"{abs(float(angle)):.1f}".replace(".", "p")
    prefix = "m" if angle < 0 else "p"
    if abs(float(angle)) < 1e-6:
        return "angle_0"
    return f"angle_{prefix}{value}"


def _supported_ray_error_statistics(
    pred_projection: torch.Tensor,
    target_projection: torch.Tensor,
    supported_rays: torch.Tensor,
    tail_fraction: float = 0.05,
    error_domain: str = "log_acf",
    delta_clip: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Return complementary ray statistics without losing error direction.

    Absolute error measures average calibration, ``signed_mean`` preserves
    over- versus under-attenuation, and ``tail`` is a per-sample CVaR-like
    mean over the worst supported rays. P95 is detached because it is a
    diagnostic, while the top-k tail remains differentiable.
    """
    fraction = float(tail_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"tail_fraction must be in (0, 1], got {fraction}")
    delta_log_acf = pred_projection - target_projection
    if error_domain == "log_acf":
        delta = delta_log_acf
    elif error_domain == "relative_acf":
        clip = float(delta_clip)
        if clip <= 0:
            raise ValueError(f"delta_clip must be positive, got {clip}")
        # ACF_pred / ACF_gt - 1 = exp(delta_log_acf) - 1. Clamping only
        # guards the exponential tail; it leaves the clinically relevant
        # neighborhood around zero unchanged.
        delta = torch.expm1(torch.clamp(delta_log_acf, -clip, clip))
    else:
        raise ValueError(f"Unsupported projection error_domain: {error_domain!r}")
    support = supported_rays > 0
    absolute_delta = torch.abs(delta)
    absolute_error = masked_mean(absolute_delta, support)
    signed_mean = masked_mean(delta, support)

    tails = []
    p95_values = []
    for batch_index in range(int(delta.shape[0])):
        values = absolute_delta[batch_index][support[batch_index]]
        if values.numel() == 0:
            tails.append(delta.new_tensor(0.0))
            p95_values.append(delta.new_tensor(0.0))
            continue
        count = max(1, int(math.ceil(float(values.numel()) * fraction)))
        worst = torch.topk(values, k=count, largest=True, sorted=False).values
        tails.append(worst.mean())
        # Nearest-rank P95 reuses the same top-k result instead of running a
        # second GPU quantile/sort in every training projection.
        p95_values.append(worst.detach().min())

    return {
        "absolute": absolute_error,
        "signed": torch.abs(signed_mean),
        "signed_mean": signed_mean,
        "tail": torch.stack(tails).mean(),
        "p95": torch.stack(p95_values).mean(),
    }


def orthogonal_projection_mu_loss(
    pred_mu: torch.Tensor,
    target_mu: torch.Tensor,
    mask: torch.Tensor | None,
    spacing_mm,
    axis_weights=None,
    ray_support_mask: torch.Tensor | None = None,
    tail_fraction: float = 0.05,
    error_domain: str = "log_acf",
    delta_clip: float = 0.5,
    return_components: bool = False,
):
    """Compare physical attenuation line integrals along tensor D/H/W axes.

    Loaded NIfTI tensors retain x/y/z spatial order, so tensor D/H/W maps to
    NIfTI x/y/z. Mu is measured in cm^-1; multiplying each voxel sum by its
    axis spacing in cm produces a dimensionless attenuation line integral.
    """
    # Projection sums model physical path integrals and must remain FP32 even
    # when the surrounding training forward runs under AMP.
    pred_mu = pred_mu.float()
    target_mu = target_mu.float()
    integration_mask = torch.ones_like(pred_mu) if mask is None else mask.float()
    support = ray_support_mask
    if support is None:
        support = integration_mask
    support = support.float()
    pred = pred_mu * integration_mask
    target = target_mu * integration_mask
    sx, sy, sz = [float(value) for value in spacing_mm]
    spacing_cm = (sx / 10.0, sy / 10.0, sz / 10.0)
    weights = _projection_weights(
        axis_weights if axis_weights is not None else (1.0, 1.0, 1.0),
        pred.dtype,
        pred.device,
        3,
        "projection_axis_weights",
    )

    components = {}
    for name, dim, voxel_spacing_cm in zip(("D", "H", "W"), (2, 3, 4), spacing_cm):
        pred_projection = pred.sum(dim=dim) * voxel_spacing_cm
        target_projection = target.sum(dim=dim) * voxel_spacing_cm
        supported_rays = (support.sum(dim=dim) > EPS).to(dtype=pred.dtype)
        statistics = _supported_ray_error_statistics(
            pred_projection,
            target_projection,
            supported_rays,
            tail_fraction=tail_fraction,
            error_domain=error_domain,
            delta_clip=delta_clip,
        )
        components[name] = statistics["absolute"]
        components[f"signed_{name}"] = statistics["signed"]
        components[f"mean_signed_{name}"] = statistics["signed_mean"]
        components[f"tail_{name}"] = statistics["tail"]
        components[f"p95_{name}"] = statistics["p95"]

    stacked = torch.stack([components["D"], components["H"], components["W"]])
    total = torch.sum(stacked * weights) / weights.sum()
    if return_components:
        return total, components
    return total


def _downsample_projection_tensor(x: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return x
    return F.avg_pool3d(x, kernel_size=factor, stride=factor, ceil_mode=False)


def _pad_dh_to_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Pad x-y to its enclosing square so oblique rotations do not crop anatomy."""
    d_size, h_size = int(x.shape[2]), int(x.shape[3])
    target = int(math.ceil(math.sqrt(float(d_size * d_size + h_size * h_size))))
    pad_d = max(0, target - d_size)
    pad_h = max(0, target - h_size)
    return F.pad(
        x,
        (
            0,
            0,
            pad_h // 2,
            pad_h - pad_h // 2,
            pad_d // 2,
            pad_d - pad_d // 2,
        ),
    )


def _rotate_dh_plane(
    x: torch.Tensor,
    angle_deg: float,
    mode: str = "bilinear",
    pad_to_diagonal: bool = True,
) -> torch.Tensor:
    """Rotate x-y in [B,C,X,Y,Z], preserving z and the full padded field."""
    if pad_to_diagonal:
        x = _pad_dh_to_diagonal(x)
    if abs(float(angle_deg)) < 1e-6:
        return x
    angle = math.radians(float(angle_deg))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    theta = torch.zeros((x.shape[0], 3, 4), dtype=x.dtype, device=x.device)
    # affine_grid coordinates are ordered as grid x/y/z -> tensor W/H/D.
    # Keep W/z unchanged and rotate H/y with D/x.
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = -sin_a
    theta[:, 2, 1] = sin_a
    theta[:, 2, 2] = cos_a
    grid = F.affine_grid(theta, size=x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)


def multi_angle_projection_mu_loss(
    pred_mu: torch.Tensor,
    target_mu: torch.Tensor,
    mask: torch.Tensor | None,
    spacing_mm,
    angles_deg,
    angle_weights=None,
    downsample_factor: int = 2,
    ray_support_mask: torch.Tensor | None = None,
    tail_fraction: float = 0.05,
    error_domain: str = "log_acf",
    delta_clip: float = 0.5,
    return_components: bool = False,
):
    """Approximate oblique PET line-integral consistency with memory-first rotations.

    Angles rotate the D/H (x/y) plane around the z axis, then project along H/y,
    which leaves an x-z projection aligned with the topogram view at angle 0.
    """
    factor = max(1, int(downsample_factor))
    pred_mu = pred_mu.float()
    target_mu = target_mu.float()
    integration_mask = torch.ones_like(pred_mu) if mask is None else mask.float()
    support = ray_support_mask
    if support is None:
        support = integration_mask
    support = support.float()
    pred = _downsample_projection_tensor(pred_mu * integration_mask, factor)
    target = _downsample_projection_tensor(target_mu * integration_mask, factor)
    support = _downsample_projection_tensor(support, factor)
    _, sy, _ = [float(value) for value in spacing_mm]
    projection_spacing_cm = (sy / 10.0) * factor

    angles = [float(value) for value in angles_deg]
    if not angles:
        zero = pred_mu.new_tensor(0.0)
        return (zero, {}) if return_components else zero
    weights = _projection_weights(
        angle_weights if angle_weights is not None else [1.0] * len(angles),
        pred.dtype,
        pred.device,
        len(angles),
        "projection_angle_weights",
    )

    components = {}
    losses = []
    for angle in angles:
        pred_rot = _rotate_dh_plane(pred, angle)
        target_rot = _rotate_dh_plane(target, angle)
        support_rot = _rotate_dh_plane(support, angle)
        pred_projection = pred_rot.sum(dim=3) * projection_spacing_cm
        target_projection = target_rot.sum(dim=3) * projection_spacing_cm
        supported_rays = (support_rot.sum(dim=3) > EPS).to(dtype=pred.dtype)
        statistics = _supported_ray_error_statistics(
            pred_projection,
            target_projection,
            supported_rays,
            tail_fraction=tail_fraction,
            error_domain=error_domain,
            delta_clip=delta_clip,
        )
        component_name = _angle_component_name(angle)
        components[component_name] = statistics["absolute"]
        components[f"signed_{component_name}"] = statistics["signed"]
        components[f"mean_signed_{component_name}"] = statistics["signed_mean"]
        components[f"tail_{component_name}"] = statistics["tail"]
        components[f"p95_{component_name}"] = statistics["p95"]
        losses.append(statistics["absolute"])

    stacked = torch.stack(losses)
    total = torch.sum(stacked * weights) / weights.sum()
    if return_components:
        return total, components
    return total


def projection_mu_loss(
    pred_mu: torch.Tensor,
    target_mu: torch.Tensor,
    mask: torch.Tensor | None,
    spacing_mm,
    axis_weights=None,
    projection_cfg: dict | None = None,
    ray_support_mask: torch.Tensor | None = None,
    return_components: bool = False,
):
    projection_cfg = projection_cfg or {}
    orth_weight = float(projection_cfg.get("orth_weight", 1.0))
    multi_angle_weight = float(projection_cfg.get("multi_angle_weight", 0.0))
    orth_axis_weights = projection_cfg.get("axis_weights", axis_weights)
    tail_fraction = float(projection_cfg.get("tail_fraction", 0.05))
    error_domain = str(projection_cfg.get("error_domain", "log_acf"))
    delta_clip = float(projection_cfg.get("delta_clip", 0.5))

    orth_loss, orth_components = orthogonal_projection_mu_loss(
        pred_mu,
        target_mu,
        mask,
        spacing_mm=spacing_mm,
        axis_weights=orth_axis_weights,
        ray_support_mask=ray_support_mask,
        tail_fraction=tail_fraction,
        error_domain=error_domain,
        delta_clip=delta_clip,
        return_components=True,
    )

    components = dict(orth_components)
    components["orth"] = orth_loss
    orth_signed = torch.stack(
        [orth_components["signed_D"], orth_components["signed_H"], orth_components["signed_W"]]
    )
    orth_weights = _projection_weights(
        orth_axis_weights if orth_axis_weights is not None else (1.0, 1.0, 1.0),
        pred_mu.dtype,
        pred_mu.device,
        3,
        "projection_axis_weights",
    )
    signed_orth_loss = torch.sum(orth_signed * orth_weights) / orth_weights.sum()
    components["signed_orth"] = signed_orth_loss
    for statistic_name in ("mean_signed", "tail", "p95"):
        values = torch.stack(
            [orth_components[f"{statistic_name}_D"], orth_components[f"{statistic_name}_H"], orth_components[f"{statistic_name}_W"]]
        )
        components[f"{statistic_name}_orth"] = torch.sum(values * orth_weights) / orth_weights.sum()
    if multi_angle_weight > 0:
        multi_angle_loss, angle_components = multi_angle_projection_mu_loss(
            pred_mu,
            target_mu,
            mask,
            spacing_mm=spacing_mm,
            angles_deg=projection_cfg.get("angles_deg", []),
            angle_weights=projection_cfg.get("angle_weights"),
            downsample_factor=int(projection_cfg.get("downsample_factor", 2)),
            ray_support_mask=ray_support_mask,
            tail_fraction=tail_fraction,
            error_domain=error_domain,
            delta_clip=delta_clip,
            return_components=True,
        )
    else:
        multi_angle_loss, angle_components = pred_mu.new_tensor(0.0), {}
    components["multi_angle"] = multi_angle_loss
    components.update(angle_components)
    signed_angle_components = [
        angle_components[f"signed_{_angle_component_name(float(angle))}"]
        for angle in projection_cfg.get("angles_deg", [])
        if f"signed_{_angle_component_name(float(angle))}" in angle_components
    ]
    if signed_angle_components:
        angle_weights_tensor = _projection_weights(
            projection_cfg.get("angle_weights") or [1.0] * len(signed_angle_components),
            pred_mu.dtype,
            pred_mu.device,
            len(signed_angle_components),
            "projection_angle_weights",
        )
        signed_multi_angle_loss = (
            torch.sum(torch.stack(signed_angle_components) * angle_weights_tensor)
            / angle_weights_tensor.sum()
        )
    else:
        signed_multi_angle_loss = pred_mu.new_tensor(0.0)
    components["signed_multi_angle"] = signed_multi_angle_loss

    def aggregate_angle_statistic(statistic_name: str) -> torch.Tensor:
        values = [
            angle_components[f"{statistic_name}_{_angle_component_name(float(angle))}"]
            for angle in projection_cfg.get("angles_deg", [])
            if f"{statistic_name}_{_angle_component_name(float(angle))}" in angle_components
        ]
        if not values:
            return pred_mu.new_tensor(0.0, dtype=torch.float32)
        statistic_weights = _projection_weights(
            projection_cfg.get("angle_weights") or [1.0] * len(values),
            values[0].dtype,
            values[0].device,
            len(values),
            "projection_angle_weights",
        )
        return torch.sum(torch.stack(values) * statistic_weights) / statistic_weights.sum()

    for statistic_name in ("mean_signed", "tail", "p95"):
        components[f"{statistic_name}_multi_angle"] = aggregate_angle_statistic(statistic_name)

    total_weight = orth_weight + multi_angle_weight
    if total_weight <= 0:
        raise ValueError("At least one of projection.orth_weight or projection.multi_angle_weight must be positive")
    total = (orth_weight * orth_loss + multi_angle_weight * multi_angle_loss) / total_weight
    components["signed"] = (
        orth_weight * signed_orth_loss + multi_angle_weight * signed_multi_angle_loss
    ) / total_weight
    for statistic_name in ("mean_signed", "tail", "p95"):
        components[statistic_name] = (
            orth_weight * components[f"{statistic_name}_orth"]
            + multi_angle_weight * components[f"{statistic_name}_multi_angle"]
        ) / total_weight
    if return_components:
        return total, components
    return total


def superior_body_fraction_mask(
    body_mask: torch.Tensor,
    superior_fraction: float = 0.22,
    spatial_axis: int = 2,
) -> torch.Tensor:
    """Return the superior body fraction in [B,C,X,Y,Z] tensor space."""
    if body_mask.ndim != 5:
        raise ValueError(f"Expected [B,C,X,Y,Z] body mask, got {tuple(body_mask.shape)}")
    if spatial_axis not in (0, 1, 2):
        raise ValueError(f"spatial_axis must be 0, 1 or 2, got {spatial_axis}")
    fraction = float(superior_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"superior_fraction must be in (0, 1], got {fraction}")

    tensor_axis = spatial_axis + 2
    reduce_dims = tuple(dim for dim in range(1, 5) if dim != tensor_axis)
    occupied = (body_mask > 0).any(dim=reduce_dims)
    output = torch.zeros_like(body_mask, dtype=body_mask.dtype)
    for batch_index in range(body_mask.shape[0]):
        indices = torch.nonzero(occupied[batch_index], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        inferior = int(indices.min().item())
        superior = int(indices.max().item())
        body_length = superior - inferior + 1
        start = max(inferior, superior - int(math.ceil(body_length * fraction)) + 1)
        slices = [slice(None)] * 5
        slices[0] = batch_index
        slices[tensor_axis] = slice(start, superior + 1)
        output[tuple(slices)] = body_mask[tuple(slices)]
    return output


def organ_bias_loss(
    pred_mu: torch.Tensor,
    target_mu: torch.Tensor,
    organ_seg: torch.Tensor | None,
    mask: torch.Tensor,
    min_voxels: int = 64,
    max_labels: int = 64,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if organ_seg is None:
        zero = pred_mu.new_tensor(0.0)
        return zero, {}

    labels = torch.unique(organ_seg.detach())
    losses = []
    per_label = {}
    for label in labels:
        label_value = int(label.item())
        if label_value == 0:
            continue
        region = (organ_seg == label_value) & (mask > 0)
        if int(region.sum().item()) < min_voxels:
            continue
        pred_mean = pred_mu[region].mean()
        target_mean = target_mu[region].mean()
        loss = torch.abs(pred_mean - target_mean)
        losses.append(loss)
        if len(per_label) < max_labels:
            per_label[f"organ_bias_label_{label_value}"] = loss.detach()

    if not losses:
        zero = pred_mu.new_tensor(0.0)
        return zero, per_label
    return torch.stack(losses).mean(), per_label


def regional_projection_loss(
    pred_mu: torch.Tensor,
    target_mu: torch.Tensor,
    prediction_mask: torch.Tensor,
    region_ids: torch.Tensor | None,
    spacing_mm,
    projection_cfg: dict,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Average one projection loss per represented body region.

    Equal averaging prevents a repeatedly sampled thorax/abdomen region from
    silently dominating lower-body or head calibration.
    """
    if region_ids is None:
        region_ids = torch.arange(pred_mu.shape[0], device=pred_mu.device)
    region_ids = region_ids.reshape(-1).to(device=pred_mu.device)
    if int(region_ids.numel()) != int(pred_mu.shape[0]):
        raise ValueError(
            f"region_ids batch mismatch: {region_ids.numel()} != {pred_mu.shape[0]}"
        )

    region_losses = []
    diagnostics: Dict[str, torch.Tensor] = {}
    for region_id in torch.unique(region_ids.detach(), sorted=True):
        selected = region_ids == region_id
        if not bool(selected.any()):
            continue
        region_loss, components = projection_mu_loss(
            pred_mu[selected],
            target_mu[selected],
            None,
            spacing_mm=spacing_mm,
            axis_weights=projection_cfg.get("axis_weights", [1.0, 1.0, 0.0]),
            projection_cfg=projection_cfg,
            ray_support_mask=prediction_mask[selected],
            return_components=True,
        )
        region_losses.append(region_loss)
        diagnostics[f"region_{int(region_id.item())}"] = region_loss.detach()
        diagnostics[f"region_{int(region_id.item())}_signed"] = components["signed"].detach()

    if not region_losses:
        return pred_mu.new_tensor(0.0), diagnostics
    return torch.stack(region_losses).mean(), diagnostics


def organ_ray_projection_loss(
    pred_mu: torch.Tensor,
    target_mu: torch.Tensor,
    organ_seg: torch.Tensor | None,
    prediction_mask: torch.Tensor,
    spacing_mm,
    projection_cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Measure full attenuation paths selected by rays intersecting each organ.

    The attenuation integral uses the complete effective mu-map. The organ mask
    only selects rays, so this does not make the common mistake of integrating
    attenuation inside the organ alone.
    """
    zero = pred_mu.new_tensor(0.0)
    if organ_seg is None:
        return zero, zero, {}

    min_voxels = int(projection_cfg.get("min_voxels", 64))
    max_labels = int(projection_cfg.get("max_labels", 16))
    labels = []
    for label in torch.unique(organ_seg.detach()):
        label_value = int(label.item())
        if label_value == 0:
            continue
        support = (organ_seg == label_value) & (prediction_mask > 0)
        if int(support.sum().item()) >= min_voxels:
            labels.append((label_value, support.float()))
        if len(labels) >= max_labels:
            break

    absolute_losses = []
    signed_losses = []
    diagnostics: Dict[str, torch.Tensor] = {}
    for label_value, support in labels:
        absolute, components = projection_mu_loss(
            pred_mu,
            target_mu,
            None,
            spacing_mm=spacing_mm,
            axis_weights=projection_cfg.get("axis_weights", [1.0, 1.0, 0.0]),
            projection_cfg=projection_cfg,
            ray_support_mask=support,
            return_components=True,
        )
        signed = components["signed"]
        absolute_losses.append(absolute)
        signed_losses.append(signed)
        diagnostics[f"organ_ray_label_{label_value}"] = absolute.detach()
        diagnostics[f"organ_ray_signed_label_{label_value}"] = signed.detach()

    if not absolute_losses:
        return zero, zero, diagnostics
    return (
        torch.stack(absolute_losses).mean(),
        torch.stack(signed_losses).mean(),
        diagnostics,
    )


class CombinedPseudoCTLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        loss_cfg = config.get("loss", config)
        self.loss_cfg = loss_cfg
        self.spacing_mm = loss_cfg.get("spacing_mm", [1.5234, 1.5234, 2.0])
        self.projection_axis_weights = loss_cfg.get("projection_axis_weights", [1.0, 1.0, 1.0])
        self.projection_cfg = dict(loss_cfg.get("projection", {}))
        self.global_projection_cfg = dict(loss_cfg.get("global_projection", {}))
        self.head_projection_cfg = dict(loss_cfg.get("head_projection", {}))
        self.local_head_projection_cfg = dict(
            loss_cfg.get("local_head_projection", self.head_projection_cfg)
        )
        self.regional_projection_cfg = dict(loss_cfg.get("regional_projection", {}))
        self.organ_ray_projection_cfg = dict(loss_cfg.get("organ_ray_projection", {}))
        self.pet_evaluator_proxy_cfg = dict(loss_cfg.get("pet_evaluator_proxy", {}))
        self.local_head_region_id = None
        local_head_region_name = str(
            self.local_head_projection_cfg.get("region_name", "head_neck")
        )
        regions = config.get("training", {}).get("region_sampling", {}).get("regions", [])
        for region_id, region in enumerate(regions):
            if str(region.get("name", "")) == local_head_region_name:
                self.local_head_region_id = int(region_id)
                break
        if self.local_head_projection_cfg.get("enabled", False):
            if self.local_head_region_id is None:
                raise ValueError(
                    "local_head_projection.enabled=true requires a matching "
                    f"training.region_sampling region named '{local_head_region_name}'"
                )
            sampling_axis = int(
                config.get("training", {}).get("region_sampling", {}).get("axis", 2)
            )
            projection_axis = int(self.local_head_projection_cfg.get("spatial_axis", 2))
            if sampling_axis != projection_axis:
                raise ValueError(
                    "local_head_projection.spatial_axis must match training.region_sampling.axis "
                    f"({projection_axis} != {sampling_axis})"
                )
        self.global_reuse_per_subject = bool(
            config.get("model", {}).get("global_context", {}).get("reuse_per_subject", True)
        )
        self.ct_clip_min = float(loss_cfg.get("ct_clip_min", -1000.0))
        self.ct_clip_max = float(loss_cfg.get("ct_clip_max", 2000.0))
        self.fwhm_mm = float(loss_cfg.get("mu_fwhm_mm", 4.0))
        self.normalize_by_initial_value = bool(loss_cfg.get("normalize_by_initial_value", False))
        self.warmup_steps_for_reference = int(loss_cfg.get("warmup_steps_for_reference", 100))
        self._step = 0
        self._ref_sums: Dict[str, float] = {}
        self._ref_counts: Dict[str, int] = {}
        self.active_terms = self._ever_active_terms()

    def _phase_lambdas(self, epoch: int) -> Dict[str, float]:
        phases = self.loss_cfg.get("phases")
        if phases:
            active = phases[0]
            for phase in phases:
                if epoch >= int(phase.get("start_epoch", 0)):
                    active = phase
            return {
                "hu": float(active.get("lambda_hu", 1.0)),
                "grad": float(active.get("lambda_grad", 0.05)),
                "mu_raw": float(active.get("lambda_mu_raw", 1.0)),
                "mu": float(active.get("lambda_mu", 0.5)),
                "proj": float(active.get("lambda_proj", 0.0)),
                "organ": float(active.get("lambda_organ", 0.0)),
                "global_mu_raw": float(active.get("lambda_global_mu_raw", 0.0)),
                "global_mu": float(active.get("lambda_global_mu", 0.0)),
                "global_acf": float(active.get("lambda_global_acf", 0.0)),
                "global_acf_signed": float(active.get("lambda_global_acf_signed", 0.0)),
                "regional_relacf": float(active.get("lambda_regional_relacf", 0.0)),
                "organ_ray_signed": float(active.get("lambda_organ_ray_signed", 0.0)),
                "local_head_acf": float(active.get("lambda_local_head_acf", 0.0)),
                "local_head_acf_signed": float(active.get("lambda_local_head_acf_signed", 0.0)),
                "local_head_acf_tail": float(active.get("lambda_local_head_acf_tail", 0.0)),
                "pet_whole_body": float(active.get("lambda_pet_whole_body", 0.0)),
                "pet_organ_bias": float(active.get("lambda_pet_organ_bias", 0.0)),
                "pet_brain_outlier": float(active.get("lambda_pet_brain_outlier", 0.0)),
            }

        return {
            "hu": float(self.loss_cfg.get("lambda_hu", 1.0)),
            "grad": float(self.loss_cfg.get("lambda_grad", 0.05)),
            "mu_raw": float(self.loss_cfg.get("lambda_mu_raw", 1.0)),
            "mu": float(self.loss_cfg.get("lambda_mu", 0.5)),
            "proj": float(self.loss_cfg.get("lambda_proj", 0.0)),
            "organ": float(self.loss_cfg.get("lambda_organ", 0.0)),
            "global_mu_raw": float(self.loss_cfg.get("lambda_global_mu_raw", 0.0)),
            "global_mu": float(self.loss_cfg.get("lambda_global_mu", 0.0)),
            "global_acf": float(self.loss_cfg.get("lambda_global_acf", 0.0)),
            "global_acf_signed": float(self.loss_cfg.get("lambda_global_acf_signed", 0.0)),
            "regional_relacf": float(self.loss_cfg.get("lambda_regional_relacf", 0.0)),
            "organ_ray_signed": float(self.loss_cfg.get("lambda_organ_ray_signed", 0.0)),
            "local_head_acf": float(self.loss_cfg.get("lambda_local_head_acf", 0.0)),
            "local_head_acf_signed": float(self.loss_cfg.get("lambda_local_head_acf_signed", 0.0)),
            "local_head_acf_tail": float(self.loss_cfg.get("lambda_local_head_acf_tail", 0.0)),
            "pet_whole_body": float(self.loss_cfg.get("lambda_pet_whole_body", 0.0)),
            "pet_organ_bias": float(self.loss_cfg.get("lambda_pet_organ_bias", 0.0)),
            "pet_brain_outlier": float(self.loss_cfg.get("lambda_pet_brain_outlier", 0.0)),
        }

    def _ever_active_terms(self) -> Dict[str, bool]:
        phases = self.loss_cfg.get("phases")
        if phases:
            lambda_sets = [
                self._phase_lambdas(int(phase.get("start_epoch", 0)))
                for phase in phases
            ]
        else:
            lambda_sets = [self._phase_lambdas(0)]
        return {
            name: any(abs(values[name]) > 0.0 for values in lambda_sets)
            for name in lambda_sets[0]
        }

    def _maybe_normalize(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if not self.normalize_by_initial_value:
            return value

        value_float = float(value.detach().item())
        if self.training and self._step <= self.warmup_steps_for_reference and math.isfinite(value_float) and value_float > EPS:
            self._ref_sums[name] = self._ref_sums.get(name, 0.0) + value_float
            self._ref_counts[name] = self._ref_counts.get(name, 0) + 1

        return value / self._reference_value(name, value)

    def _reference_value(self, name: str, value: torch.Tensor) -> float:
        if not self.normalize_by_initial_value:
            return 1.0
        count = self._ref_counts.get(name, 0)
        if count == 0:
            return max(float(value.detach().item()), EPS)
        return max(self._ref_sums[name] / count, EPS)

    @staticmethod
    def _batch_spacing(batch: dict, fallback) -> list[float]:
        spacing = batch.get("global_spacing_mm")
        if spacing is None:
            return [float(value) for value in fallback]
        spacing = torch.as_tensor(spacing).detach().float()
        if spacing.ndim > 1:
            reference = spacing[0]
            if not torch.allclose(spacing, reference.unsqueeze(0).expand_as(spacing), atol=1e-4, rtol=1e-4):
                raise ValueError("A batch contains different global spacings; use batch_size=1 for global loss")
            spacing = reference
        return [float(value.item()) for value in spacing.flatten()[:3]]

    @staticmethod
    def _batch_metadata_tensor(value, batch_size: int, width: int, device, dtype) -> torch.Tensor | None:
        if value is None:
            return None
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if width == 1:
            tensor = tensor.reshape(-1)
            if tensor.numel() != batch_size:
                return None
            return tensor
        tensor = tensor.reshape(batch_size, -1)
        if tensor.shape[1] < width:
            return None
        return tensor[:, :width]

    def _local_head_mask(
        self,
        batch: dict,
        reference: torch.Tensor,
        body_mask: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the true superior-body ROI inside mandatory head/neck crops.

        Crop coordinates and body extents remain in the original subject grid.
        This avoids interpreting the superior fraction relative to an already
        cropped patch, which would select only the top of the skull.
        """
        zero_mask = torch.zeros_like(reference)
        zero = reference.new_tensor(0.0)
        batch_size = int(reference.shape[0])
        if not self.local_head_projection_cfg.get("enabled", False):
            return zero_mask, zero, zero, zero
        if self.local_head_region_id is None:
            return zero_mask, zero, zero, zero

        region_ids = self._batch_metadata_tensor(
            batch.get("train_crop_region_id"), batch_size, 1, reference.device, torch.long
        )
        crop_start = self._batch_metadata_tensor(
            batch.get("train_crop_start"), batch_size, 3, reference.device, reference.dtype
        )
        body_axis_min = self._batch_metadata_tensor(
            batch.get("train_body_axis_min"), batch_size, 1, reference.device, reference.dtype
        )
        body_axis_max = self._batch_metadata_tensor(
            batch.get("train_body_axis_max"), batch_size, 1, reference.device, reference.dtype
        )
        if region_ids is None or crop_start is None or body_axis_min is None or body_axis_max is None:
            return zero_mask, zero, zero, zero

        selected = region_ids == int(self.local_head_region_id)
        if not bool(selected.any()):
            return zero_mask, zero, zero, zero

        spatial_axis = int(self.local_head_projection_cfg.get("spatial_axis", 2))
        if spatial_axis not in (0, 1, 2):
            raise ValueError(f"local_head_projection.spatial_axis must be 0, 1 or 2, got {spatial_axis}")
        superior_fraction = float(self.local_head_projection_cfg.get("superior_fraction", 0.22))
        if not 0.0 < superior_fraction <= 1.0:
            raise ValueError(
                f"local_head_projection.superior_fraction must be in (0, 1], got {superior_fraction}"
            )

        body_length = body_axis_max - body_axis_min + 1.0
        head_start = torch.maximum(
            body_axis_min,
            body_axis_max - torch.ceil(body_length * superior_fraction) + 1.0,
        )
        local_size = int(reference.shape[spatial_axis + 2])
        global_indices = (
            crop_start[:, spatial_axis].unsqueeze(1)
            + torch.arange(local_size, device=reference.device, dtype=reference.dtype).unsqueeze(0)
        )
        axis_mask = (global_indices >= head_start.unsqueeze(1)) & (
            global_indices <= body_axis_max.unsqueeze(1)
        )
        view_shape = [batch_size, 1, 1, 1, 1]
        view_shape[spatial_axis + 2] = local_size
        spatial_mask = axis_mask.reshape(view_shape).to(dtype=reference.dtype)
        sample_mask = selected.reshape(batch_size, 1, 1, 1, 1).to(dtype=reference.dtype)
        local_head = spatial_mask * sample_mask * body_mask * prediction_mask

        coverage = self._batch_metadata_tensor(
            batch.get("train_crop_region_transaxial_coverage"),
            batch_size,
            1,
            reference.device,
            reference.dtype,
        )
        selected_coverage = zero
        if coverage is not None:
            finite = torch.isfinite(coverage) & selected
            if bool(finite.any()):
                selected_coverage = coverage[finite].mean()

        sample_rate = selected.to(dtype=reference.dtype).mean()
        voxel_fraction = local_head.sum() / ((body_mask * prediction_mask).sum() + EPS)
        return local_head, sample_rate, voxel_fraction, selected_coverage

    def forward(
        self,
        pred_ct_hu: torch.Tensor,
        batch: dict,
        epoch: int = 0,
        aux_outputs: dict[str, torch.Tensor] | None = None,
    ):
        if self.training:
            self._step += 1

        pred_hu_unclipped = pred_ct_hu.float()
        target_hu_unclipped = batch["ct"].float()
        pred_hu = pred_hu_unclipped.clamp(self.ct_clip_min, self.ct_clip_max)
        target_hu = target_hu_unclipped.clamp(self.ct_clip_min, self.ct_clip_max)
        lambdas = self._phase_lambdas(epoch)
        zero = pred_hu.sum() * 0.0
        prediction_mask = (batch.get("prediction_mask", torch.ones_like(target_hu)) > 0).float()
        body_mask = (batch.get("body_seg", prediction_mask) > 0).float()
        mask = prediction_mask * body_mask
        official_mask = (batch.get("official_eval_mask", body_mask) > 0).float()

        abs_hu = torch.abs(pred_hu - target_hu)
        loss_hu = zero
        if self.active_terms["hu"]:
            tissue_weight = build_tissue_weight(
                target_hu,
                body_mask=body_mask,
                air_threshold=float(self.loss_cfg.get("air_threshold", -300.0)),
                bone_threshold=float(self.loss_cfg.get("bone_threshold", 150.0)),
                air_weight=float(
                    self.loss_cfg.get("air_weight", self.loss_cfg.get("lung_weight", 1.5))
                ),
                bone_weight=float(self.loss_cfg.get("bone_weight", 2.0)),
                soft_weight=float(self.loss_cfg.get("soft_weight", 1.0)),
            )
            loss_hu = masked_weighted_mean(abs_hu, mask, tissue_weight)

        pred_mu = hu_to_mu(pred_hu)
        # The official CT metric compares submitted CT directly against GT in
        # the body/liver-exclusion mask. PET reconstruction has different
        # semantics: it replaces voxels outside prediction_mask before
        # smoothing and projection.
        target_mu_official = hu_to_mu(target_hu_unclipped)
        loss_mu_raw = masked_mean(
            torch.abs(pred_mu - target_mu_official),
            official_mask,
        )
        pred_mu_effective = (
            pred_mu * prediction_mask + target_mu_official * (1.0 - prediction_mask)
        )
        pred_mu_s = gaussian_smooth_3d(
            pred_mu_effective,
            self.spacing_mm,
            self.fwhm_mm,
            padding_mode=OFFICIAL_GAUSSIAN_MODE,
        )
        target_mu_s = gaussian_smooth_3d(
            target_mu_official,
            self.spacing_mm,
            self.fwhm_mm,
            padding_mode=OFFICIAL_GAUSSIAN_MODE,
        )
        loss_mu = masked_mean(torch.abs(pred_mu_s - target_mu_s), mask) / MU_WATER

        loss_grad = zero
        if self.active_terms["grad"]:
            pdz, pdy, pdx = gradient_3d(pred_hu)
            tdz, tdy, tdx = gradient_3d(target_hu)
            mz = mask[:, :, 1:, :, :] * mask[:, :, :-1, :, :]
            my = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]
            mx = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]
            loss_grad = (
                masked_mean(torch.abs(pdz - tdz), mz)
                + masked_mean(torch.abs(pdy - tdy), my)
                + masked_mean(torch.abs(pdx - tdx), mx)
            ) / 3.0

        projection_components = {
            name: zero
            for name in (
                "D",
                "H",
                "W",
                "orth",
                "multi_angle",
                "signed",
                "mean_signed",
                "tail",
                "p95",
            )
        }
        loss_proj = zero
        if self.active_terms["proj"]:
            loss_proj, projection_components = projection_mu_loss(
                pred_mu_s,
                target_mu_s,
                None,
                spacing_mm=self.spacing_mm,
                axis_weights=self.projection_axis_weights,
                projection_cfg=self.projection_cfg,
                ray_support_mask=prediction_mask,
                return_components=True,
            )
        organ_seg = batch.get("organ_seg")
        if organ_seg is not None:
            organ_seg = organ_seg.long()
        loss_organ = zero
        per_organ: Dict[int, torch.Tensor] = {}
        if self.active_terms["organ"]:
            loss_organ, per_organ = organ_bias_loss(
                pred_mu_s,
                target_mu_s,
                organ_seg,
                mask,
            )

        loss_regional_relacf = zero
        regional_components: Dict[str, torch.Tensor] = {}
        if (
            self.active_terms["regional_relacf"]
            and self.regional_projection_cfg.get("enabled", False)
        ):
            region_ids = self._batch_metadata_tensor(
                batch.get("train_crop_region_id"),
                int(pred_mu_s.shape[0]),
                1,
                pred_mu_s.device,
                torch.long,
            )
            loss_regional_relacf, regional_components = regional_projection_loss(
                pred_mu_s,
                target_mu_s,
                prediction_mask,
                region_ids,
                spacing_mm=self.spacing_mm,
                projection_cfg=self.regional_projection_cfg,
            )

        loss_organ_ray_abs = zero
        loss_organ_ray_signed = zero
        organ_ray_components: Dict[str, torch.Tensor] = {}
        if (
            self.active_terms["organ_ray_signed"]
            and self.organ_ray_projection_cfg.get("enabled", False)
        ):
            (
                loss_organ_ray_abs,
                loss_organ_ray_signed,
                organ_ray_components,
            ) = organ_ray_projection_loss(
                pred_mu_s,
                target_mu_s,
                organ_seg,
                prediction_mask,
                spacing_mm=self.spacing_mm,
                projection_cfg=self.organ_ray_projection_cfg,
            )

        loss_local_head_acf = zero
        loss_local_head_acf_signed = zero
        loss_local_head_acf_tail = zero
        local_head_components = {
            name: zero
            for name in ("orth", "multi_angle", "signed", "mean_signed", "tail", "p95")
        }
        local_head_mask = torch.zeros_like(pred_mu_s)
        local_head_sample_rate = zero
        local_head_voxel_fraction = zero
        local_head_coverage = zero
        local_head_active = any(
            self.active_terms[name]
            for name in (
                "local_head_acf",
                "local_head_acf_signed",
                "local_head_acf_tail",
            )
        )
        if local_head_active:
            (
                local_head_mask,
                local_head_sample_rate,
                local_head_voxel_fraction,
                local_head_coverage,
            ) = self._local_head_mask(
                batch,
                pred_mu_s,
                body_mask,
                prediction_mask,
            )
        if local_head_active and float(local_head_mask.detach().sum().item()) > EPS:
            active_head_samples = local_head_mask.flatten(1).sum(dim=1) > EPS
            active_local_head_mask = local_head_mask[active_head_samples]
            loss_local_head_acf, local_head_components = projection_mu_loss(
                pred_mu_s[active_head_samples],
                target_mu_s[active_head_samples],
                active_local_head_mask,
                spacing_mm=self.spacing_mm,
                axis_weights=self.local_head_projection_cfg.get("axis_weights", [1.0, 1.0, 0.0]),
                projection_cfg=self.local_head_projection_cfg,
                ray_support_mask=active_local_head_mask,
                return_components=True,
            )
            loss_local_head_acf_signed = local_head_components["signed"]
            loss_local_head_acf_tail = local_head_components["tail"]

        loss_pet_whole_body = zero
        loss_pet_organ_bias = zero
        loss_pet_brain_outlier = zero
        pet_proxy_diagnostics: Dict[str, torch.Tensor] = {}
        pet_proxy_active = any(
            self.active_terms[name]
            for name in ("pet_whole_body", "pet_organ_bias", "pet_brain_outlier")
        )
        if pet_proxy_active and self.pet_evaluator_proxy_cfg.get("enabled", False):
            proxy_mode = str(
                self.pet_evaluator_proxy_cfg.get("training_mode", "local_patch")
            ).lower()
            if proxy_mode == "full_fov_patch_injected":
                required_global_keys = (
                    "global_official_eval_mask",
                    "global_prediction_mask",
                    "global_target_hu",
                    "global_spacing_mm",
                )
                missing_global = [key for key in required_global_keys if batch.get(key) is None]
                crop_starts = batch.get("train_crop_start", batch.get("val_crop_start"))
                original_shapes = batch.get("crop_original_shape")
                if crop_starts is None:
                    missing_global.append("train_crop_start/val_crop_start")
                if original_shapes is None:
                    missing_global.append("crop_original_shape")
                if missing_global:
                    raise RuntimeError(
                        "Full-FOV patch-injected PET proxy is missing: "
                        + ", ".join(missing_global)
                    )

                # Delta smoothing uses zero padding because unsampled voxels are
                # explicitly assumed to have zero attenuation error. This avoids
                # reflecting a patch-edge error into the unobserved full FOV.
                patch_delta_mu_s = gaussian_smooth_3d(
                    pred_mu_effective - target_mu_official,
                    self.spacing_mm,
                    self.fwhm_mm,
                    padding_mode="constant",
                )
                global_organ_seg = batch.get("global_organ_seg")
                if global_organ_seg is not None:
                    global_organ_seg = global_organ_seg.long()
                global_body_region_seg = batch.get("global_body_region_seg")
                if global_body_region_seg is not None:
                    global_body_region_seg = global_body_region_seg.long()
                pet_proxy_losses, pet_proxy_diagnostics = (
                    evaluator_aligned_full_fov_patch_proxy_losses(
                        patch_delta_mu=patch_delta_mu_s,
                        crop_starts=crop_starts,
                        original_shapes=original_shapes,
                        global_official_eval_mask=batch["global_official_eval_mask"],
                        global_prediction_mask=batch["global_prediction_mask"],
                        global_organ_seg=global_organ_seg,
                        global_body_region_seg=global_body_region_seg,
                        global_target_hu=batch["global_target_hu"],
                        global_activity_proxy=batch.get("global_activity_proxy"),
                        global_spacing_mm=batch["global_spacing_mm"],
                        config=self.pet_evaluator_proxy_cfg,
                        global_pet_eval_mask=batch.get("global_pet_eval_mask"),
                        global_pet_region_masks=batch.get("global_pet_region_masks"),
                        global_activity_is_gt_pet=batch.get(
                            "global_activity_is_gt_pet"
                        ),
                    )
                )
            elif proxy_mode == "local_patch":
                body_region_seg = batch.get("body_region_seg")
                if body_region_seg is not None:
                    body_region_seg = body_region_seg.long()
                pet_proxy_losses, pet_proxy_diagnostics = evaluator_aligned_pet_proxy_losses(
                    pred_mu_s,
                    target_mu_s,
                    official_eval_mask=official_mask,
                    prediction_mask=prediction_mask,
                    organ_seg=organ_seg,
                    body_region_seg=body_region_seg,
                    target_hu=target_hu,
                    activity_proxy=batch.get("activity_proxy"),
                    spacing_mm=self.spacing_mm,
                    config=self.pet_evaluator_proxy_cfg,
                )
            else:
                raise ValueError(
                    f"Unsupported pet_evaluator_proxy.training_mode={proxy_mode!r}"
                )
            loss_pet_whole_body = pet_proxy_losses["pet_whole_body"]
            loss_pet_organ_bias = pet_proxy_losses["pet_organ_bias"]
            loss_pet_brain_outlier = pet_proxy_losses["pet_brain_outlier"]

        loss_global_mu_raw = zero
        loss_global_mu = zero
        loss_global_acf = zero
        loss_global_acf_signed = zero
        global_components = {
            name: zero
            for name in (
                "D",
                "H",
                "W",
                "orth",
                "multi_angle",
                "signed",
                "mean_signed",
                "tail",
                "p95",
            )
        }
        global_pred_hu = None if aux_outputs is None else aux_outputs.get("global_ct")
        global_target_mu_raw = batch.get("global_target_mu_raw")
        global_target_mu_s = batch.get("global_target_mu_smooth")
        global_loss_active = any(
            self.active_terms[name]
            for name in ("global_mu_raw", "global_mu", "global_acf", "global_acf_signed")
        )
        if (
            global_loss_active
            and global_pred_hu is not None
            and (global_target_mu_raw is None or global_target_mu_s is None)
        ):
            raise RuntimeError(
                "Global model output is enabled but exact global mu targets are missing. "
                "Set model.global_context.supervise_auxiliary=true and provide the "
                "corresponding whole-body physics targets."
            )
        if global_loss_active and global_pred_hu is not None and global_target_mu_raw is not None:
            # DataLoader batch_size=1 plus train_num_samples>1 repeats one
            # subject-level target for every local patch. Match the single
            # deduplicated global prediction instead of materializing copies.
            if self.global_reuse_per_subject and global_pred_hu.shape[0] == 1:
                global_target_mu_raw = global_target_mu_raw[:1]
                global_target_mu_s = global_target_mu_s[:1]
            global_pred_hu = global_pred_hu.float().clamp(self.ct_clip_min, self.ct_clip_max)
            global_target_mu_raw = global_target_mu_raw.float()
            global_target_mu_s = global_target_mu_s.float()
            global_body_source = batch.get("global_body_mask")
            global_prediction_source = batch.get("global_prediction_mask")
            global_official_source = batch.get("global_official_eval_mask")
            if self.global_reuse_per_subject and global_pred_hu.shape[0] == 1:
                if global_body_source is not None:
                    global_body_source = global_body_source[:1]
                if global_prediction_source is not None:
                    global_prediction_source = global_prediction_source[:1]
                if global_official_source is not None:
                    global_official_source = global_official_source[:1]
            global_body = torch.ones_like(global_target_mu_raw) if global_body_source is None else global_body_source
            global_body = global_body.float().clamp(0.0, 1.0)
            global_prediction = (
                global_body if global_prediction_source is None else global_prediction_source.float().clamp(0.0, 1.0)
            )
            global_valid = global_body * global_prediction
            global_official = (
                global_body if global_official_source is None else global_official_source.float().clamp(0.0, 1.0)
            )
            global_spacing = self._batch_spacing(batch, self.spacing_mm)

            global_pred_mu = hu_to_mu(global_pred_hu)
            global_pred_mu_effective = (
                global_pred_mu * global_prediction
                + global_target_mu_raw * (1.0 - global_prediction)
            )
            loss_global_mu_raw = masked_mean(
                torch.abs(global_pred_mu - global_target_mu_raw),
                global_official,
            )
            global_pred_mu_s = gaussian_smooth_3d(
                global_pred_mu_effective,
                global_spacing,
                self.fwhm_mm,
                padding_mode=OFFICIAL_GAUSSIAN_MODE,
                truncate=OFFICIAL_GAUSSIAN_TRUNCATE,
            )
            loss_global_mu = masked_mean(
                torch.abs(global_pred_mu_s - global_target_mu_s),
                global_valid,
            ) / MU_WATER
            loss_global_acf, global_components = projection_mu_loss(
                global_pred_mu_s,
                global_target_mu_s,
                None,
                spacing_mm=global_spacing,
                axis_weights=self.global_projection_cfg.get("axis_weights", [1.0, 1.0, 1.0]),
                projection_cfg=self.global_projection_cfg,
                ray_support_mask=global_prediction,
                return_components=True,
            )
            loss_global_acf_signed = global_components["signed"]

        raw_losses = {
            "hu": loss_hu,
            "grad": loss_grad,
            "mu_raw": loss_mu_raw,
            "mu": loss_mu,
            "proj": loss_proj,
            "organ": loss_organ,
            "global_mu_raw": loss_global_mu_raw,
            "global_mu": loss_global_mu,
            "global_acf": loss_global_acf,
            "global_acf_signed": loss_global_acf_signed,
            "regional_relacf": loss_regional_relacf,
            "organ_ray_signed": loss_organ_ray_signed,
            "local_head_acf": loss_local_head_acf,
            "local_head_acf_signed": loss_local_head_acf_signed,
            "local_head_acf_tail": loss_local_head_acf_tail,
            "pet_whole_body": loss_pet_whole_body,
            "pet_organ_bias": loss_pet_organ_bias,
            "pet_brain_outlier": loss_pet_brain_outlier,
        }
        normalized = {
            name: self._maybe_normalize(name, value)
            if self.active_terms[name]
            else zero
            for name, value in raw_losses.items()
        }
        refs = {
            name: self._reference_value(name, value)
            if self.active_terms[name]
            else 0.0
            for name, value in raw_losses.items()
        }
        weighted = {name: normalized[name] * lambdas[name] for name in raw_losses}
        total = sum(weighted.values())

        bone_mask = ((target_hu > float(self.loss_cfg.get("bone_threshold", 150.0))) & (mask > 0)).float()
        lung_mask = ((target_hu < float(self.loss_cfg.get("air_threshold", -300.0))) & (mask > 0)).float()
        air_threshold = float(self.loss_cfg.get("air_threshold", -300.0))
        bone_threshold = float(self.loss_cfg.get("bone_threshold", 150.0))
        soft_mask = ((target_hu >= air_threshold) & (target_hu <= bone_threshold) & (mask > 0)).float()

        organ_metric = loss_organ.detach() if per_organ else pred_hu.new_tensor(float("nan"))
        has_local_head = float(local_head_sample_rate.detach().item()) > 0.0
        local_head_metric = (
            loss_local_head_acf.detach() if has_local_head else pred_hu.new_tensor(float("nan"))
        )
        local_head_signed_metric = (
            loss_local_head_acf_signed.detach()
            if has_local_head
            else pred_hu.new_tensor(float("nan"))
        )
        loss_dict = {
            "loss_total": total.detach(),
            "loss_hu_raw": loss_hu.detach(),
            "loss_grad_raw": loss_grad.detach(),
            "loss_mu_official_raw": loss_mu_raw.detach(),
            "loss_mu_raw": loss_mu.detach(),
            "loss_proj_raw": loss_proj.detach(),
            "loss_organ_raw": loss_organ.detach(),
            "loss_global_mu_official_raw": loss_global_mu_raw.detach(),
            "loss_global_mu_raw": loss_global_mu.detach(),
            "loss_global_acf_raw": loss_global_acf.detach(),
            "loss_global_acf_signed_raw": loss_global_acf_signed.detach(),
            "loss_regional_relacf_raw": loss_regional_relacf.detach(),
            "loss_organ_ray_relacf_raw": loss_organ_ray_abs.detach(),
            "loss_organ_ray_signed_raw": loss_organ_ray_signed.detach(),
            "loss_local_head_acf_raw": loss_local_head_acf.detach(),
            "loss_local_head_acf_signed_raw": loss_local_head_acf_signed.detach(),
            "loss_local_head_acf_tail_raw": loss_local_head_acf_tail.detach(),
            "loss_pet_whole_body_raw": loss_pet_whole_body.detach(),
            "loss_pet_organ_bias_raw": loss_pet_organ_bias.detach(),
            "loss_pet_brain_outlier_raw": loss_pet_brain_outlier.detach(),
            "loss_hu_weighted": weighted["hu"].detach(),
            "loss_grad_weighted": weighted["grad"].detach(),
            "loss_mu_official_weighted": weighted["mu_raw"].detach(),
            "loss_mu_weighted": weighted["mu"].detach(),
            "loss_proj_weighted": weighted["proj"].detach(),
            "loss_organ_weighted": weighted["organ"].detach(),
            "loss_global_mu_official_weighted": weighted["global_mu_raw"].detach(),
            "loss_global_mu_weighted": weighted["global_mu"].detach(),
            "loss_global_acf_weighted": weighted["global_acf"].detach(),
            "loss_global_acf_signed_weighted": weighted["global_acf_signed"].detach(),
            "loss_regional_relacf_weighted": weighted["regional_relacf"].detach(),
            "loss_organ_ray_signed_weighted": weighted["organ_ray_signed"].detach(),
            "loss_local_head_acf_weighted": weighted["local_head_acf"].detach(),
            "loss_local_head_acf_signed_weighted": weighted["local_head_acf_signed"].detach(),
            "loss_local_head_acf_tail_weighted": weighted["local_head_acf_tail"].detach(),
            "loss_pet_whole_body_weighted": weighted["pet_whole_body"].detach(),
            "loss_pet_organ_bias_weighted": weighted["pet_organ_bias"].detach(),
            "loss_pet_brain_outlier_weighted": weighted["pet_brain_outlier"].detach(),
            "loss_ref_hu": pred_hu.new_tensor(refs["hu"]),
            "loss_ref_grad": pred_hu.new_tensor(refs["grad"]),
            "loss_ref_mu_official": pred_hu.new_tensor(refs["mu_raw"]),
            "loss_ref_mu": pred_hu.new_tensor(refs["mu"]),
            "loss_ref_proj": pred_hu.new_tensor(refs["proj"]),
            "loss_ref_organ": pred_hu.new_tensor(refs["organ"]),
            "loss_ref_global_mu_official": pred_hu.new_tensor(refs["global_mu_raw"]),
            "loss_ref_global_mu": pred_hu.new_tensor(refs["global_mu"]),
            "loss_ref_global_acf": pred_hu.new_tensor(refs["global_acf"]),
            "loss_ref_global_acf_signed": pred_hu.new_tensor(refs["global_acf_signed"]),
            "loss_ref_regional_relacf": pred_hu.new_tensor(refs["regional_relacf"]),
            "loss_ref_organ_ray_signed": pred_hu.new_tensor(refs["organ_ray_signed"]),
            "loss_ref_local_head_acf": pred_hu.new_tensor(refs["local_head_acf"]),
            "loss_ref_local_head_acf_signed": pred_hu.new_tensor(refs["local_head_acf_signed"]),
            "loss_ref_local_head_acf_tail": pred_hu.new_tensor(refs["local_head_acf_tail"]),
            "loss_ref_pet_whole_body": pred_hu.new_tensor(refs["pet_whole_body"]),
            "loss_ref_pet_organ_bias": pred_hu.new_tensor(refs["pet_organ_bias"]),
            "loss_ref_pet_brain_outlier": pred_hu.new_tensor(refs["pet_brain_outlier"]),
            "lambda_hu": pred_hu.new_tensor(lambdas["hu"]),
            "lambda_grad": pred_hu.new_tensor(lambdas["grad"]),
            "lambda_mu_official": pred_hu.new_tensor(lambdas["mu_raw"]),
            "lambda_mu": pred_hu.new_tensor(lambdas["mu"]),
            "lambda_proj": pred_hu.new_tensor(lambdas["proj"]),
            "lambda_organ": pred_hu.new_tensor(lambdas["organ"]),
            "lambda_global_mu_official": pred_hu.new_tensor(lambdas["global_mu_raw"]),
            "lambda_global_mu": pred_hu.new_tensor(lambdas["global_mu"]),
            "lambda_global_acf": pred_hu.new_tensor(lambdas["global_acf"]),
            "lambda_global_acf_signed": pred_hu.new_tensor(lambdas["global_acf_signed"]),
            "lambda_regional_relacf": pred_hu.new_tensor(lambdas["regional_relacf"]),
            "lambda_organ_ray_signed": pred_hu.new_tensor(lambdas["organ_ray_signed"]),
            "lambda_local_head_acf": pred_hu.new_tensor(lambdas["local_head_acf"]),
            "lambda_local_head_acf_signed": pred_hu.new_tensor(lambdas["local_head_acf_signed"]),
            "lambda_local_head_acf_tail": pred_hu.new_tensor(lambdas["local_head_acf_tail"]),
            "lambda_pet_whole_body": pred_hu.new_tensor(lambdas["pet_whole_body"]),
            "lambda_pet_organ_bias": pred_hu.new_tensor(lambdas["pet_organ_bias"]),
            "lambda_pet_brain_outlier": pred_hu.new_tensor(lambdas["pet_brain_outlier"]),
            "mu_MAE_official_raw": loss_mu_raw.detach(),
            "HU_MAE_prediction_mask": masked_mean_metric(abs_hu, mask).detach(),
            "HU_MAE_body": masked_mean_metric(abs_hu, body_mask).detach(),
            "mu_MAE_smoothed": loss_mu.detach(),
            "projection_mu_error": loss_proj.detach(),
            "projection_mu_error_D": projection_components["D"].detach(),
            "projection_mu_error_H": projection_components["H"].detach(),
            "projection_mu_error_W": projection_components["W"].detach(),
            "projection_mu_error_orth": projection_components["orth"].detach(),
            "projection_mu_error_multi_angle": projection_components["multi_angle"].detach(),
            "global_mu_MAE_official_raw": loss_global_mu_raw.detach(),
            "global_mu_MAE_smoothed": loss_global_mu.detach(),
            "global_logacf_error": loss_global_acf.detach(),
            "global_logacf_signed_bias": loss_global_acf_signed.detach(),
            "global_logacf_signed_mean": global_components["mean_signed"].detach(),
            "global_logacf_tail_cvar": global_components["tail"].detach(),
            "global_logacf_p95": global_components["p95"].detach(),
            "global_logacf_error_D": global_components["D"].detach(),
            "global_logacf_error_H": global_components["H"].detach(),
            "global_logacf_error_W": global_components["W"].detach(),
            "global_logacf_error_orth": global_components["orth"].detach(),
            "global_logacf_error_multi_angle": global_components["multi_angle"].detach(),
            "global_logacf_signed_bias_orth": global_components.get("signed_orth", zero).detach(),
            "global_logacf_signed_bias_multi_angle": global_components.get("signed_multi_angle", zero).detach(),
            "local_head_logacf_error": local_head_metric,
            "local_head_logacf_signed_bias": local_head_signed_metric,
            "local_head_relacf_error": local_head_metric,
            "local_head_relacf_signed_bias": local_head_signed_metric,
            "local_head_logacf_signed_mean": (
                local_head_components["mean_signed"].detach()
                if has_local_head
                else pred_hu.new_tensor(float("nan"))
            ),
            "local_head_logacf_tail_cvar": (
                local_head_components["tail"].detach()
                if has_local_head
                else pred_hu.new_tensor(float("nan"))
            ),
            "local_head_relacf_tail_cvar": (
                local_head_components["tail"].detach()
                if has_local_head
                else pred_hu.new_tensor(float("nan"))
            ),
            "local_head_logacf_p95": (
                local_head_components["p95"].detach()
                if has_local_head
                else pred_hu.new_tensor(float("nan"))
            ),
            "local_head_logacf_error_multi_angle": local_head_components.get("multi_angle", zero).detach(),
            "local_head_logacf_signed_bias_multi_angle": local_head_components.get(
                "signed_multi_angle", zero
            ).detach(),
            "local_head_sample_rate": local_head_sample_rate.detach(),
            "local_head_voxel_fraction": local_head_voxel_fraction.detach(),
            "local_head_transaxial_coverage": local_head_coverage.detach(),
            "organ_bias_mean": organ_metric,
            "regional_relacf_error": loss_regional_relacf.detach(),
            "organ_ray_relacf_error": (
                loss_organ_ray_abs.detach()
                if organ_ray_components
                else pred_hu.new_tensor(float("nan"))
            ),
            "organ_ray_signed_bias": (
                loss_organ_ray_signed.detach()
                if organ_ray_components
                else pred_hu.new_tensor(float("nan"))
            ),
            "bone_MAE": masked_mean_metric(abs_hu, bone_mask).detach(),
            "lung_MAE": masked_mean_metric(abs_hu, lung_mask).detach(),
            "soft_tissue_MAE": masked_mean_metric(abs_hu, soft_mask).detach(),
        }
        for component_name, value in global_components.items():
            if component_name.startswith("signed_angle_"):
                suffix = component_name.removeprefix("signed_angle_")
                loss_dict[f"global_logacf_signed_bias_angle_{suffix}"] = value.detach()
        for component_name, value in local_head_components.items():
            if component_name.startswith("signed_angle_"):
                suffix = component_name.removeprefix("signed_angle_")
                loss_dict[f"local_head_logacf_signed_bias_angle_{suffix}"] = value.detach()
        for component_name, value in regional_components.items():
            loss_dict[f"regional_relacf_{component_name}"] = value.detach()
        loss_dict.update(pet_proxy_diagnostics)
        loss_dict.update(organ_ray_components)
        loss_dict.update(per_organ)
        return total, loss_dict


class MDixonLoss(CombinedPseudoCTLoss):
    """Backward-compatible alias for older train.py imports."""

    pass
