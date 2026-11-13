from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pet_supervision import PET_REGION_NAMES


EPS = 1e-6
OFFICIAL_ORGAN_LABELS = {
    "brain": 90,
    "liver": 5,
    "spleen": 1,
    "heart": 51,
    "pancreas": 7,
}


def _factor3(values: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(values, int):
        result = (values, values, values)
    else:
        result = tuple(int(value) for value in values)
    if len(result) != 3 or any(value < 1 for value in result):
        raise ValueError(f"Expected three positive downsample factors, got {values}")
    return result


def _average_pool(x: torch.Tensor, factors: tuple[int, int, int]) -> torch.Tensor:
    if factors == (1, 1, 1):
        return x
    return F.avg_pool3d(x, kernel_size=factors, stride=factors)


def _nearest_pool(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    if tuple(x.shape[2:]) == tuple(int(value) for value in size):
        return x
    return F.interpolate(x.float(), size=tuple(size), mode="nearest")


def compose_patch_delta_mu_full_fov(
    patch_delta_mu: torch.Tensor,
    crop_starts: torch.Tensor,
    original_shapes: torch.Tensor,
    output_shape: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place differentiable patch attenuation errors into one full-FOV grid.

    MONAI emits multiple crops from one subject as the effective batch. Each
    crop is sampled at its absolute location in the original NIfTI grid and
    mapped to the compact whole-body grid with ``align_corners=False``
    semantics. Overlapping crop predictions are averaged instead of summed.
    Voxels outside every crop have zero attenuation error, which isolates the
    physical effect of the pseudo-CT regions optimized by the current step.
    """
    if patch_delta_mu.ndim != 5 or int(patch_delta_mu.shape[1]) != 1:
        raise ValueError(
            f"Expected patch_delta_mu [P,1,D,H,W], got {tuple(patch_delta_mu.shape)}"
        )
    full_shape = tuple(int(value) for value in output_shape)
    if len(full_shape) != 3 or any(value <= 0 for value in full_shape):
        raise ValueError(f"Invalid full-FOV output shape: {output_shape}")

    patch_count = int(patch_delta_mu.shape[0])
    starts = torch.as_tensor(
        crop_starts,
        device=patch_delta_mu.device,
        dtype=patch_delta_mu.dtype,
    ).reshape(-1, 3)
    shapes = torch.as_tensor(
        original_shapes,
        device=patch_delta_mu.device,
        dtype=patch_delta_mu.dtype,
    ).reshape(-1, 3)
    if int(starts.shape[0]) not in {1, patch_count}:
        raise ValueError(
            f"crop_starts must contain one row or one row per patch, got {tuple(starts.shape)}"
        )
    if int(shapes.shape[0]) not in {1, patch_count}:
        raise ValueError(
            "original_shapes must contain one row or one row per patch, "
            f"got {tuple(shapes.shape)}"
        )
    if int(starts.shape[0]) == 1:
        starts = starts.expand(patch_count, -1)
    if int(shapes.shape[0]) == 1:
        shapes = shapes.expand(patch_count, -1)

    patch_shape = tuple(int(value) for value in patch_delta_mu.shape[2:])
    delta_sum = patch_delta_mu.new_zeros((1, 1, *full_shape))
    coverage_sum = patch_delta_mu.new_zeros((1, 1, *full_shape))

    output_axes = [
        torch.arange(size, device=patch_delta_mu.device, dtype=patch_delta_mu.dtype)
        for size in full_shape
    ]
    for patch_index in range(patch_count):
        normalized_axes = []
        for axis, output_axis in enumerate(output_axes):
            original_size = shapes[patch_index, axis].clamp_min(1.0)
            original_coordinate = (
                (output_axis + 0.5) * original_size / float(full_shape[axis]) - 0.5
            )
            local_coordinate = original_coordinate - starts[patch_index, axis]
            normalized = (
                2.0 * (local_coordinate + 0.5) / float(patch_shape[axis]) - 1.0
            )
            normalized_axes.append(normalized)

        grid_d, grid_h, grid_w = torch.meshgrid(
            normalized_axes[0],
            normalized_axes[1],
            normalized_axes[2],
            indexing="ij",
        )
        # grid_sample orders coordinates as W/H/D for a [B,C,D,H,W] tensor.
        grid = torch.stack((grid_w, grid_h, grid_d), dim=-1).unsqueeze(0)
        mapped_delta = F.grid_sample(
            patch_delta_mu[patch_index : patch_index + 1],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        mapped_coverage = F.grid_sample(
            torch.ones_like(patch_delta_mu[patch_index : patch_index + 1]),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        delta_sum = delta_sum + mapped_delta
        coverage_sum = coverage_sum + mapped_coverage

    composed = delta_sum / coverage_sum.clamp_min(1.0)
    covered = coverage_sum > 0
    return composed * covered.to(composed.dtype), covered.to(composed.dtype)


def _transaxial_padding(shape: Sequence[int]) -> tuple[int, int, int, int]:
    d, h = int(shape[0]), int(shape[1])
    diagonal = int(math.ceil(math.sqrt(d * d + h * h)))
    pad_d = max(0, diagonal - d)
    pad_h = max(0, diagonal - h)
    return pad_d // 2, pad_d - pad_d // 2, pad_h // 2, pad_h - pad_h // 2


def _pad_transaxial(
    x: torch.Tensor,
    padding: tuple[int, int, int, int],
) -> torch.Tensor:
    d_left, d_right, h_left, h_right = padding
    return F.pad(x, (0, 0, h_left, h_right, d_left, d_right))


def _crop_transaxial(
    x: torch.Tensor,
    padding: tuple[int, int, int, int],
    output_shape: Sequence[int],
) -> torch.Tensor:
    d_left, _, h_left, _ = padding
    d, h = int(output_shape[0]), int(output_shape[1])
    return x[:, :, d_left : d_left + d, h_left : h_left + h, :]


def _rotate_transaxial(x: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    """Rotate the NIfTI x/y plane while preserving the z axis."""
    angle = math.radians(float(angle_degrees))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    theta = x.new_tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosine, -sine, 0.0],
            [0.0, sine, cosine, 0.0],
        ]
    ).unsqueeze(0)
    theta = theta.expand(int(x.shape[0]), -1, -1)
    grid = F.affine_grid(theta, size=x.shape, align_corners=False)
    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def relative_acf_response_map(
    pred_mu: torch.Tensor,
    target_mu: torch.Tensor,
    support_mask: torch.Tensor,
    spacing_mm: Sequence[float],
    angles_degrees: Sequence[float],
    downsample_factors: int | Sequence[int] = (4, 4, 2),
    delta_tau_clip: float = 0.5,
    checkpoint_angles: bool = False,
    ray_activity: torch.Tensor | None = None,
    ray_activity_alpha: float = 0.0,
    ray_activity_weight_clip: Sequence[float] = (0.5, 2.0),
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Approximate voxelwise PET bias caused by attenuation-map error.

    For every transaxial line of response, ``delta_tau`` is the line integral
    of ``mu_pred - mu_gt`` and ``expm1(delta_tau)`` is the first-order relative
    ACF error. The signed ray errors are backprojected and averaged over angles.
    This is intentionally a PET-response surrogate, not a differentiable OSEM
    reconstruction.
    """
    if pred_mu.shape != target_mu.shape:
        raise ValueError(f"Mu-map shape mismatch: {pred_mu.shape} != {target_mu.shape}")
    if pred_mu.ndim != 5 or pred_mu.shape[1] != 1:
        raise ValueError(f"Expected [B,1,D,H,W] mu maps, got {pred_mu.shape}")
    angles = [float(value) for value in angles_degrees]
    if not angles:
        raise ValueError("At least one response angle is required")
    if float(delta_tau_clip) <= 0:
        raise ValueError("delta_tau_clip must be positive")

    factors = _factor3(downsample_factors)
    delta_mu = _average_pool((pred_mu - target_mu).float(), factors)
    low_shape = tuple(int(value) for value in delta_mu.shape[2:])
    support = _nearest_pool((support_mask > 0).float(), low_shape)
    if ray_activity is None:
        activity_low = torch.ones_like(delta_mu)
    else:
        activity_low = F.interpolate(
            ray_activity.detach().float(),
            size=low_shape,
            mode="trilinear",
            align_corners=False,
        ).clamp_min(0.0)

    padding = _transaxial_padding(low_shape[:2])
    delta_mu = _pad_transaxial(delta_mu, padding)
    support = _pad_transaxial(support, padding)
    activity_low = _pad_transaxial(activity_low, padding)

    spacing_y_cm = float(spacing_mm[1]) * float(factors[1]) / 10.0
    response_sum = torch.zeros_like(delta_mu)
    coverage_sum = torch.zeros_like(delta_mu)
    mean_absolute_rays = []
    mean_signed_rays = []
    mean_activity_weights = []
    activity_alpha = min(max(float(ray_activity_alpha), 0.0), 1.0)
    activity_clip = tuple(float(value) for value in ray_activity_weight_clip)
    if len(activity_clip) != 2 or activity_clip[0] <= 0 or activity_clip[1] < activity_clip[0]:
        raise ValueError(
            "ray_activity_weight_clip must be [positive_min, max>=min], "
            f"got {ray_activity_weight_clip}"
        )

    def angle_contribution(
        current_delta: torch.Tensor,
        current_support: torch.Tensor,
        current_activity: torch.Tensor,
        angle: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rotated_delta = _rotate_transaxial(current_delta, angle)
        rotated_support = _rotate_transaxial(current_support, angle)
        delta_tau = rotated_delta.sum(dim=3) * spacing_y_cm
        ray_support = rotated_support.sum(dim=3) > EPS
        relative_acf = torch.expm1(
            torch.clamp(delta_tau, -float(delta_tau_clip), float(delta_tau_clip))
        )

        supported_values = relative_acf[ray_support]
        if supported_values.numel() > 0:
            mean_absolute = supported_values.abs().mean()
            mean_signed = supported_values.mean()
        else:
            mean_absolute = current_delta.new_tensor(0.0)
            mean_signed = current_delta.new_tensor(0.0)

        if activity_alpha > 0.0:
            rotated_activity = _rotate_transaxial(current_activity, angle)
            ray_support_float = ray_support.float()
            ray_activity_mean = (
                (rotated_activity * rotated_support).sum(dim=3)
                / rotated_support.sum(dim=3).clamp_min(EPS)
            )
            reduce_dims = tuple(range(1, ray_activity_mean.ndim))
            supported_sum = (ray_activity_mean * ray_support_float).sum(
                dim=reduce_dims, keepdim=True
            )
            supported_count = ray_support_float.sum(dim=reduce_dims, keepdim=True)
            reference = supported_sum / supported_count.clamp_min(EPS)
            normalized_activity = ray_activity_mean / reference.clamp_min(EPS)
            activity_weight = (
                (1.0 - activity_alpha) + activity_alpha * normalized_activity
            ).clamp(min=activity_clip[0], max=activity_clip[1])
            activity_weight = torch.where(
                ray_support,
                activity_weight,
                torch.zeros_like(activity_weight),
            )
        else:
            activity_weight = ray_support.float()

        ray_field = (
            relative_acf * activity_weight
        ).unsqueeze(3).expand_as(rotated_delta)
        ray_coverage = activity_weight.unsqueeze(3).expand_as(rotated_delta)
        mean_activity_weight = (
            activity_weight.sum() / ray_support.sum().clamp_min(1)
        )
        return (
            _rotate_transaxial(ray_field, -angle),
            _rotate_transaxial(ray_coverage, -angle),
            mean_absolute,
            mean_signed,
            mean_activity_weight,
        )

    for angle in angles:
        if checkpoint_angles and torch.is_grad_enabled() and delta_mu.requires_grad:
            contribution = checkpoint(
                lambda current_delta, current_support, current_activity, a=angle: angle_contribution(
                    current_delta,
                    current_support,
                    current_activity,
                    a,
                ),
                delta_mu,
                support,
                activity_low,
                use_reentrant=False,
            )
        else:
            contribution = angle_contribution(delta_mu, support, activity_low, angle)
        (
            response_contribution,
            coverage_contribution,
            mean_absolute,
            mean_signed,
            mean_activity_weight,
        ) = contribution
        response_sum = response_sum + response_contribution
        coverage_sum = coverage_sum + coverage_contribution
        mean_absolute_rays.append(mean_absolute)
        mean_signed_rays.append(mean_signed)
        mean_activity_weights.append(mean_activity_weight)

    response = response_sum / coverage_sum.clamp_min(EPS)
    response = _crop_transaxial(response, padding, low_shape)
    support = _crop_transaxial(support, padding, low_shape)
    response = response * (support > 0).to(response.dtype)

    zero = response.new_tensor(0.0)
    diagnostics = {
        "ray_relacf_abs_mean": (
            torch.stack(mean_absolute_rays).mean() if mean_absolute_rays else zero
        ),
        "ray_relacf_signed_mean": (
            torch.stack(mean_signed_rays).mean() if mean_signed_rays else zero
        ),
        "response_abs_mean": (
            response.abs().sum() / ((support > 0).sum() + EPS)
        ),
        "ray_activity_alpha": response.new_tensor(activity_alpha),
        "ray_activity_weight_mean": (
            torch.stack(mean_activity_weights).mean()
            if mean_activity_weights
            else response.new_tensor(1.0)
        ),
    }
    return response, support, diagnostics


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = mask.to(values.dtype).clamp(min=0.0, max=1.0)
    if weights is None:
        weights = torch.ones_like(values)
    effective = mask * weights.to(values.dtype)
    return (values * effective).sum() / effective.sum().clamp_min(EPS)


def _soft_exceedance(
    error: torch.Tensor,
    threshold: float,
    temperature: float,
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-4)
    threshold = float(threshold)
    base = torch.sigmoid(error.new_tensor(-threshold / temperature))
    value = torch.sigmoid((error - threshold) / temperature)
    return torch.clamp((value - base) / (1.0 - base + EPS), min=0.0, max=1.0)


def evaluator_aligned_pet_proxy_losses(
    pred_mu: torch.Tensor,
    target_mu: torch.Tensor,
    official_eval_mask: torch.Tensor,
    prediction_mask: torch.Tensor,
    organ_seg: torch.Tensor | None,
    body_region_seg: torch.Tensor | None,
    target_hu: torch.Tensor,
    activity_proxy: torch.Tensor | None,
    spacing_mm: Sequence[float],
    config: dict,
    exact_eval_mask: torch.Tensor | None = None,
    exact_region_masks: torch.Tensor | None = None,
    activity_is_gt_pet: torch.Tensor | bool | None = None,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Build surrogates with the same aggregation shape as official PET metrics."""
    if torch.is_tensor(activity_is_gt_pet):
        uses_gt_pet = bool(float(activity_is_gt_pet.detach().reshape(-1)[0].item()) > 0.5)
    else:
        uses_gt_pet = bool(activity_is_gt_pet)
    ray_cfg = config.get("ray_activity_weighting", {})
    if bool(ray_cfg.get("enabled", False)):
        ray_activity_alpha = float(
            ray_cfg.get("gt_pet_alpha", 0.5)
            if uses_gt_pet
            else ray_cfg.get("nacpet_alpha", 0.2)
        )
    else:
        ray_activity_alpha = 0.0
    response, support_low, diagnostics = relative_acf_response_map(
        pred_mu,
        target_mu,
        prediction_mask,
        spacing_mm=spacing_mm,
        angles_degrees=config.get("angles_deg", [0, 30, 60, 90, 120, 150]),
        downsample_factors=config.get("downsample_factors", [4, 4, 2]),
        delta_tau_clip=float(config.get("delta_tau_clip", 0.5)),
        checkpoint_angles=bool(config.get("checkpoint_angles", False)),
        ray_activity=activity_proxy,
        ray_activity_alpha=ray_activity_alpha,
        ray_activity_weight_clip=ray_cfg.get("weight_clip", [0.5, 2.0]),
    )
    low_shape = response.shape[2:]
    if uses_gt_pet and exact_eval_mask is not None:
        eval_mask = F.interpolate(
            exact_eval_mask.float(),
            size=low_shape,
            mode="trilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
    else:
        eval_mask = _nearest_pool(official_eval_mask.float(), low_shape)
    target_hu_low = F.interpolate(
        target_hu.float(), size=low_shape, mode="trilinear", align_corners=False
    )

    if activity_proxy is None:
        activity = torch.ones_like(response)
    else:
        activity = F.interpolate(
            activity_proxy.float(), size=low_shape, mode="trilinear", align_corners=False
        )
        activity = activity.clamp_min(0.0)
        clip_key = "gt_activity_weight_clip" if uses_gt_pet else "activity_weight_clip"
        activity_clip = config.get(clip_key, None)
        if activity_clip is not None:
            activity = activity.clamp_max(float(activity_clip))
    activity = activity.detach()

    # Official whole-body SUV MAE is an unweighted voxel mean of absolute SUV
    # error. PET ~= PET_gt * (1 + response), so NAC-PET is used only as a
    # detached activity-magnitude proxy.
    whole_body = _masked_mean(response.abs() * activity, eval_mask)

    organ_low = (
        _nearest_pool(organ_seg.float(), low_shape).round().long()
        if organ_seg is not None
        else None
    )
    body_region_low = (
        _nearest_pool(body_region_seg.float(), low_shape).round().long()
        if body_region_seg is not None
        else None
    )
    named_masks: Dict[str, torch.Tensor] = {}
    exact_masks_active = uses_gt_pet and exact_region_masks is not None
    if exact_masks_active:
        resized_exact = F.interpolate(
            exact_region_masks.float(),
            size=low_shape,
            mode="trilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        if int(resized_exact.shape[1]) != len(PET_REGION_NAMES):
            raise ValueError(
                "Expected exact PET masks in official region order with "
                f"{len(PET_REGION_NAMES)} channels, got {tuple(resized_exact.shape)}"
            )
        named_masks = {
            name: resized_exact[:, index : index + 1]
            for index, name in enumerate(PET_REGION_NAMES)
        }
    elif organ_low is not None:
        for name, label in OFFICIAL_ORGAN_LABELS.items():
            named_masks[name] = organ_low == int(label)

    if not exact_masks_active:
        named_organ_union = torch.zeros_like(response, dtype=torch.bool)
        for mask in named_masks.values():
            named_organ_union |= mask
        body_support = support_low > 0
        named_masks["adipose"] = (
            (target_hu_low >= float(config.get("adipose_hu_min", -190.0)))
            & (target_hu_low <= float(config.get("adipose_hu_max", -30.0)))
            & body_support
            & (~named_organ_union)
        )
        named_masks["muscle"] = (
            (target_hu_low >= float(config.get("muscle_hu_min", 10.0)))
            & (target_hu_low <= float(config.get("muscle_hu_max", 100.0)))
            & body_support
            & (~named_organ_union)
        )
        if body_region_low is not None:
            named_masks["extremities"] = body_region_low == int(
                config.get("extremities_body_label", 2)
            )

    min_voxels = int(config.get("min_region_voxels_lowres", 16))
    region_biases = []
    for name in (
        "brain",
        "liver",
        "spleen",
        "heart",
        "pancreas",
        "muscle",
        "adipose",
        "extremities",
    ):
        region = named_masks.get(name)
        region_occupancy = float(region.sum().detach().item()) if region is not None else 0.0
        minimum_occupancy = EPS if exact_masks_active else float(min_voxels)
        if region is None or region_occupancy < minimum_occupancy:
            if exact_masks_active:
                raise ValueError(f"Exact PET evaluator mask '{name}' is empty after resizing")
            diagnostics[f"pet_proxy_region_present_{name}"] = response.new_tensor(0.0)
            continue
        signed_bias = _masked_mean(response, region, weights=activity)
        region_biases.append(signed_bias.abs())
        diagnostics[f"pet_proxy_region_present_{name}"] = response.new_tensor(1.0)
        diagnostics[f"pet_proxy_region_signed_bias_{name}"] = signed_bias.detach()
        diagnostics[f"pet_proxy_region_abs_bias_{name}"] = signed_bias.abs().detach()
    organ_bias = (
        torch.stack(region_biases).mean()
        if region_biases
        else response.sum() * 0.0
    )

    brain_mask = named_masks.get("brain")
    brain_terms = []
    if brain_mask is not None:
        # The official evaluator excludes brain voxels with effectively zero
        # GT PET. NAC-PET positivity is the only inference-available proxy.
        brain_mask = brain_mask.to(response.dtype) * (activity > EPS).to(response.dtype)
    brain_minimum = EPS if exact_masks_active else float(min_voxels)
    if brain_mask is not None and float(brain_mask.sum().detach().item()) >= brain_minimum:
        relative_error = response.abs()
        for threshold in config.get("brain_thresholds", [0.05, 0.10, 0.15]):
            exceedance = _soft_exceedance(
                relative_error,
                threshold=float(threshold),
                temperature=float(config.get("brain_temperature", 0.01)),
            )
            term = _masked_mean(exceedance, brain_mask)
            brain_terms.append(term)
            key = int(round(float(threshold) * 100))
            diagnostics[f"pet_proxy_brain_exceedance_{key:02d}"] = term.detach()
    brain_outlier = (
        torch.stack(brain_terms).mean()
        if brain_terms
        else response.sum() * 0.0
    )

    losses = {
        "pet_whole_body": whole_body,
        "pet_organ_bias": organ_bias,
        "pet_brain_outlier": brain_outlier,
    }
    diagnostics.update(
        {
            "pet_proxy_whole_body_suv_mae": whole_body.detach(),
            "pet_proxy_organ_bias": organ_bias.detach(),
            "pet_proxy_brain_outlier": brain_outlier.detach(),
            "pet_proxy_activity_mean": _masked_mean(activity, eval_mask).detach(),
            "pet_proxy_uses_gt_pet": response.new_tensor(1.0 if uses_gt_pet else 0.0),
            "pet_proxy_uses_exact_masks": response.new_tensor(
                1.0 if exact_masks_active else 0.0
            ),
        }
    )
    return losses, diagnostics


def evaluator_aligned_full_fov_patch_proxy_losses(
    patch_delta_mu: torch.Tensor,
    crop_starts: torch.Tensor,
    original_shapes: torch.Tensor,
    global_official_eval_mask: torch.Tensor,
    global_prediction_mask: torch.Tensor,
    global_organ_seg: torch.Tensor | None,
    global_body_region_seg: torch.Tensor | None,
    global_target_hu: torch.Tensor,
    global_activity_proxy: torch.Tensor | None,
    global_spacing_mm: Sequence[float] | torch.Tensor,
    config: dict,
    global_pet_eval_mask: torch.Tensor | None = None,
    global_pet_region_masks: torch.Tensor | None = None,
    global_activity_is_gt_pet: torch.Tensor | bool | None = None,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Evaluate local pseudo-CT errors through full-body attenuation paths.

    The current subject's patch errors are composed on the compact full-FOV
    grid. The response projector then sees complete body-length lines of
    response instead of treating every crop boundary as a scanner boundary.
    Ground-truth attenuation outside sampled patches is assumed exact, so only
    the final pseudo-CT output receives gradient.
    """
    output_shape = tuple(int(value) for value in global_official_eval_mask.shape[-3:])
    configured_shape = tuple(
        int(value) for value in config.get("full_fov_shape", output_shape)
    )
    if configured_shape != output_shape:
        raise ValueError(
            "Full-FOV proxy grid does not match configured shape: "
            f"batch={output_shape}, config={configured_shape}"
        )
    delta_mu_full, patch_coverage = compose_patch_delta_mu_full_fov(
        patch_delta_mu,
        crop_starts=crop_starts,
        original_shapes=original_shapes,
        output_shape=output_shape,
    )

    spacing = torch.as_tensor(global_spacing_mm).detach().cpu().reshape(-1, 3)[0].tolist()
    full_config = dict(config)
    full_config["angles_deg"] = full_config.get(
        "full_fov_angles_deg",
        full_config.get("angles_deg", [0, 30, 60, 90, 120, 150]),
    )
    full_config["downsample_factors"] = full_config.get(
        "full_fov_downsample_factors",
        [1, 1, 1],
    )

    zero_target = torch.zeros_like(delta_mu_full)
    losses, diagnostics = evaluator_aligned_pet_proxy_losses(
        pred_mu=delta_mu_full,
        target_mu=zero_target,
        official_eval_mask=global_official_eval_mask,
        prediction_mask=global_prediction_mask,
        organ_seg=global_organ_seg,
        body_region_seg=global_body_region_seg,
        target_hu=global_target_hu,
        activity_proxy=global_activity_proxy,
        exact_eval_mask=global_pet_eval_mask,
        exact_region_masks=global_pet_region_masks,
        activity_is_gt_pet=global_activity_is_gt_pet,
        spacing_mm=spacing,
        config=full_config,
    )
    body = (global_prediction_mask > 0).to(patch_coverage.dtype)
    diagnostics["pet_proxy_full_fov_patch_count"] = delta_mu_full.new_tensor(
        float(patch_delta_mu.shape[0])
    )
    diagnostics["pet_proxy_full_fov_coverage_fraction"] = (
        (patch_coverage * body).sum() / body.sum().clamp_min(EPS)
    ).detach()
    diagnostics["pet_proxy_full_fov_delta_mu_abs_mean"] = _masked_mean(
        delta_mu_full.abs(),
        patch_coverage * body,
    ).detach()
    return losses, diagnostics
