from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from dataset import get_subject_ct_labels, get_subject_features
from evaluator_surrogate import evaluator_aligned_pet_proxy_losses
from pet_supervision import load_or_create_exact_pet_supervision
from losses import (
    MU_WATER,
    gaussian_smooth_3d,
    projection_mu_loss as torch_projection_mu_loss,
    superior_body_fraction_mask,
)
from pet_physics import OFFICIAL_GAUSSIAN_MODE, OFFICIAL_GAUSSIAN_TRUNCATE
from predict import (
    DEFAULT_PATCH_SIZE,
    OVERLAP,
    SW_BATCH,
    build_model_from_checkpoint,
    get_metadata_tensor,
    load_checkpoint,
    normalize_checkpoint_config,
    save_prediction_nifti,
)
from transforms import get_inference_transforms


LIVER_LABEL = 5
DEFAULT_ORGAN_MIN_VOXELS = 64
INFERENCE_LABEL_KEYS = frozenset(
    {
        "ct",
        "prediction_mask",
        "body_seg",
        "organ_seg",
        "official_eval_mask",
    }
)


def inference_features_only(subject: dict) -> dict:
    """Remove label paths before applying the inference-only transform graph.

    Internal validation dataset items contain both feature paths and label paths.
    The inference transform deliberately loads features only, so leaving label
    paths in the dictionary makes downstream optional-mask handling mistake a
    string path for an already loaded tensor.
    """
    return {key: value for key, value in subject.items() if key not in INFERENCE_LABEL_KEYS}


def load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(os.path.expandvars(handle.read())) or {}


def deep_merge(base: dict, override: dict | None) -> dict:
    out = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def list_subjects(val_dir: Path, subjects: list[str] | None, limit: int | None = None) -> list[str]:
    if subjects:
        selected = subjects
    else:
        selected = sorted(path.name for path in val_dir.iterdir() if path.is_dir() and path.name.startswith("sub-"))
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise RuntimeError(f"No subjects found under {val_dir}")
    return selected


def hu_to_mu_np(ct_hu: np.ndarray) -> np.ndarray:
    """Official Carney-style bilinear HU to 511-keV mu conversion."""
    hu1000 = ct_hu.astype(np.float32, copy=False) + 1000.0
    mu = np.where(hu1000 < 1047.0, 9.6e-5 * hu1000, 5.10e-5 * hu1000 + 4.71e-2)
    return np.clip(mu, 0.0, None).astype(np.float32, copy=False)


def smooth_mu_np(mu: np.ndarray, spacing_mm, fwhm_mm: float, device: torch.device) -> np.ndarray:
    if fwhm_mm <= 0:
        return mu.astype(np.float32, copy=False)

    try:
        from scipy.ndimage import gaussian_filter

        sigma_mm = float(fwhm_mm) / 2.355
        sigma_voxel = tuple(sigma_mm / float(v) for v in spacing_mm)
        return gaussian_filter(
            mu.astype(np.float32, copy=False),
            sigma=sigma_voxel,
            mode=OFFICIAL_GAUSSIAN_MODE,
            truncate=OFFICIAL_GAUSSIAN_TRUNCATE,
        ).astype(np.float32, copy=False)
    except Exception:
        x = torch.from_numpy(mu.astype(np.float32, copy=False)).view(1, 1, *mu.shape).to(device)
        with torch.no_grad():
            y = gaussian_smooth_3d(
                x,
                spacing_mm=spacing_mm,
                fwhm_mm=fwhm_mm,
                padding_mode=OFFICIAL_GAUSSIAN_MODE,
                truncate=OFFICIAL_GAUSSIAN_TRUNCATE,
            )
        return y.cpu().numpy()[0, 0].astype(np.float32, copy=False)


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return float("nan")
    return float(np.mean(values[mask]))


def projection_mu_error(
    pred_mu_s: np.ndarray,
    gt_mu_s: np.ndarray,
    mask: np.ndarray | None,
    spacing_mm,
    axis_weights=None,
    projection_cfg: dict | None = None,
    ray_support_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Use exactly the same supported-ray implementation as training."""
    pred = torch.from_numpy(np.asarray(pred_mu_s, dtype=np.float32)).view(1, 1, *pred_mu_s.shape)
    target = torch.from_numpy(np.asarray(gt_mu_s, dtype=np.float32)).view(1, 1, *gt_mu_s.shape)
    integration = None
    if mask is not None:
        integration = torch.from_numpy(np.asarray(mask, dtype=np.float32)).view(1, 1, *mask.shape)
    support = ray_support_mask if ray_support_mask is not None else mask
    support_tensor = None
    if support is not None:
        support_tensor = torch.from_numpy(np.asarray(support, dtype=np.float32)).view(1, 1, *support.shape)
    with torch.no_grad():
        total, component_tensors = torch_projection_mu_loss(
            pred,
            target,
            integration,
            spacing_mm=spacing_mm,
            axis_weights=axis_weights,
            projection_cfg=projection_cfg,
            ray_support_mask=support_tensor,
            return_components=True,
        )
    components = {name: float(value.item()) for name, value in component_tensors.items()}
    components["mean"] = float(total.item())
    return components


def _projection_target_size(cfg: dict, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    fv_cfg = cfg.get("full_volume_validation", {})
    explicit = fv_cfg.get("global_projection_spatial_size")
    if explicit is None:
        explicit = cfg.get("model", {}).get("global_context", {}).get("spatial_size")
    if explicit is None:
        explicit = [96, 96, 128]
    target = tuple(int(value) for value in explicit)
    if len(target) != 3 or any(value <= 0 for value in target):
        raise ValueError(f"Invalid full-volume global projection size: {explicit}")
    return tuple(min(int(src), int(dst)) for src, dst in zip(shape, target))


def _area_downsample_np(x: np.ndarray, target_size: tuple[int, int, int]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if tuple(x.shape) == tuple(target_size):
        return x.astype(np.float32, copy=False)
    tensor = torch.from_numpy(x).view(1, 1, *x.shape)
    with torch.no_grad():
        out = F.interpolate(tensor, size=tuple(target_size), mode="area")
    return out.numpy()[0, 0].astype(np.float32, copy=False)


def _downsampled_spacing(spacing_mm, source_shape, target_shape) -> tuple[float, float, float]:
    return tuple(
        float(spacing) * float(src) / float(dst)
        for spacing, src, dst in zip(spacing_mm, source_shape, target_shape)
    )


def global_logacf_projection_error(
    pred_mu_s: np.ndarray,
    gt_mu_s: np.ndarray,
    prediction_mask: np.ndarray,
    spacing_mm,
    cfg: dict,
    body_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Full-volume PET-style log-ACF proxy using the training global projection config.

    Inputs are already in the PET reconstruction order: submitted CT/GT CT were
    converted to mu, face/bed regions were handled through the effective map,
    and the full field was smoothed before this area downsampling step.
    """
    shape = tuple(int(value) for value in pred_mu_s.shape)
    if tuple(gt_mu_s.shape) != shape or tuple(prediction_mask.shape) != shape:
        raise RuntimeError(
            "Global projection shape mismatch: "
            f"pred={pred_mu_s.shape}, gt={gt_mu_s.shape}, mask={prediction_mask.shape}"
        )
    if body_mask is not None and tuple(body_mask.shape) != shape:
        raise RuntimeError(f"Global body-mask shape mismatch: body={body_mask.shape}, expected={shape}")

    target_size = _projection_target_size(cfg, shape)
    pred_low = _area_downsample_np(pred_mu_s, target_size)
    gt_low = _area_downsample_np(gt_mu_s, target_size)
    support_low = _area_downsample_np(prediction_mask.astype(np.float32, copy=False), target_size)
    body_source = prediction_mask if body_mask is None else body_mask
    body_low = _area_downsample_np(body_source.astype(np.float32, copy=False), target_size)
    effective_spacing = _downsampled_spacing(spacing_mm, shape, target_size)

    projection_cfg = dict(cfg.get("loss", {}).get("global_projection", {}))
    if not projection_cfg:
        projection_cfg = dict(cfg.get("loss", {}).get("projection", {}))

    raw = projection_mu_error(
        pred_low,
        gt_low,
        None,
        spacing_mm=effective_spacing,
        axis_weights=projection_cfg.get("axis_weights", [1.0, 1.0, 1.0]),
        projection_cfg=projection_cfg,
        ray_support_mask=support_low,
    )
    metrics = {
        "full_volume_logacf_error": raw["mean"],
        "full_volume_relacf_error": raw["mean"],
        "full_volume_logacf_error_D": raw.get("D", float("nan")),
        "full_volume_logacf_error_H": raw.get("H", float("nan")),
        "full_volume_logacf_error_W": raw.get("W", float("nan")),
        "full_volume_logacf_error_orth": raw.get("orth", float("nan")),
        "full_volume_logacf_error_multi_angle": raw.get("multi_angle", float("nan")),
        "full_volume_logacf_signed_bias": raw.get("signed", float("nan")),
        "full_volume_logacf_signed_mean": raw.get("mean_signed", float("nan")),
        "full_volume_logacf_tail_cvar": raw.get("tail", float("nan")),
        "full_volume_relacf_tail_cvar": raw.get("tail", float("nan")),
        "full_volume_logacf_p95": raw.get("p95", float("nan")),
        "full_volume_logacf_signed_bias_orth": raw.get("signed_orth", float("nan")),
        "full_volume_logacf_signed_bias_multi_angle": raw.get("signed_multi_angle", float("nan")),
        "full_volume_logacf_grid_x": float(target_size[0]),
        "full_volume_logacf_grid_y": float(target_size[1]),
        "full_volume_logacf_grid_z": float(target_size[2]),
        "full_volume_logacf_spacing_x_mm": float(effective_spacing[0]),
        "full_volume_logacf_spacing_y_mm": float(effective_spacing[1]),
        "full_volume_logacf_spacing_z_mm": float(effective_spacing[2]),
    }
    for name, value in raw.items():
        if name.startswith("angle_"):
            metrics[f"full_volume_logacf_error_{name}"] = value
        elif name.startswith("signed_angle_"):
            suffix = name.removeprefix("signed_angle_")
            metrics[f"full_volume_logacf_signed_bias_angle_{suffix}"] = value
        elif name.startswith("mean_signed_angle_"):
            suffix = name.removeprefix("mean_signed_angle_")
            metrics[f"full_volume_logacf_signed_mean_angle_{suffix}"] = value
        elif name.startswith("tail_angle_"):
            suffix = name.removeprefix("tail_angle_")
            metrics[f"full_volume_logacf_tail_cvar_angle_{suffix}"] = value
        elif name.startswith("p95_angle_"):
            suffix = name.removeprefix("p95_angle_")
            metrics[f"full_volume_logacf_p95_angle_{suffix}"] = value

    head_cfg = dict(cfg.get("loss", {}).get("head_projection", {}))
    if head_cfg.get("enabled", False):
        body_tensor = torch.from_numpy(body_low).view(1, 1, *body_low.shape)
        head_region = superior_body_fraction_mask(
            body_tensor,
            superior_fraction=float(head_cfg.get("superior_fraction", 0.22)),
            spatial_axis=int(head_cfg.get("spatial_axis", 2)),
        ).numpy()[0, 0]
        head_valid = head_region * support_low
        head = projection_mu_error(
            pred_low,
            gt_low,
            head_valid,
            spacing_mm=effective_spacing,
            axis_weights=head_cfg.get("axis_weights", [1.0, 1.0, 0.0]),
            projection_cfg=head_cfg,
            ray_support_mask=head_valid,
        )
        metrics.update(
            {
                "full_volume_head_logacf_error": head["mean"],
                "full_volume_head_relacf_error": head["mean"],
                "full_volume_head_logacf_signed_bias": head.get("signed", float("nan")),
                "full_volume_head_logacf_signed_mean": head.get("mean_signed", float("nan")),
                "full_volume_head_logacf_tail_cvar": head.get("tail", float("nan")),
                "full_volume_head_relacf_tail_cvar": head.get("tail", float("nan")),
                "full_volume_head_logacf_p95": head.get("p95", float("nan")),
                "full_volume_head_logacf_error_multi_angle": head.get("multi_angle", float("nan")),
                "full_volume_head_logacf_signed_bias_multi_angle": head.get(
                    "signed_multi_angle", float("nan")
                ),
                "full_volume_head_voxel_fraction": float(head_valid.sum() / (support_low.sum() + 1e-6)),
            }
        )
        for name, value in head.items():
            if name.startswith("angle_"):
                metrics[f"full_volume_head_logacf_error_{name}"] = value
            elif name.startswith("signed_angle_"):
                suffix = name.removeprefix("signed_angle_")
                metrics[f"full_volume_head_logacf_signed_bias_angle_{suffix}"] = value
            elif name.startswith("mean_signed_angle_"):
                suffix = name.removeprefix("mean_signed_angle_")
                metrics[f"full_volume_head_logacf_signed_mean_angle_{suffix}"] = value
            elif name.startswith("tail_angle_"):
                suffix = name.removeprefix("tail_angle_")
                metrics[f"full_volume_head_logacf_tail_cvar_angle_{suffix}"] = value
            elif name.startswith("p95_angle_"):
                suffix = name.removeprefix("p95_angle_")
                metrics[f"full_volume_head_logacf_p95_angle_{suffix}"] = value
    return metrics


def liver_exclusion_mask(shape, body_mask: np.ndarray, organ_seg: np.ndarray | None, spacing_z_mm: float, exclusion_cm: float):
    if organ_seg is None:
        return body_mask.copy(), 0, -1, -1

    liver = organ_seg == LIVER_LABEL
    if not np.any(liver):
        return body_mask.copy(), 0, -1, -1

    exclusion_slices = int(round((float(exclusion_cm) * 10.0) / float(spacing_z_mm)))
    superior_slice = int(np.max(np.where(liver)[2]))
    z_min = max(0, superior_slice - exclusion_slices)
    z_max = min(shape[2], superior_slice + exclusion_slices)

    exclusion = np.zeros(shape, dtype=bool)
    exclusion[:, :, z_min:z_max] = True
    return body_mask & (~exclusion), 1, z_min, z_max


def load_optional_mask(path: str | None, shape, dtype=bool):
    if not path:
        return None
    loaded = nib.load(path).get_fdata()
    if loaded.shape != tuple(shape):
        raise RuntimeError(f"Shape mismatch for {path}: {loaded.shape} vs {shape}")
    if dtype is bool:
        return loaded > 0
    return loaded.astype(dtype, copy=False)


def load_nac_activity_proxy(
    path: str | None,
    shape,
    body_mask: np.ndarray,
    max_relative_activity: float,
) -> np.ndarray:
    """Build a subject-normalized activity proxy from raw NAC-PET.

    The official SUV factor is unavailable without GT PET. Dividing raw
    NAC-PET by its positive-voxel mean removes arbitrary subject scale while
    preserving the uptake distribution needed by the evaluator-shaped proxy.
    """
    if not path:
        return np.ones(shape, dtype=np.float32)
    activity = nib.load(path).get_fdata(dtype=np.float32)
    if activity.shape != tuple(shape):
        raise RuntimeError(f"NAC-PET shape mismatch for {path}: {activity.shape} vs {shape}")

    finite_positive = np.isfinite(activity) & (activity > 0)
    if not np.any(finite_positive):
        return np.ones(shape, dtype=np.float32)
    values = activity[finite_positive]
    lower, upper = np.percentile(values, [0.5, 99.5])
    clipped = np.clip(activity, lower, upper)
    clipped = np.where(finite_positive, clipped, 0.0).astype(np.float32, copy=False)

    reference_mask = finite_positive & body_mask
    if not np.any(reference_mask):
        reference_mask = finite_positive
    reference = max(float(np.mean(clipped[reference_mask])), 1e-6)
    return np.clip(
        np.nan_to_num(clipped / reference, nan=0.0, posinf=max_relative_activity),
        0.0,
        float(max_relative_activity),
    ).astype(np.float32, copy=False)


def compute_metrics_for_subject(
    pred_hu: np.ndarray,
    label_paths: dict,
    cfg: dict,
    metric_device: torch.device,
    min_organ_voxels: int,
    exclusion_cm: float,
    nacpet_path: str | None = None,
) -> dict:
    gt_img = nib.load(label_paths["ct"])
    gt_hu = gt_img.get_fdata(dtype=np.float32)
    if pred_hu.shape != gt_hu.shape:
        raise RuntimeError(f"Prediction/GT shape mismatch: {pred_hu.shape} vs {gt_hu.shape}")

    spacing_mm = gt_img.header.get_zooms()[:3]
    body_region_seg = load_optional_mask(
        label_paths.get("body_seg"), gt_hu.shape, dtype=np.int16
    )
    body_mask = body_region_seg > 0 if body_region_seg is not None else None
    prediction_mask = load_optional_mask(label_paths.get("prediction_mask"), gt_hu.shape)
    organ_seg = load_optional_mask(label_paths.get("organ_seg"), gt_hu.shape, dtype=np.int16)

    if body_mask is None:
        body_mask = np.isfinite(gt_hu)
    if prediction_mask is None:
        prediction_mask = body_mask

    valid_mask = body_mask & prediction_mask
    clip_min = float(cfg.get("loss", {}).get("ct_clip_min", -1000.0))
    clip_max = float(cfg.get("loss", {}).get("ct_clip_max", 2000.0))
    pred_hu_eval = np.clip(pred_hu.astype(np.float32, copy=False), clip_min, clip_max)
    gt_hu_eval = np.clip(gt_hu, clip_min, clip_max)
    pred_mu = hu_to_mu_np(pred_hu_eval)
    gt_mu_official = hu_to_mu_np(gt_hu)
    pred_mu_effective = np.where(prediction_mask, pred_mu, gt_mu_official).astype(np.float32, copy=False)

    fwhm_mm = float(cfg.get("loss", {}).get("mu_fwhm_mm", 4.0))
    pred_mu_s = smooth_mu_np(pred_mu_effective, spacing_mm=spacing_mm, fwhm_mm=fwhm_mm, device=metric_device)
    gt_mu_s = smooth_mu_np(gt_mu_official, spacing_mm=spacing_mm, fwhm_mm=fwhm_mm, device=metric_device)

    official_mask, exclusion_used, z_min, z_max = liver_exclusion_mask(
        gt_hu.shape,
        body_mask=body_mask,
        organ_seg=organ_seg,
        spacing_z_mm=spacing_mm[2],
        exclusion_cm=exclusion_cm,
    )

    abs_hu = np.abs(pred_hu_eval - gt_hu_eval)
    abs_mu = np.abs(pred_mu - gt_mu_official)
    # Official CT mu-MAE is evaluated directly on the submitted CT. The
    # effective (face/bed-swapped) map is only used by PET reconstruction
    # proxies below.
    abs_mu_official = np.abs(pred_mu - gt_mu_official)
    abs_mu_s = np.abs(pred_mu_s - gt_mu_s)

    air_thr = float(cfg.get("loss", {}).get("air_threshold", -300.0))
    bone_thr = float(cfg.get("loss", {}).get("bone_threshold", 150.0))
    lung_mask = (gt_hu_eval < air_thr) & valid_mask
    bone_mask = (gt_hu_eval > bone_thr) & valid_mask
    soft_mask = (gt_hu_eval >= air_thr) & (gt_hu_eval <= bone_thr) & valid_mask

    mu_mae_smoothed_raw = safe_mean(abs_mu_s, valid_mask)
    projection = projection_mu_error(
        pred_mu_s,
        gt_mu_s,
        None,
        spacing_mm=spacing_mm,
        axis_weights=cfg.get("loss", {}).get("projection_axis_weights", [1.0, 1.0, 1.0]),
        projection_cfg=cfg.get("loss", {}).get("projection", {}),
        ray_support_mask=prediction_mask,
    )
    global_projection = global_logacf_projection_error(
        pred_mu_s=pred_mu_s,
        gt_mu_s=gt_mu_s,
        prediction_mask=prediction_mask,
        spacing_mm=spacing_mm,
        cfg=cfg,
        body_mask=body_mask,
    )
    metrics = {
        "ct_mu_map_mae_official_like": safe_mean(abs_mu_official, official_mask),
        "mu_MAE_smoothed_raw": mu_mae_smoothed_raw,
        "mu_MAE_smoothed": mu_mae_smoothed_raw / MU_WATER if math.isfinite(mu_mae_smoothed_raw) else float("nan"),
        "projection_mu_error": projection["mean"],
        "projection_mu_error_D": projection["D"],
        "projection_mu_error_H": projection["H"],
        "projection_mu_error_W": projection["W"],
        "projection_mu_error_orth": projection["orth"],
        "projection_mu_error_multi_angle": projection["multi_angle"],
        "HU_MAE_body": safe_mean(abs_hu, body_mask),
        "HU_MAE_prediction_mask": safe_mean(abs_hu, valid_mask),
        "mu_MAE_body_raw": safe_mean(abs_mu, body_mask),
        "mu_MAE_prediction_mask_raw": safe_mean(abs_mu, valid_mask),
        "bone_MAE": safe_mean(abs_hu, bone_mask),
        "lung_MAE": safe_mean(abs_hu, lung_mask),
        "soft_tissue_MAE": safe_mean(abs_hu, soft_mask),
        "bone_mu_MAE_raw": safe_mean(abs_mu, bone_mask),
        "lung_mu_MAE_raw": safe_mean(abs_mu, lung_mask),
        "soft_tissue_mu_MAE_raw": safe_mean(abs_mu, soft_mask),
        "eval_voxels": float(np.count_nonzero(valid_mask)),
        "official_eval_voxels": float(np.count_nonzero(official_mask)),
        "liver_exclusion_used": float(exclusion_used),
        "liver_exclusion_z_min": float(z_min),
        "liver_exclusion_z_max": float(z_max),
    }
    metrics.update(global_projection)

    pet_proxy_cfg = dict(cfg.get("loss", {}).get("pet_evaluator_proxy", {}))
    if pet_proxy_cfg.get("enabled", False):
        proxy_uses_compact_grid = (
            str(pet_proxy_cfg.get("training_mode", "")).lower()
            == "full_fov_patch_injected"
        )
        target_shape = (
            tuple(
                int(value)
                for value in pet_proxy_cfg.get(
                    "full_fov_shape",
                    cfg.get("model", {})
                    .get("global_context", {})
                    .get("spatial_size", [96, 96, 128]),
                )
            )
            if proxy_uses_compact_grid
            else tuple(int(value) for value in gt_hu.shape)
        )
        exact_pet_keys = (
            "pet_gt",
            "pet_body_seg",
            "pet_organ_seg",
            "pet_tissue_seg",
        )
        exact_pet = None
        if all(label_paths.get(key) for key in exact_pet_keys):
            pet_supervision_cfg = cfg.get("pet_supervision", {})
            exact_pet = load_or_create_exact_pet_supervision(
                subject_id=str(label_paths.get("subject_id", "validation_subject")),
                pet_paths={key: label_paths[key] for key in exact_pet_keys},
                source_ct_shape=gt_hu.shape,
                source_ct_affine=gt_img.affine,
                compact_shape=target_shape,
                cache_dir=pet_supervision_cfg.get("cache_dir"),
                exclusion_cm=exclusion_cm,
                mask_supersample_factor=pet_supervision_cfg.get(
                    "mask_supersample_factor", 2
                ),
            )
            activity_proxy = exact_pet["activity_suv"].numpy()[0]
        else:
            activity_proxy = load_nac_activity_proxy(
                nacpet_path,
                gt_hu.shape,
                body_mask=body_mask,
                max_relative_activity=float(
                    pet_proxy_cfg.get("activity_weight_clip", 20.0)
                ),
            )

        def as_volume_tensor(array, dtype=torch.float32):
            return (
                torch.from_numpy(np.asarray(array))
                .to(device=metric_device, dtype=dtype)
                .view(1, 1, *array.shape)
            )

        with torch.no_grad():
            pred_mu_proxy = as_volume_tensor(pred_mu_s)
            target_mu_proxy = as_volume_tensor(gt_mu_s)
            official_mask_proxy = as_volume_tensor(official_mask)
            prediction_mask_proxy = as_volume_tensor(prediction_mask)
            organ_proxy = (
                as_volume_tensor(organ_seg, dtype=torch.float32)
                if organ_seg is not None
                else None
            )
            body_region_proxy = (
                as_volume_tensor(body_region_seg, dtype=torch.float32)
                if body_region_seg is not None
                else None
            )
            target_hu_proxy = as_volume_tensor(gt_hu_eval)
            activity_proxy_tensor = as_volume_tensor(activity_proxy)
            exact_eval_mask_tensor = (
                exact_pet["eval_mask"].unsqueeze(0).to(metric_device)
                if exact_pet is not None
                else None
            )
            exact_region_masks_tensor = (
                exact_pet["region_masks"].unsqueeze(0).to(metric_device)
                if exact_pet is not None
                else None
            )
            activity_is_gt_pet = exact_pet is not None
            proxy_spacing_mm = list(spacing_mm)

            if (
                str(pet_proxy_cfg.get("training_mode", "")).lower()
                == "full_fov_patch_injected"
            ):
                native_shape = tuple(int(value) for value in pred_mu_proxy.shape[2:])

                def resize_proxy(x, mode):
                    if x is None:
                        return None
                    kwargs = {"size": target_shape, "mode": mode}
                    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
                        kwargs["align_corners"] = False
                    return F.interpolate(x.float(), **kwargs)

                pred_mu_proxy = resize_proxy(pred_mu_proxy, "area")
                target_mu_proxy = resize_proxy(target_mu_proxy, "area")
                official_mask_proxy = resize_proxy(official_mask_proxy, "nearest")
                prediction_mask_proxy = resize_proxy(prediction_mask_proxy, "nearest")
                organ_proxy = resize_proxy(organ_proxy, "nearest")
                body_region_proxy = resize_proxy(body_region_proxy, "nearest")
                target_hu_proxy = resize_proxy(target_hu_proxy, "area")
                if exact_pet is None:
                    activity_proxy_tensor = resize_proxy(activity_proxy_tensor, "trilinear")
                proxy_spacing_mm = [
                    float(spacing_mm[axis])
                    * float(native_shape[axis])
                    / float(target_shape[axis])
                    for axis in range(3)
                ]
                pet_proxy_cfg["angles_deg"] = pet_proxy_cfg.get(
                    "full_fov_angles_deg",
                    pet_proxy_cfg.get("angles_deg", []),
                )
                pet_proxy_cfg["downsample_factors"] = pet_proxy_cfg.get(
                    "full_fov_downsample_factors",
                    [1, 1, 1],
                )

            pet_proxy_losses, pet_proxy_diagnostics = (
                evaluator_aligned_pet_proxy_losses(
                    pred_mu=pred_mu_proxy,
                    target_mu=target_mu_proxy,
                    official_eval_mask=official_mask_proxy,
                    prediction_mask=prediction_mask_proxy,
                    organ_seg=organ_proxy,
                    body_region_seg=body_region_proxy,
                    target_hu=target_hu_proxy,
                    activity_proxy=activity_proxy_tensor,
                    spacing_mm=proxy_spacing_mm,
                    config=pet_proxy_cfg,
                    exact_eval_mask=exact_eval_mask_tensor,
                    exact_region_masks=exact_region_masks_tensor,
                    activity_is_gt_pet=activity_is_gt_pet,
                )
            )
        metrics.update(
            {
                "full_volume_pet_proxy_whole_body": float(
                    pet_proxy_losses["pet_whole_body"].item()
                ),
                "full_volume_pet_proxy_organ_bias": float(
                    pet_proxy_losses["pet_organ_bias"].item()
                ),
                "full_volume_pet_proxy_brain_outlier": float(
                    pet_proxy_losses["pet_brain_outlier"].item()
                ),
            }
        )
        for name, value in pet_proxy_diagnostics.items():
            if torch.is_tensor(value) and value.numel() == 1:
                metrics[f"full_volume_{name}"] = float(value.item())

    if organ_seg is not None:
        organ_mu_biases = []
        organ_mu_maes = []
        organ_hu_maes = []
        for label in sorted(int(v) for v in np.unique(organ_seg) if int(v) != 0):
            region = (organ_seg == label) & body_mask
            if int(np.count_nonzero(region)) < int(min_organ_voxels):
                continue
            mu_bias = abs(float(np.mean(pred_mu_s[region])) - float(np.mean(gt_mu_s[region])))
            mu_mae = float(np.mean(abs_mu_s[region]))
            hu_mae = float(np.mean(abs_hu[region]))
            organ_mu_biases.append(mu_bias)
            organ_mu_maes.append(mu_mae)
            organ_hu_maes.append(hu_mae)
            metrics[f"organ_mu_bias_label_{label}"] = mu_bias
            metrics[f"organ_mu_MAE_label_{label}"] = mu_mae
            metrics[f"organ_HU_MAE_label_{label}"] = hu_mae

        metrics["organ_bias_mean"] = float(np.mean(organ_mu_biases)) if organ_mu_biases else float("nan")
        metrics["organ_mu_MAE_mean"] = float(np.mean(organ_mu_maes)) if organ_mu_maes else float("nan")
        metrics["organ_HU_MAE_mean"] = float(np.mean(organ_hu_maes)) if organ_hu_maes else float("nan")
    else:
        metrics["organ_bias_mean"] = float("nan")
        metrics["organ_mu_MAE_mean"] = float("nan")
        metrics["organ_HU_MAE_mean"] = float("nan")

    return metrics


def predict_full_volume(subject: dict, cfg: dict, model, device: torch.device) -> np.ndarray:
    data = get_inference_transforms(cfg)(inference_features_only(subject))
    x = data["input"].unsqueeze(0)
    global_input = data.get("global_input")
    if global_input is not None:
        global_input = global_input.unsqueeze(0).to(device)
    metadata = get_metadata_tensor(subject, cfg, device)
    patch_size = tuple(cfg.get("inference", {}).get("patch_size", cfg.get("training", {}).get("patch_size", DEFAULT_PATCH_SIZE)))

    global_context = None
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        if getattr(model, "global_context_enabled", False):
            global_context = model.prepare_global_context(global_input)

    def predictor(window):
        window = window.to(device, non_blocking=True)
        meta = metadata
        if meta is not None and meta.shape[0] != window.shape[0]:
            meta = meta.expand(window.shape[0], -1)
        return model(window, metadata=meta, global_context=global_context)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        pred = sliding_window_inference(
            x,
            roi_size=patch_size,
            sw_batch_size=cfg.get("inference", {}).get("sw_batch_size", SW_BATCH),
            predictor=predictor,
            overlap=cfg.get("inference", {}).get("overlap", OVERLAP),
            mode="gaussian",
            progress=True,
            sw_device=device,
            device=torch.device("cpu"),
        )

    output_units = cfg.get("model", {}).get("output_units", "hu")
    if output_units == "normalized":
        pred_hu = pred.cpu().float().clamp(0.0, 1.0).numpy()[0, 0] * 3000.0 - 1000.0
    else:
        pred_hu = pred.cpu().float().numpy()[0, 0]
    return np.clip(pred_hu, -1000.0, 2000.0).astype(np.float32, copy=False)


def save_prediction(pred_hu: np.ndarray, features_dir: Path, out_path: Path):
    save_prediction_nifti(pred_hu, features_dir / "nacpet.nii.gz", out_path)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "subject_id",
        "ct_mu_map_mae_official_like",
        "mu_MAE_smoothed_raw",
        "mu_MAE_smoothed",
        "projection_mu_error",
        "projection_mu_error_D",
        "projection_mu_error_H",
        "projection_mu_error_W",
        "projection_mu_error_orth",
        "projection_mu_error_multi_angle",
        "full_volume_pet_proxy_whole_body",
        "full_volume_pet_proxy_organ_bias",
        "full_volume_pet_proxy_brain_outlier",
        "full_volume_logacf_error",
        "full_volume_relacf_error",
        "full_volume_logacf_error_D",
        "full_volume_logacf_error_H",
        "full_volume_logacf_error_W",
        "full_volume_logacf_error_orth",
        "full_volume_logacf_error_multi_angle",
        "full_volume_logacf_signed_bias",
        "full_volume_logacf_signed_mean",
        "full_volume_logacf_tail_cvar",
        "full_volume_logacf_p95",
        "full_volume_logacf_signed_bias_orth",
        "full_volume_logacf_signed_bias_multi_angle",
        "full_volume_head_logacf_error",
        "full_volume_head_relacf_error",
        "full_volume_head_logacf_signed_bias",
        "full_volume_head_logacf_signed_mean",
        "full_volume_head_logacf_tail_cvar",
        "full_volume_head_relacf_tail_cvar",
        "full_volume_head_logacf_p95",
        "robust_full_volume_logacf_error",
        "robust_full_volume_head_logacf_error",
        "robust_full_volume_head_logacf_tail_cvar",
        "full_volume_head_logacf_error_multi_angle",
        "full_volume_head_logacf_signed_bias_multi_angle",
        "full_volume_head_voxel_fraction",
        "organ_bias_mean",
        "HU_MAE_prediction_mask",
        "HU_MAE_body",
        "bone_MAE",
        "lung_MAE",
        "soft_tissue_MAE",
    ]
    fieldnames = [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict]) -> dict:
    aggregate = {"subject_id": "MEAN"}
    keys = sorted({key for row in rows for key in row.keys() if key != "subject_id"})
    for key in keys:
        values = []
        for row in rows:
            try:
                value = float(row[key])
            except Exception:
                continue
            if math.isfinite(value):
                values.append(value)
        aggregate[key] = float(np.mean(values)) if values else float("nan")
    return aggregate


def add_robust_aggregate_metrics(rows: list[dict], aggregate: dict) -> dict:
    """Add mean + 0.5 * worst-subject summaries for sparse internal val."""
    output = dict(aggregate)
    for metric_name in (
        "full_volume_logacf_error",
        "full_volume_relacf_error",
        "full_volume_head_logacf_error",
        "full_volume_head_relacf_error",
        "full_volume_head_logacf_tail_cvar",
        "full_volume_head_relacf_tail_cvar",
        "full_volume_pet_proxy_whole_body",
        "full_volume_pet_proxy_organ_bias",
        "full_volume_pet_proxy_brain_outlier",
    ):
        values = []
        for row in rows:
            try:
                value = float(row[metric_name])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        output[f"robust_{metric_name}"] = (
            float(np.mean(values) + 0.5 * np.max(values)) if values else float("nan")
        )
    return output


def main():
    parser = argparse.ArgumentParser(description="Full-volume internal validation for BIC-MAC pseudo-CT models.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pth")
    parser.add_argument("--config", default="config.yaml", help="Config YAML. Checkpoint config overrides matching fields.")
    parser.add_argument("--val_dir", default=None, help="Internal labeled val dir. Defaults to config data.val_dir.")
    parser.add_argument("--subjects", nargs="+", default=None, help="Subject IDs, e.g. sub-000 sub-001")
    parser.add_argument("--limit", type=int, default=None, help="Optional subject limit for smoke runs")
    parser.add_argument("--output_dir", default="outputs_full_volume_val", help="Output directory for metrics/predictions")
    parser.add_argument("--no_save_predictions", action="store_true", help="Do not save predicted CT NIfTI files")
    parser.add_argument("--metric_device", choices=["cpu", "cuda"], default="cpu", help="Fallback torch smoothing device if scipy is unavailable")
    parser.add_argument("--min_organ_voxels", type=int, default=DEFAULT_ORGAN_MIN_VOXELS)
    parser.add_argument("--exclusion_cm", type=float, default=4.0, help="Official-like liver-adjacent exclusion width in cm")
    args = parser.parse_args()

    file_cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_device = torch.device("cuda" if args.metric_device == "cuda" and torch.cuda.is_available() else "cpu")

    state_dict, checkpoint_cfg = load_checkpoint(Path(args.checkpoint), device)
    cfg = normalize_checkpoint_config(deep_merge(file_cfg, checkpoint_cfg))

    val_dir = Path(args.val_dir or cfg.get("data", {}).get("val_dir", "../data/val_labeled"))
    subjects = list_subjects(val_dir, args.subjects, args.limit)
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "predictions"

    model = build_model_from_checkpoint(cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Metric fallback device: {metric_device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Val dir: {val_dir}")
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Output dir: {output_dir}")

    rows = []
    for subject_id in tqdm(subjects, desc="Full-volume validation"):
        subject_dir = val_dir / subject_id
        features_dir = subject_dir / "features"
        ct_label_dir = subject_dir / "ct-label"
        subject = get_subject_features(str(features_dir))
        label_paths = get_subject_ct_labels(str(ct_label_dir))

        pred_hu = predict_full_volume(subject, cfg, model, device)
        if not args.no_save_predictions:
            save_prediction(pred_hu, features_dir, pred_dir / subject_id / "ct.nii.gz")

        metrics = compute_metrics_for_subject(
            pred_hu=pred_hu,
            label_paths=label_paths,
            cfg=cfg,
            metric_device=metric_device,
            min_organ_voxels=args.min_organ_voxels,
            exclusion_cm=args.exclusion_cm,
            nacpet_path=str(features_dir / "nacpet.nii.gz"),
        )
        row = {"subject_id": subject_id}
        row.update(metrics)
        rows.append(row)

        print(
            f"{subject_id}: "
            f"official_mu={row['ct_mu_map_mae_official_like']:.6f} "
            f"mu_s_raw={row['mu_MAE_smoothed_raw']:.6f} "
            f"mu_s_norm={row['mu_MAE_smoothed']:.5f} "
            f"proj={row['projection_mu_error']:.6f} "
            f"global_logacf={row['full_volume_logacf_error']:.6f} "
            f"global_signed={row.get('full_volume_logacf_signed_bias', float('nan')):.6f} "
            f"head_logacf={row.get('full_volume_head_logacf_error', float('nan')):.6f} "
            f"head_cvar={row.get('full_volume_head_logacf_tail_cvar', float('nan')):.6f} "
            f"pet_wb={row.get('full_volume_pet_proxy_whole_body', float('nan')):.6f} "
            f"pet_organ={row.get('full_volume_pet_proxy_organ_bias', float('nan')):.6f} "
            f"pet_brain={row.get('full_volume_pet_proxy_brain_outlier', float('nan')):.6f} "
            f"organ={row['organ_bias_mean']:.6f}"
        )

    aggregate = add_robust_aggregate_metrics(rows, aggregate_rows(rows))
    write_csv(output_dir / "full_volume_metrics_per_subject.csv", rows)
    write_csv(output_dir / "full_volume_metrics_aggregate.csv", [aggregate])

    print("\nAggregate:")
    for key in [
        "ct_mu_map_mae_official_like",
        "mu_MAE_smoothed_raw",
        "mu_MAE_smoothed",
        "projection_mu_error",
        "full_volume_logacf_error",
        "full_volume_logacf_signed_bias",
        "full_volume_head_logacf_error",
        "full_volume_head_logacf_signed_bias",
        "full_volume_pet_proxy_whole_body",
        "full_volume_pet_proxy_organ_bias",
        "full_volume_pet_proxy_brain_outlier",
        "organ_bias_mean",
        "HU_MAE_prediction_mask",
        "bone_MAE",
        "lung_MAE",
        "soft_tissue_MAE",
    ]:
        value = aggregate.get(key, float("nan"))
        print(f"  {key}: {value:.6f}")
    print(f"\nSaved metrics: {output_dir / 'full_volume_metrics_per_subject.csv'}")
    print(f"Saved aggregate: {output_dir / 'full_volume_metrics_aggregate.csv'}")


if __name__ == "__main__":
    main()
