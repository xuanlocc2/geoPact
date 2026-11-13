from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism
from tqdm import tqdm

from dataset import get_dataset
from full_volume_validate import (
    add_robust_aggregate_metrics,
    aggregate_rows,
    compute_metrics_for_subject,
    predict_full_volume,
    save_prediction,
    write_csv,
)
from losses import CombinedPseudoCTLoss
from transforms import DEFAULT_INPUT_CHANNELS, get_train_transforms, get_val_transforms
from unet import build_model


DEFAULT_METADATA_KEYS = ["sex", "age", "height", "weight", "bmi"]
LOG_KEYS = [
    "loss_total",
    "loss_hu_raw",
    "loss_grad_raw",
    "loss_mu_official_raw",
    "loss_mu_raw",
    "loss_proj_raw",
    "loss_organ_raw",
    "loss_global_mu_official_raw",
    "loss_global_mu_raw",
    "loss_global_acf_raw",
    "loss_global_acf_signed_raw",
    "loss_regional_relacf_raw",
    "loss_organ_ray_relacf_raw",
    "loss_organ_ray_signed_raw",
    "loss_local_head_acf_raw",
    "loss_local_head_acf_signed_raw",
    "loss_local_head_acf_tail_raw",
    "loss_pet_whole_body_raw",
    "loss_pet_organ_bias_raw",
    "loss_pet_brain_outlier_raw",
    "loss_hu_weighted",
    "loss_grad_weighted",
    "loss_mu_official_weighted",
    "loss_mu_weighted",
    "loss_proj_weighted",
    "loss_organ_weighted",
    "loss_global_mu_official_weighted",
    "loss_global_mu_weighted",
    "loss_global_acf_weighted",
    "loss_global_acf_signed_weighted",
    "loss_regional_relacf_weighted",
    "loss_organ_ray_signed_weighted",
    "loss_local_head_acf_weighted",
    "loss_local_head_acf_signed_weighted",
    "loss_local_head_acf_tail_weighted",
    "loss_pet_whole_body_weighted",
    "loss_pet_organ_bias_weighted",
    "loss_pet_brain_outlier_weighted",
    "loss_ref_hu",
    "loss_ref_grad",
    "loss_ref_mu_official",
    "loss_ref_mu",
    "loss_ref_proj",
    "loss_ref_organ",
    "loss_ref_global_mu_official",
    "loss_ref_global_mu",
    "loss_ref_global_acf",
    "loss_ref_global_acf_signed",
    "loss_ref_regional_relacf",
    "loss_ref_organ_ray_signed",
    "loss_ref_local_head_acf",
    "loss_ref_local_head_acf_signed",
    "loss_ref_local_head_acf_tail",
    "loss_ref_pet_whole_body",
    "loss_ref_pet_organ_bias",
    "loss_ref_pet_brain_outlier",
    "lambda_hu",
    "lambda_grad",
    "lambda_mu_official",
    "lambda_mu",
    "lambda_proj",
    "lambda_organ",
    "lambda_global_mu_official",
    "lambda_global_mu",
    "lambda_global_acf",
    "lambda_global_acf_signed",
    "lambda_regional_relacf",
    "lambda_organ_ray_signed",
    "lambda_local_head_acf",
    "lambda_local_head_acf_signed",
    "lambda_local_head_acf_tail",
    "lambda_pet_whole_body",
    "lambda_pet_organ_bias",
    "lambda_pet_brain_outlier",
    "pet_proxy_whole_body_suv_mae",
    "pet_proxy_organ_bias",
    "pet_proxy_brain_outlier",
    "pet_proxy_activity_mean",
    "pet_proxy_full_fov_patch_count",
    "pet_proxy_full_fov_coverage_fraction",
    "pet_proxy_full_fov_delta_mu_abs_mean",
    "ray_relacf_abs_mean",
    "ray_relacf_signed_mean",
    "response_abs_mean",
    "ray_activity_alpha",
    "ray_activity_weight_mean",
    "pet_proxy_uses_gt_pet",
    "pet_proxy_uses_exact_masks",
    "pet_proxy_brain_exceedance_05",
    "pet_proxy_brain_exceedance_10",
    "pet_proxy_brain_exceedance_15",
    "pet_proxy_region_present_brain",
    "pet_proxy_region_present_liver",
    "pet_proxy_region_present_spleen",
    "pet_proxy_region_present_heart",
    "pet_proxy_region_present_pancreas",
    "pet_proxy_region_present_muscle",
    "pet_proxy_region_present_adipose",
    "pet_proxy_region_present_extremities",
    "pet_proxy_region_abs_bias_brain",
    "pet_proxy_region_abs_bias_liver",
    "pet_proxy_region_abs_bias_spleen",
    "pet_proxy_region_abs_bias_heart",
    "pet_proxy_region_abs_bias_pancreas",
    "pet_proxy_region_abs_bias_muscle",
    "pet_proxy_region_abs_bias_adipose",
    "pet_proxy_region_abs_bias_extremities",
    "mu_MAE_official_raw",
    "HU_MAE_prediction_mask",
    "HU_MAE_body",
    "mu_MAE_smoothed",
    "projection_mu_error",
    "projection_mu_error_D",
    "projection_mu_error_H",
    "projection_mu_error_W",
    "projection_mu_error_orth",
    "projection_mu_error_multi_angle",
    "global_mu_MAE_official_raw",
    "global_mu_MAE_smoothed",
    "global_logacf_error",
    "global_logacf_signed_bias",
    "global_logacf_signed_mean",
    "global_logacf_tail_cvar",
    "global_logacf_p95",
    "global_logacf_error_D",
    "global_logacf_error_H",
    "global_logacf_error_W",
    "global_logacf_error_orth",
    "global_logacf_error_multi_angle",
    "global_logacf_signed_bias_orth",
    "global_logacf_signed_bias_multi_angle",
    "global_logacf_signed_bias_angle_m67p5",
    "global_logacf_signed_bias_angle_m45",
    "global_logacf_signed_bias_angle_m22p5",
    "global_logacf_signed_bias_angle_0",
    "global_logacf_signed_bias_angle_p22p5",
    "global_logacf_signed_bias_angle_p45",
    "global_logacf_signed_bias_angle_p67p5",
    "local_head_logacf_error",
    "local_head_logacf_signed_bias",
    "local_head_relacf_error",
    "local_head_relacf_signed_bias",
    "local_head_relacf_tail_cvar",
    "local_head_logacf_signed_mean",
    "local_head_logacf_tail_cvar",
    "local_head_logacf_p95",
    "local_head_logacf_error_multi_angle",
    "local_head_logacf_signed_bias_multi_angle",
    "local_head_logacf_signed_bias_angle_m45",
    "local_head_logacf_signed_bias_angle_0",
    "local_head_logacf_signed_bias_angle_p45",
    "local_head_sample_rate",
    "local_head_voxel_fraction",
    "local_head_transaxial_coverage",
    "organ_bias_mean",
    "regional_relacf_error",
    "organ_ray_relacf_error",
    "organ_ray_signed_bias",
    "bone_MAE",
    "lung_MAE",
    "soft_tissue_MAE",
    "crop_mask_fraction_mean",
    "crop_mask_fraction_min",
    "crop_mask_fraction_max",
    "crop_body_axis_fraction_mean",
    "crop_body_axis_fraction_min",
    "crop_body_axis_fraction_max",
    "crop_region_lower_body_rate",
    "crop_region_abdomen_pelvis_rate",
    "crop_region_thorax_rate",
    "crop_region_head_neck_rate",
    "region_prediction_mask_fraction_mean",
    "region_prediction_mask_fraction_min",
    "region_prediction_mask_fraction_max",
    "region_body_fraction_mean",
    "region_body_fraction_min",
    "region_body_fraction_max",
    "region_air_fraction_mean",
    "region_air_fraction_min",
    "region_air_fraction_max",
    "region_soft_fraction_mean",
    "region_soft_fraction_min",
    "region_soft_fraction_max",
    "region_bone_fraction_mean",
    "region_bone_fraction_min",
    "region_bone_fraction_max",
    "data_wait_seconds_per_batch",
    "step_seconds_per_batch",
]
FULL_VOLUME_LOG_KEYS = [
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
    "full_volume_pet_proxy_uses_gt_pet",
    "full_volume_pet_proxy_uses_exact_masks",
    "full_volume_ray_activity_alpha",
    "full_volume_ray_activity_weight_mean",
    "full_volume_pet_proxy_region_abs_bias_brain",
    "full_volume_pet_proxy_region_abs_bias_liver",
    "full_volume_pet_proxy_region_abs_bias_spleen",
    "full_volume_pet_proxy_region_abs_bias_heart",
    "full_volume_pet_proxy_region_abs_bias_pancreas",
    "full_volume_pet_proxy_region_abs_bias_muscle",
    "full_volume_pet_proxy_region_abs_bias_adipose",
    "full_volume_pet_proxy_region_abs_bias_extremities",
    "robust_full_volume_pet_proxy_whole_body",
    "robust_full_volume_pet_proxy_organ_bias",
    "robust_full_volume_pet_proxy_brain_outlier",
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
    "full_volume_relacf_tail_cvar",
    "full_volume_logacf_p95",
    "full_volume_logacf_signed_bias_orth",
    "full_volume_logacf_signed_bias_multi_angle",
    "full_volume_logacf_signed_bias_angle_m67p5",
    "full_volume_logacf_signed_bias_angle_m45",
    "full_volume_logacf_signed_bias_angle_m22p5",
    "full_volume_logacf_signed_bias_angle_0",
    "full_volume_logacf_signed_bias_angle_p22p5",
    "full_volume_logacf_signed_bias_angle_p45",
    "full_volume_logacf_signed_bias_angle_p67p5",
    "full_volume_head_logacf_error",
    "full_volume_head_relacf_error",
    "full_volume_head_logacf_signed_bias",
    "full_volume_head_logacf_signed_mean",
    "full_volume_head_logacf_tail_cvar",
    "full_volume_head_relacf_tail_cvar",
    "full_volume_head_logacf_p95",
    "robust_full_volume_logacf_error",
    "robust_full_volume_relacf_error",
    "robust_full_volume_head_logacf_error",
    "robust_full_volume_head_logacf_tail_cvar",
    "robust_full_volume_head_relacf_error",
    "robust_full_volume_head_relacf_tail_cvar",
    "full_volume_head_logacf_error_multi_angle",
    "full_volume_head_logacf_signed_bias_multi_angle",
    "full_volume_head_logacf_error_angle_m45",
    "full_volume_head_logacf_error_angle_0",
    "full_volume_head_logacf_error_angle_p45",
    "full_volume_head_logacf_signed_bias_angle_m45",
    "full_volume_head_logacf_signed_bias_angle_0",
    "full_volume_head_logacf_signed_bias_angle_p45",
    "full_volume_head_voxel_fraction",
    "full_volume_logacf_error_angle_m67p5",
    "full_volume_logacf_error_angle_m45",
    "full_volume_logacf_error_angle_m22p5",
    "full_volume_logacf_error_angle_0",
    "full_volume_logacf_error_angle_p22p5",
    "full_volume_logacf_error_angle_p45",
    "full_volume_logacf_error_angle_p67p5",
    "full_volume_logacf_grid_x",
    "full_volume_logacf_grid_y",
    "full_volume_logacf_grid_z",
    "full_volume_logacf_spacing_x_mm",
    "full_volume_logacf_spacing_y_mm",
    "full_volume_logacf_spacing_z_mm",
    "organ_bias_mean",
    "HU_MAE_prediction_mask",
    "HU_MAE_body",
    "bone_MAE",
    "lung_MAE",
    "soft_tissue_MAE",
    "bone_mu_MAE_raw",
    "lung_mu_MAE_raw",
    "soft_tissue_mu_MAE_raw",
    "organ_mu_MAE_mean",
    "organ_HU_MAE_mean",
    "eval_voxels",
    "official_eval_voxels",
]


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as handle:
        expanded = os.path.expandvars(handle.read())
    unresolved = sorted(set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", expanded)))
    if unresolved:
        raise ValueError("Set the required configuration environment variables: " + ", ".join(unresolved))
    cfg = yaml.safe_load(expanded)
    cfg.setdefault("input", {"channels": DEFAULT_INPUT_CHANNELS})
    channel_count = len(cfg["input"]["channels"])
    declared_channels = int(cfg["model"].get("in_channels", channel_count))
    if declared_channels != channel_count:
        print(f"Warning: model.in_channels={declared_channels} but input has {channel_count} channels; using {channel_count}.")
    cfg["model"]["in_channels"] = channel_count
    return cfg


def split_dataset(all_data, cfg):
    val_subjects = cfg["data"].get("val_subjects", [])
    include_monitor_in_train = bool(
        cfg["data"].get("include_val_subjects_in_train", False)
    )
    if val_subjects:
        val_set = set(val_subjects)
        val_data = [item for item in all_data if item["subject_id"] in val_set]
        train_data = (
            list(all_data)
            if include_monitor_in_train
            else [item for item in all_data if item["subject_id"] not in val_set]
        )
    else:
        val_count = cfg["data"].get("val_count", 2)
        val_data = all_data[:val_count]
        train_data = list(all_data) if include_monitor_in_train else all_data[val_count:]
    return train_data, val_data


def load_train_val_data(cfg):
    train_dir = cfg["data"].get("train_dir")
    val_dir = cfg["data"].get("val_dir")
    include_monitor_in_train = bool(
        cfg["data"].get("include_val_subjects_in_train", False)
    )

    if train_dir or val_dir:
        if not train_dir or not val_dir:
            raise RuntimeError("Both data.train_dir and data.val_dir must be set together")
        train_data = get_dataset(train_dir)
        val_data = get_dataset(val_dir)
        val_subjects = cfg["data"].get("val_subjects", [])
        if val_subjects:
            val_set = set(val_subjects)
            val_data = [item for item in val_data if item["subject_id"] in val_set]
            missing = sorted(val_set - {item["subject_id"] for item in val_data})
            if missing:
                raise RuntimeError(f"Configured val_subjects not found in data.val_dir: {missing}")
            if include_monitor_in_train:
                train_ids = {item["subject_id"] for item in train_data}
                train_data.extend(
                    item for item in val_data if item["subject_id"] not in train_ids
                )
            else:
                train_data = [item for item in train_data if item["subject_id"] not in val_set]
        return _limit_dataset(train_data, cfg["data"].get("max_train_subjects"), "train"), _limit_dataset(
            val_data, cfg["data"].get("max_val_subjects"), "val"
        )

    all_data = get_dataset(cfg["data"]["data_dir"])
    train_data, val_data = split_dataset(all_data, cfg)
    return _limit_dataset(train_data, cfg["data"].get("max_train_subjects"), "train"), _limit_dataset(
        val_data, cfg["data"].get("max_val_subjects"), "val"
    )


def _limit_dataset(data, limit, split_name: str):
    if limit is None:
        return data
    limit = int(limit)
    if limit <= 0:
        raise ValueError(f"data.max_{split_name}_subjects must be positive when set, got {limit}")
    return data[:limit]


def make_loader(data, transform, cfg, shuffle):
    data_cfg = cfg["data"]
    dataset = CacheDataset(
        data=data,
        transform=transform,
        cache_rate=data_cfg.get("cache_rate", 0.0),
        num_workers=data_cfg["num_workers"],
    )
    num_workers = int(data_cfg["num_workers"])
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
        persistent_workers=bool(data_cfg.get("persistent_workers", num_workers > 0)) and num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return DataLoader(**loader_kwargs)


def prepare_pet_supervision_sampling(train_data, val_data, cfg):
    """Validate PET labels and oversample them without dropping other subjects."""
    pet_cfg = cfg.get("pet_supervision", {})
    if not bool(pet_cfg.get("enabled", False)):
        return train_data

    train_with_pet = [item for item in train_data if item.get("has_pet_supervision", False)]
    val_with_pet = [item for item in val_data if item.get("has_pet_supervision", False)]
    train_ids = {item["subject_id"] for item in train_with_pet}
    val_ids = {item["subject_id"] for item in val_with_pet}
    expected_train = set(pet_cfg.get("expected_train_subjects", []))
    expected_val = set(pet_cfg.get("expected_val_subjects", []))
    strict = bool(pet_cfg.get("require_expected_subjects", True))
    missing_train = sorted(expected_train - train_ids)
    missing_val = sorted(expected_val - val_ids)
    if strict and (missing_train or missing_val):
        raise RuntimeError(
            "V4.5 exact PET supervision is incomplete. "
            f"Missing train={missing_train}, val={missing_val}. Download train/sub-*/pet-label/*."
        )
    monitor_name = (
        "train-monitor"
        if bool(cfg.get("data", {}).get("include_val_subjects_in_train", False))
        else "val"
    )
    print(
        "Exact PET supervision: "
        f"train={len(train_with_pet)}/{len(train_data)} "
        f"{monitor_name}={len(val_with_pet)}/{len(val_data)}"
    )

    oversample_cfg = pet_cfg.get("oversample", {})
    if not bool(oversample_cfg.get("enabled", False)) or not train_with_pet:
        return train_data
    target_fraction = float(oversample_cfg.get("target_fraction", 0.25))
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("pet_supervision.oversample.target_fraction must be in (0, 1)")
    current_positive = len(train_with_pet)
    total = len(train_data)
    extra_count = max(
        0,
        int(math.ceil((target_fraction * total - current_positive) / (1.0 - target_fraction))),
    )
    expanded = list(train_data)
    for index in range(extra_count):
        replica = dict(train_with_pet[index % len(train_with_pet)])
        replica["pet_oversample_replica"] = index + 1
        expanded.append(replica)
    effective_fraction = (current_positive + extra_count) / max(1, len(expanded))
    print(
        "PET-supervised sampling: "
        f"records={len(expanded)}, extra={extra_count}, "
        f"effective_fraction={effective_fraction:.3f}"
    )
    return expanded


def move_batch_to_device(batch, device, cfg=None):
    required_tensor_keys = {
        "input", "ct", "prediction_mask", "body_seg", "organ_seg", "official_eval_mask",
        "global_input", "global_target_mu_raw", "global_target_mu_smooth",
        "global_prediction_mask", "global_body_mask",
        "global_official_eval_mask", "global_organ_seg", "global_body_region_seg",
        "global_target_hu", "global_activity_proxy", "global_spacing_mm",
        "global_pet_eval_mask", "global_pet_body_mask", "global_pet_region_masks",
        "global_activity_is_gt_pet",
    }
    global_cfg = (cfg or {}).get("model", {}).get("global_context", {})
    reuse_global = bool(global_cfg.get("enabled", False) and global_cfg.get("reuse_per_subject", True))
    if reuse_global and int((cfg or {}).get("training", {}).get("batch_size", 1)) != 1:
        raise ValueError("global_context.reuse_per_subject requires training.batch_size=1")
    moved = {}
    for key, value in batch.items():
        should_move = key in required_tensor_keys or key.startswith("meta_")
        if reuse_global and key.startswith("global_") and torch.is_tensor(value) and value.ndim >= 1:
            value = value[:1]
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) and should_move else value
    return moved


def get_metadata_tensor(batch, cfg, device):
    metadata_cfg = cfg.get("metadata", {})
    if not metadata_cfg.get("use_film", False):
        return None
    keys = metadata_cfg.get("keys", DEFAULT_METADATA_KEYS)
    values = []
    batch_size = batch["input"].shape[0]
    for key in keys:
        batch_key = f"meta_{key}"
        value = batch.get(batch_key)
        if value is None:
            value = torch.zeros(batch_size, dtype=torch.float32, device=device)
        elif not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32, device=device)
        else:
            value = value.to(device=device, dtype=torch.float32)
        values.append(value.view(batch_size, 1))
    return torch.cat(values, dim=1)


def get_global_input_tensor(batch, cfg):
    """Return one whole-body overview for the patches of one subject.

    MONAI flattens the ``train_num_samples`` list into the effective batch.
    With data-loader batch_size=1 those patches all belong to the same subject,
    so encoding the repeated global tensor once saves substantial accelerator memory.
    """
    global_input = batch.get("global_input")
    global_cfg = cfg.get("model", {}).get("global_context", {})
    if global_input is None or not global_cfg.get("enabled", False):
        return global_input
    if global_cfg.get("reuse_per_subject", True):
        if int(cfg.get("training", {}).get("batch_size", 1)) != 1:
            raise ValueError("global_context.reuse_per_subject requires training.batch_size=1")
        return global_input[:1]
    return global_input


def init_csv(path: Path, overwrite: bool = True):
    if path.exists() and not overwrite:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = [
            "epoch", "split", "lr", "grad_norm",
            "gpu_mem_alloc_mb", "gpu_mem_reserved_mb",
            "gpu_mem_peak_alloc_mb", "gpu_mem_peak_reserved_mb",
        ] + LOG_KEYS
        writer.writerow(columns)


def append_csv(path: Path, epoch: int, split: str, lr: float, metrics: dict):
    with open(path, "a", newline="", encoding="utf-8") as handle:
        row = [
            epoch,
            split,
            lr,
            metrics.get("grad_norm", 0.0),
            metrics.get("gpu_mem_alloc_mb", 0.0),
            metrics.get("gpu_mem_reserved_mb", 0.0),
            metrics.get("gpu_mem_peak_alloc_mb", 0.0),
            metrics.get("gpu_mem_peak_reserved_mb", 0.0),
        ]
        row.extend(metrics.get(key, 0.0) for key in LOG_KEYS)
        csv.writer(handle).writerow(row)


def init_step_progress_csv(path: Path, overwrite: bool = True):
    if path.exists() and not overwrite:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "step",
                "steps_per_epoch",
                "lr",
                "grad_norm",
                "data_wait_seconds_avg",
                "step_seconds_avg",
                "gpu_mem_alloc_mb",
                "gpu_mem_reserved_mb",
                "gpu_mem_peak_alloc_mb",
                "gpu_mem_peak_reserved_mb",
                "emergency_checkpoint_saved",
                "loss_total",
                "loss_hu_raw",
                "loss_mu_official_raw",
                "loss_mu_raw",
                "loss_proj_raw",
                "loss_global_acf_raw",
                "loss_global_acf_signed_raw",
                "loss_local_head_acf_raw",
                "loss_local_head_acf_signed_raw",
                "loss_pet_whole_body_raw",
                "loss_pet_organ_bias_raw",
                "loss_pet_brain_outlier_raw",
                "mu_MAE_smoothed",
                "projection_mu_error",
                "crop_mask_fraction_mean",
                "region_body_fraction_mean",
                "region_bone_fraction_mean",
                "region_air_fraction_mean",
                "region_soft_fraction_mean",
                "local_head_sample_rate",
                "local_head_transaxial_coverage",
            ]
        )


def append_step_progress_csv(
    path: Path,
    epoch: int,
    step: int,
    steps_per_epoch: int,
    lr: float,
    grad_norm: float,
    data_wait_seconds_avg: float,
    step_seconds_avg: float,
    metrics: dict,
    gpu_metrics: dict,
    emergency_saved: bool,
):
    with open(path, "a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            [
                epoch,
                step,
                steps_per_epoch,
                lr,
                grad_norm,
                data_wait_seconds_avg,
                step_seconds_avg,
                gpu_metrics.get("gpu_mem_alloc_mb", 0.0),
                gpu_metrics.get("gpu_mem_reserved_mb", 0.0),
                gpu_metrics.get("gpu_mem_peak_alloc_mb", 0.0),
                gpu_metrics.get("gpu_mem_peak_reserved_mb", 0.0),
                int(emergency_saved),
                metrics.get("loss_total", float("nan")),
                metrics.get("loss_hu_raw", float("nan")),
                metrics.get("loss_mu_official_raw", float("nan")),
                metrics.get("loss_mu_raw", float("nan")),
                metrics.get("loss_proj_raw", float("nan")),
                metrics.get("loss_global_acf_raw", float("nan")),
                metrics.get("loss_global_acf_signed_raw", float("nan")),
                metrics.get("loss_local_head_acf_raw", float("nan")),
                metrics.get("loss_local_head_acf_signed_raw", float("nan")),
                metrics.get("loss_pet_whole_body_raw", float("nan")),
                metrics.get("loss_pet_organ_bias_raw", float("nan")),
                metrics.get("loss_pet_brain_outlier_raw", float("nan")),
                metrics.get("mu_MAE_smoothed", float("nan")),
                metrics.get("projection_mu_error", float("nan")),
                metrics.get("crop_mask_fraction_mean", float("nan")),
                metrics.get("region_body_fraction_mean", float("nan")),
                metrics.get("region_bone_fraction_mean", float("nan")),
                metrics.get("region_air_fraction_mean", float("nan")),
                metrics.get("region_soft_fraction_mean", float("nan")),
                metrics.get("local_head_sample_rate", float("nan")),
                metrics.get("local_head_transaxial_coverage", float("nan")),
            ]
        )


def init_full_volume_csv(path: Path, overwrite: bool = True):
    header = ["epoch", "subject_id"] + FULL_VOLUME_LOG_KEYS
    should_write_header = True
    if path.exists() and not overwrite:
        with open(path, "r", newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            should_write_header = True
        elif rows[0] == header:
            return
        else:
            old_header = rows[0]
            migrated = [header]
            for row in rows[1:]:
                values = {key: row[index] for index, key in enumerate(old_header) if index < len(row)}
                migrated.append([values.get(key, "") for key in header])
            with open(path, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(migrated)
            return
    if not should_write_header:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)


def append_full_volume_csv(path: Path, epoch: int, rows: list[dict], aggregate: dict):
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in rows + [aggregate]:
            writer.writerow([epoch, row.get("subject_id", "")] + [row.get(key, float("nan")) for key in FULL_VOLUME_LOG_KEYS])


def init_checkpoint_index(path: Path, overwrite: bool = True):
    if path.exists() and not overwrite:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "filename", "score_name", "score_value", "bytes", "path"])


def append_checkpoint_index(
    path: Path,
    epoch: int,
    filename: str,
    score_name: str,
    score_value: float,
    bytes_written: int,
    checkpoint_path: Path,
):
    with open(path, "a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            [epoch, filename, score_name, score_value, bytes_written, str(checkpoint_path)]
        )


def is_finite_metric(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def init_metric_accumulators():
    return {key: 0.0 for key in LOG_KEYS}, {key: 0 for key in LOG_KEYS}


def update_metric_accumulators(sums: dict, counts: dict, values: dict):
    for key in LOG_KEYS:
        value = float(values.get(key, float("nan")))
        if math.isfinite(value):
            sums[key] += value
            counts[key] += 1


def finalize_metric_accumulators(sums: dict, counts: dict):
    return {key: sums[key] / counts[key] if counts[key] > 0 else float("nan") for key in sums}


def batch_crop_mask_fraction(batch):
    value = batch.get("train_crop_mask_fraction")
    if value is None:
        value = batch.get("val_crop_mask_fraction")
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32)
    value = value.detach().float().view(-1)
    value = value[torch.isfinite(value)]
    if value.numel() == 0:
        return None
    return {
        "crop_mask_fraction_mean": float(value.mean().item()),
        "crop_mask_fraction_min": float(value.min().item()),
        "crop_mask_fraction_max": float(value.max().item()),
    }


def batch_crop_region_metrics(batch, cfg):
    metrics = {}
    axis_fraction = batch.get("train_crop_body_axis_fraction")
    if axis_fraction is not None:
        values = torch.as_tensor(axis_fraction).detach().float().view(-1)
        values = values[torch.isfinite(values)]
        if values.numel() > 0:
            metrics.update(
                {
                    "crop_body_axis_fraction_mean": float(values.mean().item()),
                    "crop_body_axis_fraction_min": float(values.min().item()),
                    "crop_body_axis_fraction_max": float(values.max().item()),
                }
            )

    region_ids = batch.get("train_crop_region_id")
    regions = cfg.get("training", {}).get("region_sampling", {}).get("regions", [])
    if region_ids is None or not regions:
        return metrics
    region_ids = torch.as_tensor(region_ids).detach().view(-1)
    denominator = max(int(region_ids.numel()), 1)
    for region_id, region in enumerate(regions):
        name = str(region.get("name", f"region_{region_id}"))
        safe_name = "".join(character if character.isalnum() else "_" for character in name.lower())
        metrics[f"crop_region_{safe_name}_rate"] = float((region_ids == region_id).sum().item()) / denominator
    return metrics


def _fraction_stats(prefix: str, mask: torch.Tensor | None, batch_size: int):
    if mask is None:
        return {}
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.detach().float()
    while mask.ndim < 5:
        mask = mask.unsqueeze(0)
    if mask.shape[0] != batch_size:
        mask = mask.reshape(batch_size, -1)
    else:
        mask = mask.reshape(batch_size, -1)
    values = mask.mean(dim=1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {}
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
    }


def batch_region_fractions(batch, cfg):
    if "input" not in batch:
        return {}

    batch_size = int(batch["input"].shape[0])
    metrics = {}

    prediction_mask = batch.get("prediction_mask")
    body_seg = batch.get("body_seg")
    ct = batch.get("ct")

    pred_bool = (prediction_mask > 0) if torch.is_tensor(prediction_mask) else None
    body_bool = (body_seg > 0) if torch.is_tensor(body_seg) else None
    valid_bool = pred_bool if pred_bool is not None else body_bool

    metrics.update(_fraction_stats("region_prediction_mask_fraction", pred_bool, batch_size))
    metrics.update(_fraction_stats("region_body_fraction", body_bool, batch_size))

    if torch.is_tensor(ct):
        loss_cfg = cfg.get("loss", {})
        air_threshold = float(loss_cfg.get("air_threshold", -300.0))
        bone_threshold = float(loss_cfg.get("bone_threshold", 150.0))
        if valid_bool is None:
            valid_bool = torch.ones_like(ct, dtype=torch.bool)
        else:
            valid_bool = valid_bool.to(device=ct.device)

        air_mask = (ct < air_threshold) & valid_bool
        soft_mask = (ct >= air_threshold) & (ct <= bone_threshold) & valid_bool
        bone_mask = (ct > bone_threshold) & valid_bool

        metrics.update(_fraction_stats("region_air_fraction", air_mask, batch_size))
        metrics.update(_fraction_stats("region_soft_fraction", soft_mask, batch_size))
        metrics.update(_fraction_stats("region_bone_fraction", bone_mask, batch_size))

    return metrics


def metrics_to_float(loss_dict):
    return {key: float(value.detach().item()) if torch.is_tensor(value) else float(value) for key, value in loss_dict.items()}


def current_gpu_metrics(device):
    if device.type != "cuda":
        return {
            "gpu_mem_alloc_mb": 0.0,
            "gpu_mem_reserved_mb": 0.0,
            "gpu_mem_peak_alloc_mb": 0.0,
            "gpu_mem_peak_reserved_mb": 0.0,
        }
    return {
        "gpu_mem_alloc_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "gpu_mem_reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        "gpu_mem_peak_alloc_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "gpu_mem_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
    }


def compute_grad_norm(model) -> float:
    total_sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().data.norm(2).item()
        total_sq += grad_norm * grad_norm
    return total_sq**0.5


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    use_amp,
    epoch: int,
    cfg,
    checkpoints_dir: Path | None = None,
    step_progress_csv: Path | None = None,
):
    model.train()
    criterion.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metric_sums, metric_counts = init_metric_accumulators()
    grad_accum_steps = max(1, int(cfg["training"].get("grad_accum_steps", 1)))
    grad_clip = float(cfg["training"].get("grad_clip", 0.0))
    emergency_every = int(cfg["training"].get("emergency_save_interval_steps", 0) or 0)
    optimizer.zero_grad(set_to_none=True)
    grad_norm_value = 0.0
    data_wait_seconds = 0.0
    step_seconds = 0.0
    previous_step_end = time.perf_counter()

    pbar = tqdm(loader, leave=False)
    for step, raw_batch in enumerate(pbar, start=1):
        step_start = time.perf_counter()
        data_wait_seconds += step_start - previous_step_end
        batch = move_batch_to_device(raw_batch, device, cfg)
        x = batch["input"]
        assert x.ndim == 5, x.shape
        assert x.shape[1] == cfg["model"]["in_channels"], (x.shape, cfg["model"]["in_channels"])

        with torch.amp.autocast("cuda", enabled=use_amp):
            model_output = model(
                x,
                metadata=get_metadata_tensor(batch, cfg, device),
                global_input=get_global_input_tensor(batch, cfg),
                return_aux=True,
            )
            pred = model_output["ct"] if isinstance(model_output, dict) else model_output
            aux_outputs = model_output if isinstance(model_output, dict) else None
            loss, loss_dict = criterion(pred, batch, epoch, aux_outputs=aux_outputs)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()
        should_step = step % grad_accum_steps == 0 or step == len(loader)
        if should_step:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # Preserve the old post-clipping log semantics without a second
                # full parameter traversal and hundreds of GPU synchronizations.
                grad_norm_value = min(float(total_norm.detach().item()), grad_clip)
            else:
                scaler.unscale_(optimizer)
                grad_norm_value = compute_grad_norm(model)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        float_metrics = metrics_to_float(loss_dict)
        crop_metrics = batch_crop_mask_fraction(batch)
        if crop_metrics:
            float_metrics.update(crop_metrics)
        float_metrics.update(batch_crop_region_metrics(batch, cfg))
        float_metrics.update(batch_region_fractions(batch, cfg))
        update_metric_accumulators(metric_sums, metric_counts, float_metrics)
        crop_desc = ""
        if crop_metrics:
            crop_desc = f" mask {crop_metrics['crop_mask_fraction_mean']:.2f}"
        pbar.set_description(
            f"train {float_metrics['loss_total']:.3f} mu {float_metrics['mu_MAE_smoothed']:.4f}{crop_desc}"
        )
        step_end = time.perf_counter()
        step_seconds += step_end - step_start
        previous_step_end = step_end
        pbar.set_postfix(load=f"{data_wait_seconds / step:.1f}s", step=f"{step_seconds / step:.1f}s")

        emergency_saved = False
        if (
            checkpoints_dir is not None
            and emergency_every > 0
            and (step % emergency_every == 0 or step == len(loader))
        ):
            emergency_path = checkpoints_dir / "last_model.pth"
            # Store epoch-1 for partial-epoch checkpoints. Resume will restart
            # the current epoch using the latest partial weights, avoiding an
            # unsafe scheduler/epoch skip after interruption.
            emergency_epoch = max(0, epoch - 1)
            save_checkpoint(
                emergency_path,
                emergency_epoch,
                model,
                optimizer,
                scheduler,
                cfg,
                "emergency_train_step",
                float(step),
                criterion,
                scaler,
            )
            emergency_saved = True
            print(
                f"Emergency saved last_model.pth: current_epoch={epoch} "
                f"resume_epoch={emergency_epoch} step={step}/{len(loader)}",
                flush=True,
            )

        if step_progress_csv is not None:
            append_step_progress_csv(
                step_progress_csv,
                epoch,
                step,
                len(loader),
                optimizer.param_groups[0]["lr"],
                grad_norm_value,
                data_wait_seconds / step,
                step_seconds / step,
                float_metrics,
                current_gpu_metrics(device),
                emergency_saved,
            )

    averaged = finalize_metric_accumulators(metric_sums, metric_counts)
    averaged["grad_norm"] = grad_norm_value
    averaged["data_wait_seconds_per_batch"] = data_wait_seconds / max(1, len(loader))
    averaged["step_seconds_per_batch"] = step_seconds / max(1, len(loader))
    averaged.update(current_gpu_metrics(device))
    return averaged


@torch.no_grad()
def run_validation(model, loader, criterion, device, use_amp, epoch: int, cfg):
    model.eval()
    criterion.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metric_sums, metric_counts = init_metric_accumulators()
    data_wait_seconds = 0.0
    step_seconds = 0.0
    previous_step_end = time.perf_counter()

    for raw_batch in loader:
        step_start = time.perf_counter()
        data_wait_seconds += step_start - previous_step_end
        batch = move_batch_to_device(raw_batch, device, cfg)
        x = batch["input"]
        assert x.ndim == 5, x.shape
        assert x.shape[1] == cfg["model"]["in_channels"], (x.shape, cfg["model"]["in_channels"])
        with torch.amp.autocast("cuda", enabled=use_amp):
            model_output = model(
                x,
                metadata=get_metadata_tensor(batch, cfg, device),
                global_input=get_global_input_tensor(batch, cfg),
                return_aux=True,
            )
            pred = model_output["ct"] if isinstance(model_output, dict) else model_output
            aux_outputs = model_output if isinstance(model_output, dict) else None
            _, loss_dict = criterion(pred, batch, epoch, aux_outputs=aux_outputs)

        float_metrics = metrics_to_float(loss_dict)
        crop_metrics = batch_crop_mask_fraction(batch)
        if crop_metrics:
            float_metrics.update(crop_metrics)
        float_metrics.update(batch_crop_region_metrics(batch, cfg))
        float_metrics.update(batch_region_fractions(batch, cfg))
        update_metric_accumulators(metric_sums, metric_counts, float_metrics)
        step_end = time.perf_counter()
        step_seconds += step_end - step_start
        previous_step_end = step_end

    averaged = finalize_metric_accumulators(metric_sums, metric_counts)
    averaged["data_wait_seconds_per_batch"] = data_wait_seconds / max(1, len(loader))
    averaged["step_seconds_per_batch"] = step_seconds / max(1, len(loader))
    averaged.update(current_gpu_metrics(device))
    return averaged


@torch.no_grad()
def run_full_volume_validation(model, val_data, cfg, device, epoch: int, output_dir: Path):
    fv_cfg = cfg.get("full_volume_validation", {})
    selected = val_data
    monitor_name = (
        "train-monitor"
        if bool(cfg.get("data", {}).get("include_val_subjects_in_train", False))
        else "validation"
    )

    subjects = fv_cfg.get("subjects") or []
    if subjects:
        selected_set = set(subjects)
        selected = [item for item in selected if item["subject_id"] in selected_set]

    max_subjects = fv_cfg.get("max_subjects")
    if max_subjects is not None:
        selected = selected[: int(max_subjects)]

    if not selected:
        raise RuntimeError(
            f"Full-volume {monitor_name} is enabled but selected subject list is empty"
        )

    was_training = model.training
    model.eval()

    metric_device_name = fv_cfg.get("metric_device", "cpu")
    metric_device = torch.device("cuda" if metric_device_name == "cuda" and torch.cuda.is_available() else "cpu")
    min_organ_voxels = int(fv_cfg.get("min_organ_voxels", 64))
    exclusion_cm = float(fv_cfg.get("exclusion_cm", 4.0))
    save_predictions = bool(fv_cfg.get("save_predictions", False))

    epoch_dir = output_dir / f"epoch_{epoch:04d}"
    pred_dir = epoch_dir / "predictions"
    rows = []

    print(
        f"Running full-volume {monitor_name} at epoch {epoch} "
        f"on {len(selected)} subject(s)..."
    )
    for item in selected:
        subject_id = item["subject_id"]
        pred_hu = predict_full_volume(item, cfg, model, device)

        if save_predictions:
            features_dir = Path(item["nacpet"]).parent
            save_prediction(pred_hu, features_dir, pred_dir / subject_id / "ct.nii.gz")

        label_paths = {
            key: item[key]
            for key in (
                "ct", "prediction_mask", "body_seg", "organ_seg",
                "pet_gt", "pet_body_seg", "pet_organ_seg", "pet_tissue_seg",
            )
            if key in item
        }
        label_paths["subject_id"] = subject_id
        metrics = compute_metrics_for_subject(
            pred_hu=pred_hu,
            label_paths=label_paths,
            cfg=cfg,
            metric_device=metric_device,
            min_organ_voxels=min_organ_voxels,
            exclusion_cm=exclusion_cm,
            nacpet_path=item.get("nacpet"),
        )
        row = {"subject_id": subject_id}
        row.update(metrics)
        rows.append(row)
        print(
            f"  {subject_id}: "
            f"official_mu={row['ct_mu_map_mae_official_like']:.6f} "
            f"mu_s_raw={row['mu_MAE_smoothed_raw']:.6f} "
            f"proj={row['projection_mu_error']:.6f} "
            f"(D/H/W={row['projection_mu_error_D']:.6f}/"
            f"{row['projection_mu_error_H']:.6f}/{row['projection_mu_error_W']:.6f}) "
            f"global_logacf={row['full_volume_logacf_error']:.6f} "
            f"global_signed={row.get('full_volume_logacf_signed_bias', float('nan')):.6f} "
            f"head_logacf={row.get('full_volume_head_logacf_error', float('nan')):.6f} "
            f"pet_wb={row.get('full_volume_pet_proxy_whole_body', float('nan')):.6f} "
            f"pet_organ={row.get('full_volume_pet_proxy_organ_bias', float('nan')):.6f} "
            f"pet_brain={row.get('full_volume_pet_proxy_brain_outlier', float('nan')):.6f} "
            f"organ={row['organ_bias_mean']:.6f}"
        )

    aggregate = add_robust_aggregate_metrics(rows, aggregate_rows(rows))
    write_csv(epoch_dir / "full_volume_metrics_per_subject.csv", rows)
    write_csv(epoch_dir / "full_volume_metrics_aggregate.csv", [aggregate])

    if was_training:
        model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"Full-volume epoch {epoch:03d}: "
        f"official_mu={aggregate['ct_mu_map_mae_official_like']:.6f} "
        f"mu_s_raw={aggregate['mu_MAE_smoothed_raw']:.6f} "
        f"proj={aggregate['projection_mu_error']:.6f} "
        f"(D/H/W={aggregate['projection_mu_error_D']:.6f}/"
        f"{aggregate['projection_mu_error_H']:.6f}/{aggregate['projection_mu_error_W']:.6f}) "
        f"global_logacf={aggregate['full_volume_logacf_error']:.6f} "
        f"global_signed={aggregate.get('full_volume_logacf_signed_bias', float('nan')):.6f} "
        f"head_logacf={aggregate.get('full_volume_head_logacf_error', float('nan')):.6f} "
        f"pet_wb={aggregate.get('full_volume_pet_proxy_whole_body', float('nan')):.6f} "
        f"pet_organ={aggregate.get('full_volume_pet_proxy_organ_bias', float('nan')):.6f} "
        f"pet_brain={aggregate.get('full_volume_pet_proxy_brain_outlier', float('nan')):.6f} "
        f"organ={aggregate['organ_bias_mean']:.6f}"
    )
    upload_full_volume_epoch(epoch_dir, cfg, epoch, [row["subject_id"] for row in rows])
    return rows, aggregate


def upload_full_volume_epoch(epoch_dir: Path, cfg: dict, epoch: int, subjects: list[str]):
    """Upload one completed full-volume validation epoch to Hugging Face.

    Authentication is resolved by huggingface_hub from HF_TOKEN or its cached
    login. Tokens are never stored in the experiment config or output files.
    """
    fv_cfg = cfg.get("full_volume_validation", {})
    upload_cfg = fv_cfg.get("hf_upload", {})
    if not bool(upload_cfg.get("enabled", False)):
        return None

    repo_id = str(upload_cfg.get("repo_id", "")).strip()
    if not repo_id:
        raise ValueError("full_volume_validation.hf_upload.repo_id is required when upload is enabled")

    repo_type = str(upload_cfg.get("repo_type", "dataset"))
    prefix = str(upload_cfg.get("path_prefix", "outputs/v4_5_dgx")).strip("/")
    path_in_repo = f"{prefix}/full_volume_val/{epoch_dir.name}"
    fail_on_error = bool(upload_cfg.get("fail_on_error", False))

    manifest = {
        "experiment_version": cfg.get("experiment", {}).get("version"),
        "recipe_revision": cfg.get("experiment", {}).get("recipe_revision"),
        "epoch": int(epoch),
        "subjects": list(subjects),
        "remote_path": path_in_repo,
    }
    with (epoch_dir / "hf_upload_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)

    try:
        from huggingface_hub import HfApi

        token_env = str(upload_cfg.get("token_env", "HF_TOKEN"))
        token = os.environ.get(token_env) or None
        api = HfApi(token=token)
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=str(epoch_dir),
            path_in_repo=path_in_repo,
            allow_patterns=[
                "predictions/**",
                "full_volume_metrics_*.csv",
                "hf_upload_manifest.yaml",
            ],
            commit_message=(
                f"Upload V4.5 DGX full-volume epoch {epoch:04d} "
                f"for early PET reconstruction"
            ),
        )
        print(f"Uploaded full-volume epoch {epoch:04d} to hf://{repo_type}s/{repo_id}/{path_in_repo}")
        return path_in_repo
    except Exception as exc:
        message = f"Hugging Face upload failed at epoch {epoch:04d}: {type(exc).__name__}: {exc}"
        print(f"Warning: {message}")
        (epoch_dir / "hf_upload_error.txt").write_text(message + "\n", encoding="utf-8")
        if fail_on_error:
            raise
        return None


def _criterion_state(criterion):
    return {
        "step": int(getattr(criterion, "_step", 0)),
        "ref_sums": dict(getattr(criterion, "_ref_sums", {})),
        "ref_counts": dict(getattr(criterion, "_ref_counts", {})),
    }


def _load_criterion_state(criterion, state):
    if not isinstance(state, dict):
        return
    criterion._step = int(state.get("step", getattr(criterion, "_step", 0)))
    criterion._ref_sums = dict(state.get("ref_sums", getattr(criterion, "_ref_sums", {})))
    criterion._ref_counts = dict(state.get("ref_counts", getattr(criterion, "_ref_counts", {})))


def save_checkpoint(
    path: Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    cfg,
    score_name: str,
    score_value: float,
    criterion=None,
    scaler=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg,
        "score_name": score_name,
        "score_value": score_value,
    }
    if criterion is not None:
        payload["criterion_state"] = _criterion_state(criterion)
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"Checkpoint was written as an empty file: {path}")
    return size


def resolve_resume_path(resume_arg: str | None, checkpoints_dir: Path) -> Path | None:
    if not resume_arg:
        return None
    if resume_arg == "auto":
        auto_path = checkpoints_dir / "last_model.pth"
        return auto_path if auto_path.exists() else None
    return Path(resume_arg).expanduser()


def load_resume_checkpoint(path: Path, model, optimizer, scheduler, criterion, scaler, device, cfg: dict) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    # Keep the serialized optimizer/model payload off GPU while validating the
    # recipe and restoring it. Optimizer.load_state_dict moves state tensors to
    # their parameter devices and this avoids an unnecessary resume-time peak.
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported checkpoint format for resume: {path}")

    current_revision = cfg.get("experiment", {}).get("recipe_revision")
    saved_revision = checkpoint.get("config", {}).get("experiment", {}).get("recipe_revision")
    if current_revision and saved_revision != current_revision:
        raise RuntimeError(
            "Refusing to resume across recipe revisions: "
            f"checkpoint={saved_revision!r}, current={current_revision!r}. "
            "Use a fresh output directory for the P0-fixed objective."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "criterion_state" in checkpoint:
        _load_criterion_state(criterion, checkpoint["criterion_state"])
    else:
        print("Warning: resume checkpoint has no criterion_state; loss refs will continue from current defaults.")
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    print(f"Resumed checkpoint: {path} (epoch={epoch})")
    return epoch


def restore_best_scores_from_index(path: Path, best_scores: dict[str, float]) -> dict[str, float]:
    if not path.exists():
        return best_scores
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                score_name = row.get("score_name")
                if score_name not in best_scores:
                    continue
                try:
                    score_value = float(row.get("score_value", "nan"))
                except ValueError:
                    continue
                if is_finite_metric(score_value):
                    best_scores[score_name] = min(best_scores[score_name], score_value)
    except OSError as exc:
        print(f"Warning: could not restore best scores from {path}: {exc}")
    return best_scores


def plot_losses(history: dict[str, list[float]], output_path: Path):
    plt.figure(figsize=(8, 5))
    for key, values in history.items():
        if values:
            plt.plot(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Monitor tracked metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def maybe_write_effective_config(cfg, output_dir: Path):
    with open(output_dir / "effective_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_v4_5.yaml")))
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from checkpoint path, or use 'auto' to load output_dir/checkpoints/last_model.pth if present.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg["model"]["in_channels"] != len(cfg["input"]["channels"]):
        raise RuntimeError(
            f"model.in_channels={cfg['model']['in_channels']} but input.channels has "
            f"{len(cfg['input']['channels'])}: {cfg['input']['channels']}"
        )

    seed = cfg["training"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["training"].get("amp", True) and device.type == "cuda"

    train_data, val_data = load_train_val_data(cfg)
    if not train_data:
        raise RuntimeError("Training split is empty")
    if not val_data:
        raise RuntimeError("Validation/monitor split is empty")

    monitor_is_leaked = bool(
        cfg.get("data", {}).get("include_val_subjects_in_train", False)
    )

    print(f"Using device: {device}")
    print(f"Input channels ({len(cfg['input']['channels'])}): {cfg['input']['channels']}")
    print(f"Train subjects: {len(train_data)}")
    if monitor_is_leaked:
        print(f"Train-monitor subjects (also receive gradients): {len(val_data)}")
        print("Train-monitor IDs:", ", ".join(item["subject_id"] for item in val_data))
        print(
            "WARNING: train-monitor metrics are training-leaked optimization "
            "diagnostics, not an unbiased generalization estimate."
        )
    else:
        print(f"Val subjects: {len(val_data)}")
        print("Val IDs:", ", ".join(item["subject_id"] for item in val_data))

    sampled_train_data = prepare_pet_supervision_sampling(train_data, val_data, cfg)
    train_loader = make_loader(
        sampled_train_data,
        get_train_transforms(cfg),
        cfg,
        shuffle=True,
    )
    val_loader = make_loader(val_data, get_val_transforms(cfg), cfg, shuffle=False)

    model = build_model(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
        dropout=cfg["model"]["dropout"],
        metadata_dim=len(cfg.get("metadata", {}).get("keys", DEFAULT_METADATA_KEYS)),
        use_metadata_film=cfg.get("metadata", {}).get("use_film", False),
        model_name=cfg["model"].get("name", "mdixon_resunet3d"),
        geometry_in_channels=cfg["model"].get("geometry_in_channels"),
        geometry_channel_indices=cfg["model"].get("geometry_channel_indices"),
        topogram_channel_index=cfg["model"].get("topogram_channel_index", -1),
        topogram_gate_init_bias=cfg["model"].get("topogram_gate_init_bias", -2.0),
        topogram_conditioned=cfg["model"].get("topogram_conditioned"),
        mri_channel_indices=cfg["model"].get("mri_channel_indices"),
        mri_base_channels=cfg["model"].get("mri_base_channels"),
        mri_gate_init_bias=cfg["model"].get("mri_gate_init_bias", -2.0),
        global_context=cfg["model"].get("global_context"),
    ).to(device)
    print(f"Model: {cfg['model'].get('name', 'mdixon_resunet3d')}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
    )
    criterion = CombinedPseudoCTLoss(cfg).to(device)
    active_objectives = [
        name for name, enabled in criterion.active_terms.items() if enabled
    ]
    print("Active loss objectives:", ", ".join(active_objectives))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
        init_scale=float(cfg["training"].get("amp_init_scale", 4096.0)),
    )

    output_dir = Path(cfg["output"]["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    plots_dir = output_dir / "plots"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    log_csv = logs_dir / "train_log.csv"
    full_volume_csv = logs_dir / "full_volume_val_log.csv"
    step_progress_csv = logs_dir / "step_progress.csv"
    checkpoint_index_csv = checkpoints_dir / "checkpoint_index.csv"
    resume_path = resolve_resume_path(args.resume, checkpoints_dir)
    is_resuming = resume_path is not None
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Log CSV: {log_csv.resolve()}")
    print(f"Checkpoint dir: {checkpoints_dir.resolve()}")

    best_scores = {
        "mu_MAE_official_raw": float("inf"),
        "mu_MAE_smoothed": float("inf"),
        "projection_mu_error": float("inf"),
        "organ_bias_mean": float("inf"),
        "loss_total": float("inf"),
        "ct_mu_map_mae_official_like": float("inf"),
        "robust_full_volume_pet_proxy_whole_body": float("inf"),
        "robust_full_volume_pet_proxy_organ_bias": float("inf"),
        "robust_full_volume_pet_proxy_brain_outlier": float("inf"),
        "full_volume_relacf_error": float("inf"),
        "robust_full_volume_head_relacf_tail_cvar": float("inf"),
        "full_volume_logacf_error": float("inf"),
        "full_volume_logacf_signed_bias": float("inf"),
        "full_volume_head_logacf_error": float("inf"),
        "full_volume_head_logacf_tail_cvar": float("inf"),
        "robust_full_volume_head_logacf_tail_cvar": float("inf"),
    }
    history = {
        "val_total": [],
        "val_mu_raw": [],
        "val_mu": [],
        "val_global_acf": [],
        "val_global_signed": [],
        "val_pet_whole_body": [],
        "val_pet_organ_bias": [],
        "val_pet_brain_outlier": [],
        "val_organ": [],
    }
    if is_resuming:
        resumed_epoch = load_resume_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            criterion,
            scaler,
            device,
            cfg,
        )
        best_scores = restore_best_scores_from_index(checkpoint_index_csv, best_scores)
    else:
        resumed_epoch = 0

    # Do not overwrite an existing run's provenance until a requested resume
    # checkpoint has passed the recipe-revision guard.
    maybe_write_effective_config(cfg, output_dir)
    init_csv(log_csv, overwrite=not is_resuming)
    init_full_volume_csv(full_volume_csv, overwrite=not is_resuming)
    init_step_progress_csv(step_progress_csv, overwrite=not is_resuming)
    init_checkpoint_index(checkpoint_index_csv, overwrite=not is_resuming)

    start_epoch = resumed_epoch + 1
    max_epoch = int(cfg["training"]["epochs"])
    if start_epoch > max_epoch:
        print(f"Checkpoint already reached epoch {resumed_epoch}; target epochs={max_epoch}. Nothing to do.")
        return

    for epoch in range(start_epoch, max_epoch + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            use_amp,
            epoch,
            cfg,
            checkpoints_dir=checkpoints_dir,
            step_progress_csv=step_progress_csv,
        )
        val_metrics = run_validation(model, val_loader, criterion, device, use_amp, epoch, cfg)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        append_csv(log_csv, epoch, "train", lr, train_metrics)
        append_csv(
            log_csv,
            epoch,
            "train_monitor" if monitor_is_leaked else "val",
            lr,
            val_metrics,
        )

        history["val_total"].append(val_metrics["loss_total"])
        history["val_mu_raw"].append(val_metrics["mu_MAE_official_raw"])
        history["val_mu"].append(val_metrics["mu_MAE_smoothed"])
        history["val_global_acf"].append(val_metrics["global_logacf_error"])
        history["val_global_signed"].append(val_metrics["global_logacf_signed_bias"])
        history["val_pet_whole_body"].append(
            val_metrics.get("pet_proxy_whole_body_suv_mae", float("nan"))
        )
        history["val_pet_organ_bias"].append(
            val_metrics.get("pet_proxy_organ_bias", float("nan"))
        )
        history["val_pet_brain_outlier"].append(
            val_metrics.get("pet_proxy_brain_outlier", float("nan"))
        )
        history["val_organ"].append(val_metrics["organ_bias_mean"])

        print(
            f"Epoch {epoch:03d} "
            f"train_total={train_metrics['loss_total']:.4f} "
            f"monitor_total={val_metrics['loss_total']:.4f} "
            f"monitor_mu_raw={val_metrics['mu_MAE_official_raw']:.6f} "
            f"monitor_mu={val_metrics['mu_MAE_smoothed']:.5f} "
            f"monitor_global_acf={val_metrics['global_logacf_error']:.5f} "
            f"monitor_global_signed={val_metrics['global_logacf_signed_bias']:.5f} "
            f"monitor_pet_wb={val_metrics.get('pet_proxy_whole_body_suv_mae', float('nan')):.5f} "
            f"monitor_pet_organ={val_metrics.get('pet_proxy_organ_bias', float('nan')):.5f} "
            f"monitor_pet_brain={val_metrics.get('pet_proxy_brain_outlier', float('nan')):.5f} "
            f"monitor_organ={val_metrics['organ_bias_mean']:.5f} "
            f"train_peak_reserved={train_metrics['gpu_mem_peak_reserved_mb'] / 1024.0:.2f}GiB"
        )

        checkpoint_specs = [
            ("mu_MAE_official_raw", "best_by_official_mu_mae.pth"),
            ("mu_MAE_smoothed", "best_by_mu_mae.pth"),
            ("projection_mu_error", "best_by_projection_mu.pth"),
            ("organ_bias_mean", "best_by_organ_bias.pth"),
            ("loss_total", "best_by_total_val_loss.pth"),
        ]
        for score_name, filename in checkpoint_specs:
            score_value = val_metrics[score_name]
            if not is_finite_metric(score_value):
                continue
            if score_value < best_scores[score_name]:
                best_scores[score_name] = score_value
                checkpoint_path = checkpoints_dir / filename
                size = save_checkpoint(
                    checkpoint_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    cfg,
                    score_name,
                    score_value,
                    criterion,
                    scaler,
                )
                append_checkpoint_index(
                    checkpoint_index_csv,
                    epoch,
                    filename,
                    score_name,
                    score_value,
                    size,
                    checkpoint_path,
                )
                print(f"Saved {filename}: epoch={epoch} {score_name}={score_value:.6g}")
                if score_name == "mu_MAE_official_raw":
                    alias_path = checkpoints_dir / "best_model.pth"
                    alias_size = save_checkpoint(
                        alias_path,
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        cfg,
                        score_name,
                        score_value,
                        criterion,
                        scaler,
                    )
                    append_checkpoint_index(
                        checkpoint_index_csv,
                        epoch,
                        "best_model.pth",
                        score_name,
                        score_value,
                        alias_size,
                        alias_path,
                    )

        last_path = checkpoints_dir / "last_model.pth"
        last_size = save_checkpoint(
            last_path,
            epoch,
            model,
            optimizer,
            scheduler,
            cfg,
            "loss_total",
            val_metrics["loss_total"],
            criterion,
            scaler,
        )
        append_checkpoint_index(
            checkpoint_index_csv,
            epoch,
            "last_model.pth",
            "loss_total",
            val_metrics["loss_total"],
            last_size,
            last_path,
        )

        save_every = int(cfg["training"].get("save_every", 0) or 0)
        if save_every > 0 and epoch % save_every == 0:
            periodic_name = f"epoch_{epoch:04d}.pth"
            periodic_path = checkpoints_dir / periodic_name
            periodic_size = save_checkpoint(
                periodic_path,
                epoch,
                model,
                optimizer,
                scheduler,
                cfg,
                "loss_total",
                val_metrics["loss_total"],
                criterion,
                scaler,
            )
            append_checkpoint_index(
                checkpoint_index_csv,
                epoch,
                periodic_name,
                "loss_total",
                val_metrics["loss_total"],
                periodic_size,
                periodic_path,
            )
            print(f"Saved {periodic_name}: epoch={epoch}")

        fv_cfg = cfg.get("full_volume_validation", {})
        fv_enabled = bool(fv_cfg.get("enabled", False))
        fv_every = int(fv_cfg.get("every", 0) or 0)
        fv_start_epoch = int(fv_cfg.get("start_epoch", fv_every if fv_every > 0 else 1))
        if fv_enabled and fv_every > 0 and epoch >= fv_start_epoch and epoch % fv_every == 0:
            fv_output_dir = output_dir / fv_cfg.get("output_dir", "full_volume_val")
            full_rows, full_metrics = run_full_volume_validation(model, val_data, cfg, device, epoch, fv_output_dir)
            append_full_volume_csv(full_volume_csv, epoch, full_rows, full_metrics)

            full_checkpoint_specs = [
                ("ct_mu_map_mae_official_like", "best_by_full_volume_ct_mu.pth"),
                (
                    "robust_full_volume_pet_proxy_whole_body",
                    "best_by_full_volume_pet_whole_body.pth",
                ),
                (
                    "robust_full_volume_pet_proxy_organ_bias",
                    "best_by_full_volume_pet_organ_bias.pth",
                ),
                (
                    "robust_full_volume_pet_proxy_brain_outlier",
                    "best_by_full_volume_pet_brain_outlier.pth",
                ),
            ]
            for full_score_name, full_best_name in full_checkpoint_specs:
                full_score_value = full_metrics.get(full_score_name, float("nan"))
                if not is_finite_metric(full_score_value):
                    continue
                if full_score_value >= best_scores[full_score_name]:
                    continue

                best_scores[full_score_name] = full_score_value
                full_best_path = checkpoints_dir / full_best_name
                full_best_size = save_checkpoint(
                    full_best_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    cfg,
                    full_score_name,
                    full_score_value,
                    criterion,
                    scaler,
                )
                append_checkpoint_index(
                    checkpoint_index_csv,
                    epoch,
                    full_best_name,
                    full_score_name,
                    full_score_value,
                    full_best_size,
                    full_best_path,
                )
                print(f"Saved {full_best_name}: epoch={epoch} {full_score_name}={full_score_value:.6g}")

        plot_losses(history, plots_dir / "validation_metrics.png")


if __name__ == "__main__":
    main()
