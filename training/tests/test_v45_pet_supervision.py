from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from evaluator_surrogate import (
    evaluator_aligned_pet_proxy_losses,
    relative_acf_response_map,
)
from losses import CombinedPseudoCTLoss
from pet_supervision import PET_REGION_NAMES, load_or_create_exact_pet_supervision


def test_uniform_ray_mode_is_activity_invariant():
    shape = (1, 1, 8, 8, 4)
    target = torch.zeros(shape)
    pred = torch.zeros(shape)
    pred[:, :, 2:6, 2:6] = 0.01
    support = torch.ones(shape)
    activity_a = torch.ones(shape)
    activity_b = torch.linspace(0.1, 5.0, pred.numel()).reshape(shape)

    response_a, _, _ = relative_acf_response_map(
        pred,
        target,
        support,
        spacing_mm=(2.0, 2.0, 2.0),
        angles_degrees=(0.0, 45.0, 90.0),
        downsample_factors=(1, 1, 1),
        ray_activity=activity_a,
        ray_activity_alpha=0.0,
    )
    response_b, _, _ = relative_acf_response_map(
        pred,
        target,
        support,
        spacing_mm=(2.0, 2.0, 2.0),
        angles_degrees=(0.0, 45.0, 90.0),
        downsample_factors=(1, 1, 1),
        ray_activity=activity_b,
        ray_activity_alpha=0.0,
    )
    torch.testing.assert_close(response_a, response_b, atol=0.0, rtol=0.0)


def test_controlled_gt_ray_weighting_is_tightly_bounded():
    shape = (1, 1, 8, 8, 4)
    target = torch.zeros(shape)
    pred = torch.zeros(shape)
    pred[:, :, 2:6, 2:6] = 0.01
    support = torch.ones(shape)
    activity = torch.linspace(0.0, 20.0, pred.numel()).reshape(shape)

    _, _, diagnostics = relative_acf_response_map(
        pred,
        target,
        support,
        spacing_mm=(2.0, 2.0, 2.0),
        angles_degrees=(0.0, 45.0, 90.0),
        downsample_factors=(1, 1, 1),
        ray_activity=activity,
        ray_activity_alpha=0.15,
        ray_activity_weight_clip=(0.8, 1.2),
    )
    assert diagnostics["ray_activity_alpha"].item() == pytest.approx(0.15)
    assert 0.8 <= diagnostics["ray_activity_weight_mean"].item() <= 1.2


def test_exact_soft_masks_keep_all_evaluator_regions():
    shape = (1, 1, 8, 8, 4)
    target = torch.zeros(shape)
    pred = torch.zeros(shape)
    pred[:, :, 2:6, 2:6] = 0.01
    support = torch.ones(shape)
    activity = torch.ones(shape)
    exact_masks = torch.zeros((1, len(PET_REGION_NAMES), *shape[2:]))
    for index in range(len(PET_REGION_NAMES)):
        x = index % shape[2]
        y = (index * 3) % shape[3]
        exact_masks[:, index, x, y, 1] = 0.25

    losses, diagnostics = evaluator_aligned_pet_proxy_losses(
        pred_mu=pred,
        target_mu=target,
        official_eval_mask=support,
        prediction_mask=support,
        organ_seg=None,
        body_region_seg=None,
        target_hu=torch.zeros(shape),
        activity_proxy=activity,
        spacing_mm=(2.0, 2.0, 2.0),
        config={
            "angles_deg": [0.0, 90.0],
            "downsample_factors": [1, 1, 1],
            "brain_thresholds": [0.05, 0.10, 0.15],
            "brain_temperature": 0.01,
            "ray_activity_weighting": {"enabled": False},
        },
        exact_eval_mask=support,
        exact_region_masks=exact_masks,
        activity_is_gt_pet=True,
    )
    assert torch.isfinite(losses["pet_organ_bias"])
    for name in PET_REGION_NAMES:
        assert diagnostics[f"pet_proxy_region_present_{name}"].item() == 1.0


def _save(path: Path, values: np.ndarray, affine: np.ndarray):
    nib.save(nib.Nifti1Image(values, affine), path)


def test_pet_supervision_resamples_flipped_pet_grid_and_caches(tmp_path):
    pet_shape = (12, 12, 12)
    ct_shape = (12, 12, 12)
    compact_shape = (6, 6, 6)
    ct_affine = np.eye(4, dtype=np.float64)
    pet_affine = np.eye(4, dtype=np.float64)
    pet_affine[2, 2] = -1.0
    pet_affine[2, 3] = pet_shape[2] - 1

    pet = np.ones(pet_shape, dtype=np.float32)
    body = np.ones(pet_shape, dtype=np.int16)
    body[:2] = 2
    organ = np.zeros(pet_shape, dtype=np.int16)
    tissue = np.zeros(pet_shape, dtype=np.int16)
    organ[2:4, 2:4, 1:3] = 90
    organ[4:8, 4:8, 4:7] = 5
    organ[2:4, 7:9, 5:7] = 1
    organ[7:9, 2:4, 5:7] = 51
    organ[5:7, 8:10, 7:9] = 7
    tissue[1:6, 1:6, 8:11] = 3
    tissue[6:11, 6:11, 8:11] = 1

    paths = {}
    for key, values in {
        "pet_gt": pet,
        "pet_body_seg": body,
        "pet_organ_seg": organ,
        "pet_tissue_seg": tissue,
    }.items():
        path = tmp_path / f"{key}.nii.gz"
        _save(path, values, pet_affine)
        paths[key] = str(path)

    kwargs = dict(
        subject_id="sub-test",
        pet_paths=paths,
        source_ct_shape=ct_shape,
        source_ct_affine=ct_affine,
        compact_shape=compact_shape,
        cache_dir=tmp_path / "cache",
        exclusion_cm=0.2,
        mask_supersample_factor=2,
    )
    first = load_or_create_exact_pet_supervision(**kwargs)
    second = load_or_create_exact_pet_supervision(**kwargs)

    assert first["activity_suv"].shape == (1, *compact_shape)
    assert first["region_masks"].shape == (len(PET_REGION_NAMES), *compact_shape)
    assert torch.all(first["region_masks"].sum(dim=(1, 2, 3)) > 0)
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    torch.testing.assert_close(first["region_masks"], second["region_masks"])


def test_combined_loss_uses_exact_full_fov_pet_targets_and_backpropagates():
    patch_shape = (4, 4, 4)
    global_shape = (8, 8, 8)
    pred = torch.full((1, 1, *patch_shape), 50.0, requires_grad=True)
    ones_patch = torch.ones_like(pred)
    ones_global = torch.ones((1, 1, *global_shape))
    exact_regions = torch.ones((1, len(PET_REGION_NAMES), *global_shape))
    config = {
        "loss": {
            "spacing_mm": [2.0, 2.0, 2.0],
            "mu_fwhm_mm": 0.5,
            "normalize_by_initial_value": False,
            "pet_evaluator_proxy": {
                "enabled": True,
                "training_mode": "full_fov_patch_injected",
                "full_fov_shape": list(global_shape),
                "full_fov_angles_deg": [0.0, 90.0],
                "full_fov_downsample_factors": [1, 1, 1],
                "delta_tau_clip": 0.5,
                "brain_thresholds": [0.05, 0.10, 0.15],
                "brain_temperature": 0.01,
                "ray_activity_weighting": {
                    "enabled": True,
                    "gt_pet_alpha": 0.15,
                    "nacpet_alpha": 0.0,
                    "weight_clip": [0.8, 1.2],
                },
            },
            "phases": [
                {
                    "start_epoch": 1,
                    "lambda_hu": 0.0,
                    "lambda_grad": 0.0,
                    "lambda_mu_raw": 0.0,
                    "lambda_mu": 0.0,
                    "lambda_proj": 0.0,
                    "lambda_organ": 0.0,
                    "lambda_global_mu_raw": 0.0,
                    "lambda_global_mu": 0.0,
                    "lambda_global_acf": 0.0,
                    "lambda_global_acf_signed": 0.0,
                    "lambda_regional_relacf": 0.0,
                    "lambda_organ_ray_signed": 0.0,
                    "lambda_local_head_acf": 0.0,
                    "lambda_local_head_acf_signed": 0.0,
                    "lambda_local_head_acf_tail": 0.0,
                    "lambda_pet_whole_body": 1.0,
                    "lambda_pet_organ_bias": 1.0,
                    "lambda_pet_brain_outlier": 1.0,
                }
            ],
        }
    }
    batch = {
        "ct": torch.zeros_like(pred),
        "prediction_mask": ones_patch,
        "body_seg": ones_patch,
        "official_eval_mask": ones_patch,
        "train_crop_start": torch.tensor([[2, 2, 2]]),
        "crop_original_shape": torch.tensor([list(global_shape)]),
        "global_official_eval_mask": ones_global,
        "global_prediction_mask": ones_global,
        "global_target_hu": torch.zeros_like(ones_global),
        "global_activity_proxy": ones_global,
        "global_spacing_mm": torch.tensor([[2.0, 2.0, 2.0]]),
        "global_pet_eval_mask": ones_global,
        "global_pet_region_masks": exact_regions,
        "global_activity_is_gt_pet": torch.tensor([[1.0]]),
    }

    criterion = CombinedPseudoCTLoss(config)
    total, values = criterion(pred, batch, epoch=1)
    assert torch.isfinite(total)
    assert values["pet_proxy_uses_gt_pet"].item() == 1.0
    assert values["pet_proxy_uses_exact_masks"].item() == 1.0
    assert values["ray_activity_alpha"].item() == pytest.approx(0.15)
    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0
