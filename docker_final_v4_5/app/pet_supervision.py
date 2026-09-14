from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.processing import resample_from_to


PET_REGION_NAMES = (
    "brain",
    "liver",
    "spleen",
    "heart",
    "pancreas",
    "muscle",
    "adipose",
    "extremities",
)
PET_SUPERVISION_CACHE_VERSION = 2


def _compact_affine(
    source_affine: np.ndarray,
    source_shape: Sequence[int],
    target_shape: Sequence[int],
) -> np.ndarray:
    """Affine matching PyTorch interpolate(..., align_corners=False) centers."""
    source_affine = np.asarray(source_affine, dtype=np.float64)
    source_shape = np.asarray(source_shape, dtype=np.float64)
    target_shape = np.asarray(target_shape, dtype=np.float64)
    scale = source_shape / target_shape

    target_affine = source_affine.copy()
    target_affine[:3, :3] = source_affine[:3, :3] @ np.diag(scale)
    target_affine[:3, 3] = (
        source_affine[:3, :3] @ ((scale - 1.0) / 2.0)
        + source_affine[:3, 3]
    )
    return target_affine


def _resample_array(
    values: np.ndarray,
    affine: np.ndarray,
    target_shape: Sequence[int],
    target_affine: np.ndarray,
    order: int,
) -> np.ndarray:
    image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), affine)
    result = resample_from_to(
        image,
        (tuple(int(value) for value in target_shape), target_affine),
        order=int(order),
        mode="constant",
        cval=0.0,
    )
    return np.asarray(result.dataobj, dtype=np.float32)


def _soft_resample_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    source_ct_shape: Sequence[int],
    source_ct_affine: np.ndarray,
    compact_shape: Sequence[int],
    supersample_factor: int,
) -> np.ndarray:
    factor = max(1, int(supersample_factor))
    high_shape = tuple(int(value) * factor for value in compact_shape)
    high_affine = _compact_affine(source_ct_affine, source_ct_shape, high_shape)
    high = _resample_array(
        np.asarray(mask, dtype=np.float32),
        affine,
        high_shape,
        high_affine,
        order=1,
    )
    high_tensor = torch.from_numpy(high).view(1, 1, *high_shape)
    if factor > 1:
        compact = F.avg_pool3d(
            high_tensor,
            kernel_size=factor,
            stride=factor,
        )
    else:
        compact = high_tensor
    return compact[0, 0].clamp_(0.0, 1.0).numpy().astype(np.float32)


def _suv_activity(pet_img: nib.spatialimages.SpatialImage, body_img) -> np.ndarray:
    pet = np.asarray(pet_img.dataobj, dtype=np.float32)
    body = np.asarray(body_img.dataobj) > 0
    if pet.shape != body.shape:
        raise ValueError(
            f"GT PET/body segmentation shape mismatch: pet={pet.shape}, body={body.shape}"
        )
    pet_voxel_ml = float(np.prod(pet_img.header.get_zooms()[:3])) / 1000.0
    body_voxel_ml = float(np.prod(body_img.header.get_zooms()[:3])) / 1000.0
    total_activity = float(np.sum(pet, dtype=np.float64)) * pet_voxel_ml
    body_weight_g = float(np.sum(body, dtype=np.float64)) * body_voxel_ml
    if not np.isfinite(total_activity) or total_activity <= 0.0:
        raise ValueError("GT PET has non-positive or non-finite integrated activity")
    if not np.isfinite(body_weight_g) or body_weight_g <= 0.0:
        raise ValueError("PET body segmentation has zero volume")
    suv = pet * (body_weight_g / total_activity)
    return np.where(body & np.isfinite(suv), np.maximum(suv, 0.0), 0.0).astype(np.float32)


def _official_pet_masks(pet_paths: Mapping[str, str], exclusion_cm: float):
    pet_img = nib.load(pet_paths["pet_gt"])
    body_img = nib.load(pet_paths["pet_body_seg"])
    organ_img = nib.load(pet_paths["pet_organ_seg"])
    tissue_img = nib.load(pet_paths["pet_tissue_seg"])

    body = np.asarray(body_img.dataobj) > 0
    organ = np.asarray(organ_img.dataobj)
    tissue = np.asarray(tissue_img.dataobj)
    if body.shape != organ.shape or body.shape != tissue.shape:
        raise ValueError(
            "PET evaluator segmentation shape mismatch: "
            f"body={body.shape}, organ={organ.shape}, tissue={tissue.shape}"
        )

    liver = organ == 5
    if not np.any(liver):
        raise ValueError("PET organ segmentation has no liver label 5")
    exclusion_slices = int(
        round((float(exclusion_cm) * 10.0) / float(pet_img.header.get_zooms()[2]))
    )
    superior_slice = int(np.min(np.where(liver)[2]))
    z_min = max(0, superior_slice - exclusion_slices)
    z_max = min(body.shape[2], superior_slice + exclusion_slices)
    exclusion = np.zeros_like(body, dtype=bool)
    exclusion[:, :, z_min:z_max] = True

    regions = {
        "brain": (organ == 90, organ_img.affine),
        "liver": (liver, organ_img.affine),
        "spleen": (organ == 1, organ_img.affine),
        "heart": (organ == 51, organ_img.affine),
        "pancreas": (organ == 7, organ_img.affine),
        "muscle": (tissue == 3, tissue_img.affine),
        "adipose": ((tissue == 1) | (tissue == 2), tissue_img.affine),
        "extremities": (np.asarray(body_img.dataobj) == 2, body_img.affine),
    }
    empty = [name for name, (mask, _) in regions.items() if not np.any(mask)]
    if empty:
        raise ValueError(f"Official PET evaluator masks are empty: {empty}")

    return pet_img, body_img, body & (~exclusion), regions


def _cache_path(
    cache_dir: str | os.PathLike[str],
    subject_id: str,
    compact_shape: Sequence[int],
    supersample_factor: int,
) -> Path:
    shape_tag = "x".join(str(int(value)) for value in compact_shape)
    return Path(cache_dir) / (
        f"{subject_id}_pet-evaluator-v{PET_SUPERVISION_CACHE_VERSION}_"
        f"{shape_tag}_ss{int(supersample_factor)}.npz"
    )


def _read_cache(
    path: Path,
    compact_shape: Sequence[int],
    target_affine: np.ndarray,
    exclusion_cm: float,
    mask_supersample_factor: int,
):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if int(metadata.get("version", -1)) != PET_SUPERVISION_CACHE_VERSION:
            return None
        if tuple(metadata.get("compact_shape", [])) != tuple(int(v) for v in compact_shape):
            return None
        if not np.isclose(float(metadata.get("exclusion_cm", -1.0)), float(exclusion_cm)):
            return None
        if int(metadata.get("mask_supersample_factor", -1)) != int(mask_supersample_factor):
            return None
        if tuple(metadata.get("region_names", [])) != PET_REGION_NAMES:
            return None
        cached_affine = np.asarray(payload["target_affine"], dtype=np.float64)
        if not np.allclose(cached_affine, target_affine, atol=1e-4, rtol=1e-5):
            return None
        return {
            "activity_suv": np.asarray(payload["activity_suv"], dtype=np.float32),
            "eval_mask": np.asarray(payload["eval_mask"], dtype=np.float32),
            "body_mask": np.asarray(payload["body_mask"], dtype=np.float32),
            "region_masks": np.asarray(payload["region_masks"], dtype=np.float32),
            "cache_hit": True,
        }


def _write_cache(path: Path, arrays: Mapping[str, np.ndarray], metadata: dict, target_affine):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".npz",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            np.savez_compressed(
                handle,
                activity_suv=np.asarray(arrays["activity_suv"], dtype=np.float32),
                eval_mask=np.asarray(arrays["eval_mask"], dtype=np.float16),
                body_mask=np.asarray(arrays["body_mask"], dtype=np.float16),
                region_masks=np.asarray(arrays["region_masks"], dtype=np.float16),
                target_affine=np.asarray(target_affine, dtype=np.float64),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def load_or_create_exact_pet_supervision(
    *,
    subject_id: str,
    pet_paths: Mapping[str, str],
    source_ct_shape: Sequence[int],
    source_ct_affine: np.ndarray,
    compact_shape: Sequence[int],
    cache_dir: str | os.PathLike[str] | None,
    exclusion_cm: float = 4.0,
    mask_supersample_factor: int = 2,
) -> dict[str, torch.Tensor | bool]:
    """Create GT-PET activity and official masks on the compact CT grid."""
    compact_shape = tuple(int(value) for value in compact_shape)
    source_ct_shape = tuple(int(value) for value in source_ct_shape)
    target_affine = _compact_affine(source_ct_affine, source_ct_shape, compact_shape)
    path = (
        _cache_path(cache_dir, subject_id, compact_shape, mask_supersample_factor)
        if cache_dir
        else None
    )
    cached = (
        _read_cache(
            path,
            compact_shape,
            target_affine,
            exclusion_cm=exclusion_cm,
            mask_supersample_factor=mask_supersample_factor,
        )
        if path
        else None
    )
    if cached is None:
        pet_img, body_img, eval_mask, regions = _official_pet_masks(
            pet_paths,
            exclusion_cm=exclusion_cm,
        )
        activity_native = _suv_activity(pet_img, body_img)
        activity = _resample_array(
            activity_native,
            pet_img.affine,
            compact_shape,
            target_affine,
            order=1,
        )
        body_native = np.asarray(body_img.dataobj) > 0
        body_mask = _soft_resample_mask(
            body_native,
            body_img.affine,
            source_ct_shape,
            source_ct_affine,
            compact_shape,
            mask_supersample_factor,
        )
        activity = np.where(body_mask > 0.0, np.maximum(activity, 0.0), 0.0).astype(np.float32)
        eval_compact = _soft_resample_mask(
            eval_mask,
            body_img.affine,
            source_ct_shape,
            source_ct_affine,
            compact_shape,
            mask_supersample_factor,
        )
        compact_regions = []
        for name in PET_REGION_NAMES:
            region_mask, region_affine = regions[name]
            compact_region = _soft_resample_mask(
                region_mask,
                region_affine,
                source_ct_shape,
                source_ct_affine,
                compact_shape,
                mask_supersample_factor,
            )
            if float(compact_region.sum()) <= 0.0:
                raise ValueError(
                    f"PET region '{name}' vanished after physical resampling for {subject_id}"
                )
            compact_regions.append(compact_region)
        cached = {
            "activity_suv": activity,
            "eval_mask": eval_compact,
            "body_mask": body_mask,
            "region_masks": np.stack(compact_regions, axis=0),
            "cache_hit": False,
        }
        if path:
            _write_cache(
                path,
                cached,
                metadata={
                    "version": PET_SUPERVISION_CACHE_VERSION,
                    "subject_id": subject_id,
                    "compact_shape": list(compact_shape),
                    "region_names": list(PET_REGION_NAMES),
                    "exclusion_cm": float(exclusion_cm),
                    "mask_supersample_factor": int(mask_supersample_factor),
                },
                target_affine=target_affine,
            )

    return {
        "activity_suv": torch.from_numpy(cached["activity_suv"]).unsqueeze(0).float(),
        "eval_mask": torch.from_numpy(cached["eval_mask"]).unsqueeze(0).float(),
        "body_mask": torch.from_numpy(cached["body_mask"]).unsqueeze(0).float(),
        "region_masks": torch.from_numpy(cached["region_masks"]).float(),
        "cache_hit": bool(cached["cache_hit"]),
    }
