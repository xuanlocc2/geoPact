from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    MapTransform,
)

from pet_physics import load_or_compute_global_mu_targets
from pet_supervision import load_or_create_exact_pet_supervision


BASE_IMAGE_KEYS = ["nacpet", "mri_in", "mri_out"]
OPTIONAL_IMAGE_KEYS = ["topogram"]
LABEL_KEYS = ["ct", "prediction_mask", "body_seg", "organ_seg"]
PET_SUPERVISION_PATH_KEYS = [
    "pet_gt",
    "pet_body_seg",
    "pet_organ_seg",
    "pet_tissue_seg",
]
LOAD_KEYS = BASE_IMAGE_KEYS + OPTIONAL_IMAGE_KEYS + LABEL_KEYS
DEFAULT_INPUT_CHANNELS = ["nacpet", "mri_in", "mri_out"]


def _as_4d(x) -> torch.Tensor:
    arr = torch.as_tensor(x, dtype=torch.float32)
    if arr.ndim == 3:
        arr = arr.unsqueeze(0)
    if arr.ndim != 4:
        raise RuntimeError(f"Expected channel-first 3D tensor, got shape {tuple(arr.shape)}")
    return arr


def _spacing_from_image(x, fallback: Sequence[float]) -> tuple[float, float, float]:
    """Read physical x/y/z spacing from a MONAI MetaTensor when available."""
    affine = getattr(x, "affine", None)
    if affine is not None:
        affine_tensor = torch.as_tensor(affine, dtype=torch.float64)
        if affine_tensor.shape[-2:] == (4, 4):
            spacing = torch.linalg.vector_norm(affine_tensor[:3, :3], dim=0)
            if bool(torch.all(torch.isfinite(spacing))) and bool(torch.all(spacing > 0)):
                return tuple(float(value) for value in spacing.tolist())
    return tuple(float(value) for value in fallback)


def _match_spatial(x: torch.Tensor, ref_shape: Sequence[int]) -> torch.Tensor:
    if tuple(x.shape[1:]) == tuple(ref_shape):
        return x

    # Broadcast singleton dimensions such as topogram shape [1, 512, 1, 531].
    if all(src == dst or src == 1 for src, dst in zip(x.shape[1:], ref_shape)):
        return x.expand((x.shape[0], *ref_shape)).clone()

    x5 = x.unsqueeze(0)
    y = F.interpolate(x5, size=tuple(ref_shape), mode="trilinear", align_corners=False)
    return y.squeeze(0)


def _binarize_mask(x):
    if torch.is_tensor(x):
        return (x > 0).to(dtype=torch.float32)
    return (x > 0).astype("float32")


def _clamp_start(start: int, dim: int, roi: int) -> int:
    return max(0, min(int(start), max(0, int(dim) - int(roi))))


def _crop_spatial_tensor(x, start: Sequence[int], roi: Sequence[int]):
    if not torch.is_tensor(x) or x.ndim < 3:
        return x
    spatial_shape = tuple(x.shape[-3:])
    if len(spatial_shape) != 3:
        return x
    roi = tuple(min(int(r), int(s)) for r, s in zip(roi, spatial_shape))
    start = tuple(_clamp_start(s, dim, r) for s, dim, r in zip(start, spatial_shape, roi))
    slices = tuple(slice(s, s + r) for s, r in zip(start, roi))
    return x[(..., *slices)]


def _spatial_mask_fraction(mask, start: Sequence[int], roi: Sequence[int]) -> float:
    if mask is None or not torch.is_tensor(mask):
        return float("nan")
    mask_3d = mask.detach()
    while mask_3d.ndim > 3:
        mask_3d = mask_3d[0]
    if mask_3d.ndim != 3:
        return float("nan")
    slices = tuple(slice(int(s), int(s) + int(r)) for s, r in zip(start, roi))
    patch = mask_3d[slices]
    if patch.numel() == 0:
        return float("nan")
    return float((patch > 0).float().mean().item())


class DeterministicSpatialCropSamplesD(MapTransform):
    """Create fixed validation crops, centered on the body/prediction mask when available."""

    def __init__(
        self,
        keys,
        roi_size: Sequence[int],
        num_samples: int = 1,
        mask_key: str = "prediction_mask",
        allow_missing_keys: bool = True,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.roi_size = tuple(int(v) for v in roi_size)
        self.num_samples = max(1, int(num_samples))
        self.mask_key = mask_key

    def _spatial_shape(self, d) -> tuple[int, int, int]:
        for key in self.keys:
            if key in d and torch.is_tensor(d[key]) and d[key].ndim >= 3:
                return tuple(int(v) for v in d[key].shape[-3:])
        raise RuntimeError("DeterministicSpatialCropSamplesD could not infer spatial shape")

    def _bbox_from_mask(self, d, spatial_shape):
        mask = d.get(self.mask_key)
        if mask is None and self.mask_key != "body_seg":
            mask = d.get("body_seg")
        if mask is None or not torch.is_tensor(mask):
            return None

        mask_3d = mask.detach()
        while mask_3d.ndim > 3:
            mask_3d = mask_3d[0]
        if tuple(mask_3d.shape[-3:]) != tuple(spatial_shape):
            return None

        coords = torch.nonzero(mask_3d > 0, as_tuple=False)
        if coords.numel() == 0:
            return None
        lo = coords.min(dim=0).values.float()
        hi = coords.max(dim=0).values.float()
        return lo, hi

    def _starts(self, d) -> list[tuple[int, int, int]]:
        spatial_shape = self._spatial_shape(d)
        roi = tuple(min(int(r), int(s)) for r, s in zip(self.roi_size, spatial_shape))
        bbox = self._bbox_from_mask(d, spatial_shape)

        if bbox is None:
            lo = torch.zeros(3, dtype=torch.float32)
            hi = torch.tensor(spatial_shape, dtype=torch.float32) - 1
        else:
            lo, hi = bbox

        extent = hi - lo + 1
        sweep_axis = int(torch.argmax(extent).item())
        center = (lo + hi) / 2.0
        starts = []

        if self.num_samples == 1:
            sweep_centers = [float(center[sweep_axis].item())]
        else:
            min_center = float((lo[sweep_axis] + roi[sweep_axis] / 2.0).item())
            max_center = float((hi[sweep_axis] - roi[sweep_axis] / 2.0).item())
            if max_center < min_center:
                min_center = max_center = float(center[sweep_axis].item())
            sweep_centers = torch.linspace(min_center, max_center, self.num_samples).tolist()

        for sweep_center in sweep_centers:
            crop_center = center.clone()
            crop_center[sweep_axis] = sweep_center
            start = [round(float(crop_center[i].item()) - roi[i] / 2.0) for i in range(3)]
            starts.append(tuple(_clamp_start(start[i], spatial_shape[i], roi[i]) for i in range(3)))

        # Keep order deterministic and avoid repeated identical crops when the mask is short.
        unique_starts = []
        seen = set()
        for start in starts:
            if start not in seen:
                unique_starts.append(start)
                seen.add(start)
        while len(unique_starts) < self.num_samples:
            unique_starts.append(unique_starts[-1])
        return unique_starts

    def __call__(self, data):
        d = dict(data)
        samples = []
        for start in self._starts(d):
            item = dict(d)
            for key in self.key_iterator(item):
                item[key] = _crop_spatial_tensor(item[key], start, self.roi_size)
            item["crop_original_shape"] = torch.tensor(self._spatial_shape(d), dtype=torch.int64)
            item["val_crop_start"] = torch.tensor(start, dtype=torch.int64)
            item["val_crop_mask_fraction"] = torch.tensor(
                _spatial_mask_fraction(d.get(self.mask_key), start, self.roi_size), dtype=torch.float32
            )
            samples.append(item)
        return samples


class RandomMaskCropSamplesD(MapTransform):
    """Create foreground crops, optionally stratified along the body z extent.

    Region-balanced sampling is label-free beyond the foreground mask. The
    superior-most fraction is treated as head/neck, so the sampler does not
    depend on a challenge-specific organ label map.
    """

    def __init__(
        self,
        keys,
        roi_size: Sequence[int],
        num_samples: int = 1,
        mask_key: str = "prediction_mask",
        min_mask_fraction: float = 0.35,
        max_crop_attempts: int = 16,
        region_sampling: dict | None = None,
        allow_missing_keys: bool = True,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.roi_size = tuple(int(v) for v in roi_size)
        self.num_samples = max(1, int(num_samples))
        self.mask_key = mask_key
        self.min_mask_fraction = float(min_mask_fraction)
        self.max_crop_attempts = max(1, int(max_crop_attempts))
        self.region_sampling = dict(region_sampling or {})
        self.region_sampling_enabled = bool(self.region_sampling.get("enabled", False))
        self.region_axis = int(self.region_sampling.get("axis", 2))
        if self.region_axis not in (0, 1, 2):
            raise ValueError(f"region_sampling.axis must be 0, 1 or 2, got {self.region_axis}")

        raw_regions = self.region_sampling.get("regions", [])
        self.regions = []
        for region_id, region in enumerate(raw_regions):
            name = str(region.get("name", f"region_{region_id}"))
            lower = float(region.get("min", 0.0))
            upper = float(region.get("max", 1.0))
            weight = float(region.get("weight", 1.0))
            if not 0.0 <= lower < upper <= 1.0:
                raise ValueError(f"Invalid normalized bounds for region '{name}': [{lower}, {upper}]")
            if weight < 0.0:
                raise ValueError(f"Region '{name}' has a negative sampling weight")
            self.regions.append(
                {"id": region_id, "name": name, "min": lower, "max": upper, "weight": weight}
            )
        if self.region_sampling_enabled and not self.regions:
            raise ValueError("region_sampling.enabled=true requires at least one configured region")
        self.mandatory_region_names = [
            str(name) for name in self.region_sampling.get("mandatory_regions", [])
        ]

    def _spatial_shape(self, d) -> tuple[int, int, int]:
        for key in self.keys:
            if key in d and torch.is_tensor(d[key]) and d[key].ndim >= 3:
                return tuple(int(v) for v in d[key].shape[-3:])
        raise RuntimeError("RandomMaskCropSamplesD could not infer spatial shape")

    def _mask_3d(self, d, spatial_shape):
        mask = d.get(self.mask_key)
        if mask is None and self.mask_key != "body_seg":
            mask = d.get("body_seg")
        if mask is None or not torch.is_tensor(mask):
            return None

        mask_3d = mask.detach()
        while mask_3d.ndim > 3:
            mask_3d = mask_3d[0]
        if tuple(mask_3d.shape[-3:]) != tuple(spatial_shape):
            return None
        return mask_3d > 0

    def _mask_coords(self, mask_3d):
        if mask_3d is None:
            return None
        coords = torch.nonzero(mask_3d > 0, as_tuple=False)
        if coords.numel() == 0:
            return None
        return coords

    def _random_start(self, spatial_shape, roi):
        return tuple(random.randint(0, max(0, dim - size)) for dim, size in zip(spatial_shape, roi))

    def _foreground_start(self, coord, spatial_shape, roi):
        start = []
        for axis in range(3):
            jitter = random.randint(-roi[axis] // 4, roi[axis] // 4)
            axis_start = int(coord[axis].item()) - roi[axis] // 2 + jitter
            start.append(_clamp_start(axis_start, spatial_shape[axis], roi[axis]))
        return tuple(start)

    def _mask_fraction(self, mask_3d, start, roi) -> float:
        fraction = _spatial_mask_fraction(mask_3d, start, roi)
        return 1.0 if math.isnan(fraction) else fraction

    def _candidate_start(self, coords, spatial_shape, roi):
        if coords is None:
            return self._random_start(spatial_shape, roi)
        coord = coords[random.randrange(int(coords.shape[0]))]
        return self._foreground_start(coord, spatial_shape, roi)

    def _body_region_coords(self, coords, region):
        if coords is None or region is None:
            return coords
        axis_values = coords[:, self.region_axis].float()
        body_min = axis_values.min()
        body_max = axis_values.max()
        denominator = torch.clamp(body_max - body_min, min=1.0)
        normalized = (axis_values - body_min) / denominator
        lower = float(region["min"])
        upper = float(region["max"])
        if upper >= 1.0:
            selected = (normalized >= lower) & (normalized <= upper)
        else:
            selected = (normalized >= lower) & (normalized < upper)
        region_coords = coords[selected]
        return region_coords if region_coords.numel() > 0 else coords

    def _region_schedule(self):
        if not self.region_sampling_enabled:
            return [None] * self.num_samples

        by_name = {region["name"]: region for region in self.regions}
        mandatory = [by_name[name] for name in self.mandatory_region_names if name in by_name]
        random.shuffle(mandatory)
        schedule = mandatory[: self.num_samples]

        mandatory_names = set(self.mandatory_region_names)
        candidates = [region for region in self.regions if region["name"] not in mandatory_names]
        if not candidates:
            candidates = self.regions
        weights = [float(region["weight"]) for region in candidates]
        if sum(weights) <= 0.0:
            weights = [1.0] * len(candidates)
        while len(schedule) < self.num_samples:
            schedule.append(random.choices(candidates, weights=weights, k=1)[0])
        random.shuffle(schedule)
        return schedule

    def _accepted_start(self, mask_3d, coords, spatial_shape, roi):
        best_start = None
        best_fraction = -1.0

        for _ in range(self.max_crop_attempts):
            start = self._candidate_start(coords, spatial_shape, roi)
            fraction = self._mask_fraction(mask_3d, start, roi)
            if fraction > best_fraction:
                best_start = start
                best_fraction = fraction
            if fraction >= self.min_mask_fraction:
                return start, fraction

        return best_start, best_fraction

    def __call__(self, data):
        d = dict(data)
        spatial_shape = self._spatial_shape(d)
        roi = tuple(min(int(r), int(s)) for r, s in zip(self.roi_size, spatial_shape))
        mask_3d = self._mask_3d(d, spatial_shape)
        coords = self._mask_coords(mask_3d)
        samples = []

        for region in self._region_schedule():
            candidate_coords = self._body_region_coords(coords, region)
            start, fraction = self._accepted_start(mask_3d, candidate_coords, spatial_shape, roi)

            region_coverage = float("nan")
            region_transaxial_coverage = float("nan")
            if candidate_coords is not None and candidate_coords.numel() > 0:
                starts = torch.tensor(start, dtype=candidate_coords.dtype, device=candidate_coords.device)
                stops = starts + torch.tensor(roi, dtype=candidate_coords.dtype, device=candidate_coords.device)
                inside = (candidate_coords >= starts) & (candidate_coords < stops)
                region_coverage = float(inside.all(dim=1).float().mean().item())
                transaxial_axes = [axis for axis in range(3) if axis != self.region_axis]
                region_transaxial_coverage = float(
                    inside[:, transaxial_axes].all(dim=1).float().mean().item()
                )

            item = dict(d)
            for key in self.key_iterator(item):
                item[key] = _crop_spatial_tensor(item[key], start, self.roi_size)
            item["crop_original_shape"] = torch.tensor(spatial_shape, dtype=torch.int64)
            item["train_crop_start"] = torch.tensor(start, dtype=torch.int64)
            item["train_crop_mask_fraction"] = torch.tensor(fraction, dtype=torch.float32)
            item["train_crop_region_coverage_fraction"] = torch.tensor(
                region_coverage, dtype=torch.float32
            )
            item["train_crop_region_transaxial_coverage"] = torch.tensor(
                region_transaxial_coverage, dtype=torch.float32
            )
            region_id = -1 if region is None else int(region["id"])
            item["train_crop_region_id"] = torch.tensor(region_id, dtype=torch.int64)
            if coords is not None:
                body_axis_min = float(coords[:, self.region_axis].min().item())
                body_axis_max = float(coords[:, self.region_axis].max().item())
                item["train_body_axis_min"] = torch.tensor(body_axis_min, dtype=torch.float32)
                item["train_body_axis_max"] = torch.tensor(body_axis_max, dtype=torch.float32)
                center = float(start[self.region_axis]) + float(roi[self.region_axis]) / 2.0
                z_fraction = (center - body_axis_min) / max(body_axis_max - body_axis_min, 1.0)
                item["train_crop_body_axis_fraction"] = torch.tensor(
                    min(max(z_fraction, 0.0), 1.0), dtype=torch.float32
                )
            samples.append(item)
        return samples


class CreateOfficialCTEvalMaskD(MapTransform):
    """Create the official CT mu-MAE mask in full-subject CT space.

    The challenge metric uses the body mask and excludes +/- ``exclusion_cm``
    around the superior liver slice. CT arrays are x/y/z and the superior-most
    liver voxel is the largest z index in the official evaluator.
    """

    def __init__(
        self,
        body_key: str = "body_seg",
        organ_key: str = "organ_seg",
        output_key: str = "official_eval_mask",
        liver_label: int = 5,
        spacing_z_mm: float = 2.0,
        exclusion_cm: float = 4.0,
    ):
        super().__init__(keys=[body_key, organ_key], allow_missing_keys=True)
        self.body_key = body_key
        self.organ_key = organ_key
        self.output_key = output_key
        self.liver_label = int(liver_label)
        self.spacing_z_mm = float(spacing_z_mm)
        self.exclusion_cm = float(exclusion_cm)

    def __call__(self, data):
        d = dict(data)
        body_source = d.get(self.body_key, d.get("prediction_mask"))
        if body_source is None:
            return d

        body = _as_4d(body_source) > 0
        eval_mask = body.clone()
        organ_source = d.get(self.organ_key)
        if organ_source is not None:
            organ = _match_spatial(_as_4d(organ_source), body.shape[1:])
            liver = organ[0] == self.liver_label
            coordinates = torch.nonzero(liver, as_tuple=False)
            if coordinates.numel() > 0:
                superior_slice = int(coordinates[:, 2].max().item())
                exclusion_slices = int(round((self.exclusion_cm * 10.0) / self.spacing_z_mm))
                z_min = max(0, superior_slice - exclusion_slices)
                z_max = min(int(body.shape[-1]), superior_slice + exclusion_slices)
                eval_mask[..., z_min:z_max] = False

        d[self.output_key] = eval_mask.to(dtype=torch.float32)
        return d


class ComputeSubjectNormalizationStatsD(MapTransform):
    """Compute full-subject intensity stats using input-only rules before patch crop."""

    def __init__(
        self,
        keys,
        specs: dict[str, dict],
        eps: float = 1e-8,
        max_quantile_samples: int = 1_000_000,
        allow_missing_keys: bool = True,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.specs = specs
        self.eps = float(eps)
        self.max_quantile_samples = int(max_quantile_samples)

    def _select_quantile_values(self, values):
        if values.numel() <= self.max_quantile_samples:
            return values
        step = max(1, values.numel() // self.max_quantile_samples)
        return values[::step][: self.max_quantile_samples]

    def _valid_mask(self, channel: torch.Tensor, spec: dict) -> torch.Tensor:
        valid = torch.isfinite(channel)
        if spec.get("positive", False):
            valid = valid & (channel > 0)
        if spec.get("nonzero", False):
            valid = valid & (channel != 0)
        return valid

    def _stats_for_channel(self, channel: torch.Tensor, spec: dict) -> dict[str, float | bool]:
        valid = self._valid_mask(channel, spec)
        values = channel[valid]
        if values.numel() == 0:
            return {
                "lower": 0.0,
                "upper": 0.0,
                "mean": 0.0,
                "std": 1.0,
                "activity_mean": 1.0,
                "log1p": bool(spec.get("log1p", False)),
            }

        quantile_values = self._select_quantile_values(values)
        lower = torch.quantile(quantile_values, float(spec.get("lower", 0.5)) / 100.0)
        upper = torch.quantile(quantile_values, float(spec.get("upper", 99.5)) / 100.0)

        clipped = torch.clamp(channel, min=lower, max=upper)
        activity_values = clipped[valid].clamp_min(0.0)
        activity_mean = activity_values.mean() if activity_values.numel() else clipped.new_tensor(1.0)
        if spec.get("log1p", False):
            clipped = torch.log1p(torch.clamp(clipped, min=0.0))
        norm_values = clipped[valid]
        mean = norm_values.mean()
        std = norm_values.std(unbiased=False)
        if float(std.item()) <= self.eps:
            std = torch.tensor(1.0, dtype=mean.dtype, device=mean.device)

        return {
            "lower": float(lower.item()),
            "upper": float(upper.item()),
            "mean": float(mean.item()),
            "std": float(std.item()),
            "activity_mean": max(float(activity_mean.item()), self.eps),
            "log1p": bool(spec.get("log1p", False)),
        }

    def __call__(self, data):
        d = dict(data)
        all_stats = dict(d.get("_subject_norm_stats", {}))

        for key in self.key_iterator(d):
            spec = self.specs.get(key, {})
            arr = _as_4d(d[key])
            all_stats[key] = [self._stats_for_channel(arr[channel_idx], spec) for channel_idx in range(arr.shape[0])]

        d["_subject_norm_stats"] = all_stats
        return d


class ApplySubjectNormalizationStatsD(MapTransform):
    """Normalize a cropped patch using precomputed full-subject stats."""

    def __init__(
        self,
        keys,
        stats_key: str = "_subject_norm_stats",
        eps: float = 1e-8,
        delete_stats: bool = True,
        allow_missing_keys: bool = True,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.stats_key = stats_key
        self.eps = float(eps)
        self.delete_stats = bool(delete_stats)

    def _normalize_channel(self, channel: torch.Tensor, stats: dict) -> torch.Tensor:
        finite = torch.isfinite(channel)
        lower = float(stats.get("lower", 0.0))
        upper = float(stats.get("upper", lower))
        mean = float(stats.get("mean", 0.0))
        std = max(float(stats.get("std", 1.0)), self.eps)

        clipped = torch.clamp(channel, min=lower, max=upper)
        if stats.get("log1p", False):
            clipped = torch.log1p(torch.clamp(clipped, min=0.0))
        normalized = (clipped - mean) / std
        return torch.where(finite, normalized, torch.zeros_like(normalized))

    def __call__(self, data):
        d = dict(data)
        all_stats = d.get(self.stats_key)
        if not isinstance(all_stats, dict):
            raise RuntimeError(
                f"ApplySubjectNormalizationStatsD expected '{self.stats_key}' in data. "
                "Run ComputeSubjectNormalizationStatsD before crop/normalization."
            )

        for key in self.key_iterator(d):
            arr = _as_4d(d[key])
            key_stats = all_stats.get(key)
            if key_stats is None:
                raise RuntimeError(f"Missing subject normalization stats for key='{key}'")

            out = torch.zeros_like(arr, dtype=torch.float32)
            for channel_idx in range(arr.shape[0]):
                stats_idx = min(channel_idx, len(key_stats) - 1)
                out[channel_idx] = self._normalize_channel(arr[channel_idx], key_stats[stats_idx])
            d[key] = out

        if self.delete_stats:
            d.pop(self.stats_key, None)
        return d


class PreserveBodyRegionLabelsD(MapTransform):
    """Keep the original body labels before ``body_seg`` becomes a binary mask."""

    def __init__(
        self,
        source_key: str = "body_seg",
        output_key: str = "body_region_seg",
    ):
        super().__init__(keys=[source_key], allow_missing_keys=True)
        self.source_key = source_key
        self.output_key = output_key

    def __call__(self, data):
        d = dict(data)
        if self.source_key in d:
            d[self.output_key] = _as_4d(d[self.source_key]).clone()
        return d


class CreateNACActivityProxyD(MapTransform):
    """Create a positive NAC-PET activity proxy before image normalization.

    The cropped raw NAC-PET is clipped with input-only subject percentiles and
    divided by its full-subject positive-voxel mean. This preserves relative
    uptake, keeps true zero background at zero and avoids retaining another
    full-volume tensor in each DataLoader worker.
    """

    def __init__(
        self,
        source_key: str = "nacpet",
        output_key: str = "activity_proxy",
        stats_key: str = "_subject_norm_stats",
        max_relative_activity: float = 20.0,
    ):
        super().__init__(keys=[source_key], allow_missing_keys=True)
        self.source_key = source_key
        self.output_key = output_key
        self.stats_key = stats_key
        self.max_relative_activity = float(max_relative_activity)

    def __call__(self, data):
        d = dict(data)
        if self.source_key not in d:
            return d
        all_stats = d.get(self.stats_key)
        if not isinstance(all_stats, dict) or self.source_key not in all_stats:
            raise RuntimeError(
                "CreateNACActivityProxyD requires subject normalization statistics"
            )

        raw = _as_4d(d[self.source_key]).float()
        stats = all_stats[self.source_key][0]
        lower = float(stats.get("lower", 0.0))
        upper = float(stats.get("upper", lower))
        positive = torch.isfinite(raw) & (raw > 0)
        activity = torch.clamp(raw, min=lower, max=upper)
        activity = torch.where(positive, activity.clamp_min(0.0), torch.zeros_like(activity))
        reference = max(float(stats.get("activity_mean", 1.0)), 1e-6)
        activity = torch.nan_to_num(
            activity / reference,
            nan=0.0,
            posinf=self.max_relative_activity,
            neginf=0.0,
        )
        d[self.output_key] = activity.clamp(
            min=0.0,
            max=self.max_relative_activity,
        )
        return d


class CreateWholeBodyContextD(MapTransform):
    """Build a compact, inference-available whole-body overview before crop.

    The model receives normalized NAC-PET, normalized topogram and absolute Z
    coordinate. Label-derived tensors are emitted only as auxiliary targets;
    they are never concatenated to the model input.
    """

    def __init__(
        self,
        spatial_size: Sequence[int] = (96, 96, 128),
        spacing_mm: Sequence[float] = (1.5234, 1.5234, 2.0),
        stats_key: str = "_subject_norm_stats",
        input_key: str = "global_input",
    ):
        super().__init__(keys=["nacpet", "topogram"], allow_missing_keys=True)
        if len(spatial_size) != 3 or any(int(value) <= 0 for value in spatial_size):
            raise ValueError(f"global context spatial_size must be three positive values, got {spatial_size}")
        if len(spacing_mm) != 3 or any(float(value) <= 0 for value in spacing_mm):
            raise ValueError(f"global context spacing_mm must be three positive values, got {spacing_mm}")
        self.spatial_size = tuple(int(value) for value in spatial_size)
        self.spacing_mm = tuple(float(value) for value in spacing_mm)
        self.stats_key = stats_key
        self.input_key = input_key

    @staticmethod
    def _normalize_channel(channel: torch.Tensor, stats: dict) -> torch.Tensor:
        finite = torch.isfinite(channel)
        lower = float(stats.get("lower", 0.0))
        upper = float(stats.get("upper", lower))
        mean = float(stats.get("mean", 0.0))
        std = max(float(stats.get("std", 1.0)), 1e-8)
        clipped = torch.clamp(channel, min=lower, max=upper)
        if stats.get("log1p", False):
            clipped = torch.log1p(torch.clamp(clipped, min=0.0))
        normalized = (clipped - mean) / std
        return torch.where(finite, normalized, torch.zeros_like(normalized))

    def _normalized_key(self, d, key: str, fallback: torch.Tensor | None = None) -> torch.Tensor:
        source = d.get(key)
        if source is None:
            if fallback is None:
                raise RuntimeError(f"Missing required whole-body context key '{key}'")
            return torch.zeros_like(fallback)
        arr = _as_4d(source)
        all_stats = d.get(self.stats_key, {})
        key_stats = all_stats.get(key)
        if not key_stats:
            raise RuntimeError(f"Missing subject normalization statistics for whole-body key '{key}'")
        out = torch.zeros_like(arr, dtype=torch.float32)
        for channel_idx in range(arr.shape[0]):
            out[channel_idx] = self._normalize_channel(arr[channel_idx], key_stats[min(channel_idx, len(key_stats) - 1)])
        return out

    def _resize(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        kwargs = {"size": self.spatial_size, "mode": mode}
        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        return F.interpolate(x.unsqueeze(0).float(), **kwargs).squeeze(0)

    def _activity_proxy(self, d) -> torch.Tensor:
        raw = _as_4d(d["nacpet"]).float()
        all_stats = d.get(self.stats_key, {})
        key_stats = all_stats.get("nacpet")
        if not key_stats:
            raise RuntimeError(
                "Whole-body activity proxy requires subject normalization statistics for NAC-PET"
            )
        stats = key_stats[0]
        lower = float(stats.get("lower", 0.0))
        upper = float(stats.get("upper", lower))
        positive = torch.isfinite(raw) & (raw > 0)
        activity = torch.clamp(raw, min=lower, max=upper)
        activity = torch.where(positive, activity.clamp_min(0.0), torch.zeros_like(activity))
        reference = max(float(stats.get("activity_mean", 1.0)), 1e-6)
        return torch.nan_to_num(
            activity / reference,
            nan=0.0,
            posinf=20.0,
            neginf=0.0,
        ).clamp(0.0, 20.0)

    def __call__(self, data):
        d = dict(data)
        nacpet_full = self._normalized_key(d, "nacpet")
        original_shape = tuple(int(value) for value in nacpet_full.shape[1:])
        nacpet = self._resize(nacpet_full, "trilinear")
        topogram_full = self._normalized_key(d, "topogram", fallback=torch.zeros_like(nacpet_full))
        topogram = self._resize(topogram_full, "trilinear")

        z = torch.linspace(-1.0, 1.0, self.spatial_size[2], dtype=torch.float32, device=nacpet.device)
        z_coord = z.view(1, 1, 1, -1).expand((1, *self.spatial_size)).clone()
        d[self.input_key] = torch.cat([nacpet, topogram, z_coord], dim=0).float()

        source_spacing = _spacing_from_image(d.get("ct", d.get("nacpet")), self.spacing_mm)
        effective_spacing = [
            source_spacing[axis] * float(original_shape[axis]) / float(self.spatial_size[axis])
            for axis in range(3)
        ]
        d["global_spacing_mm"] = torch.tensor(effective_spacing, dtype=torch.float32)
        d["global_activity_proxy"] = self._resize(self._activity_proxy(d), "trilinear")
        if "ct" in d:
            d["global_target_hu"] = self._resize(_as_4d(d["ct"]), "area")

        label_specs = {
            "prediction_mask": "global_prediction_mask",
            "body_seg": "global_body_mask",
            "official_eval_mask": "global_official_eval_mask",
        }
        for source_key, target_key in label_specs.items():
            if source_key in d:
                # Area resizing preserves fractional model responsibility at
                # coarse voxel boundaries instead of turning them fully on/off.
                d[target_key] = self._resize(_as_4d(d[source_key]), "area").clamp(0.0, 1.0)
        categorical_specs = {
            "organ_seg": "global_organ_seg",
            "body_region_seg": "global_body_region_seg",
        }
        for source_key, target_key in categorical_specs.items():
            if source_key in d:
                d[target_key] = self._resize(_as_4d(d[source_key]), "nearest")
        return d


class CreateWholeBodyPhysicsTargetsD(MapTransform):
    """Load/cache full-resolution HU -> mu -> smooth -> area targets."""

    def __init__(
        self,
        spatial_size: Sequence[int],
        spacing_mm: Sequence[float],
        fwhm_mm: float = 4.0,
        cache_dir: str | None = None,
        missing_policy: str = "compute",
    ):
        super().__init__(keys=["ct"], allow_missing_keys=True)
        self.spatial_size = tuple(int(value) for value in spatial_size)
        self.spacing_mm = tuple(float(value) for value in spacing_mm)
        self.fwhm_mm = float(fwhm_mm)
        self.cache_dir = str(Path(cache_dir)) if cache_dir else None
        self.missing_policy = str(missing_policy)

    def __call__(self, data):
        d = dict(data)
        if "ct" not in d:
            return d
        subject_id = str(d.get("subject_id", "unknown_subject"))
        source = d["ct"]
        spacing = _spacing_from_image(source, self.spacing_mm)
        mu_raw, mu_smooth, cache_hit = load_or_compute_global_mu_targets(
            _as_4d(source),
            subject_id=subject_id,
            spacing_mm=spacing,
            spatial_size=self.spatial_size,
            fwhm_mm=self.fwhm_mm,
            cache_dir=self.cache_dir,
            missing_policy=self.missing_policy,
        )
        d["global_target_mu_raw"] = mu_raw
        d["global_target_mu_smooth"] = mu_smooth
        d["global_physics_cache_hit"] = bool(cache_hit)
        return d


class CreateExactPETSupervisionD(MapTransform):
    """Replace NAC approximations with evaluator-exact PET supervision when available.

    PET targets remain loss-only. They are physically resampled from the PET
    NIfTI grid to the compact CT grid and never enter ``input`` or
    ``global_input``.
    """

    def __init__(
        self,
        spatial_size: Sequence[int],
        cache_dir: str | None,
        exclusion_cm: float = 4.0,
        mask_supersample_factor: int = 2,
        enabled: bool = True,
    ):
        super().__init__(keys=PET_SUPERVISION_PATH_KEYS, allow_missing_keys=True)
        self.spatial_size = tuple(int(value) for value in spatial_size)
        self.cache_dir = cache_dir
        self.exclusion_cm = float(exclusion_cm)
        self.mask_supersample_factor = int(mask_supersample_factor)
        self.enabled = bool(enabled)
        self._memory_cache: dict[str, dict] = {}

    def __call__(self, data):
        d = dict(data)
        d["global_activity_is_gt_pet"] = torch.tensor([0.0], dtype=torch.float32)
        d["global_pet_mask_cache_hit"] = torch.tensor([0.0], dtype=torch.float32)
        if not self.enabled or not all(key in d for key in PET_SUPERVISION_PATH_KEYS):
            return d
        if "ct" not in d:
            raise RuntimeError("Exact PET supervision requires the CT target grid")

        ct = d["ct"]
        ct_affine = getattr(ct, "affine", None)
        if ct_affine is None:
            raise RuntimeError("CT MetaTensor affine is required for PET-to-CT resampling")
        subject_id = str(d.get("subject_id", "unknown_subject"))
        result = self._memory_cache.get(subject_id)
        if result is None:
            result = load_or_create_exact_pet_supervision(
                subject_id=subject_id,
                pet_paths={key: str(d[key]) for key in PET_SUPERVISION_PATH_KEYS},
                source_ct_shape=tuple(int(value) for value in _as_4d(ct).shape[1:]),
                source_ct_affine=torch.as_tensor(ct_affine).detach().cpu().numpy(),
                compact_shape=self.spatial_size,
                cache_dir=self.cache_dir,
                exclusion_cm=self.exclusion_cm,
                mask_supersample_factor=self.mask_supersample_factor,
            )
            self._memory_cache[subject_id] = result
        d["global_activity_proxy"] = result["activity_suv"]
        d["global_pet_eval_mask"] = result["eval_mask"]
        d["global_pet_body_mask"] = result["body_mask"]
        d["global_pet_region_masks"] = result["region_masks"]
        d["global_activity_is_gt_pet"] = torch.tensor([1.0], dtype=torch.float32)
        d["global_pet_mask_cache_hit"] = torch.tensor(
            [1.0 if result["cache_hit"] else 0.0], dtype=torch.float32
        )
        return d


class PercentileClipNormalizeD(MapTransform):
    def __init__(
        self,
        keys,
        lower=1.0,
        upper=99.0,
        nonzero=False,
        mask_key: str | None = None,
        log1p=False,
        eps=1e-8,
        max_quantile_samples=1_000_000,
        allow_missing_keys=True,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.lower = lower
        self.upper = upper
        self.nonzero = nonzero
        self.mask_key = mask_key
        self.log1p = log1p
        self.eps = eps
        self.max_quantile_samples = max_quantile_samples

    def _select_quantile_values(self, values):
        if values.numel() <= self.max_quantile_samples:
            return values
        step = max(1, values.numel() // self.max_quantile_samples)
        return values[::step][: self.max_quantile_samples]

    def __call__(self, data):
        d = dict(data)
        mask = None
        if self.mask_key and self.mask_key in d:
            mask = _as_4d(d[self.mask_key]) > 0

        for key in self.key_iterator(d):
            try:
                arr = _as_4d(d[key])

                out = torch.zeros_like(arr, dtype=torch.float32)
                for channel_idx in range(arr.shape[0]):
                    channel = arr[channel_idx]
                    finite = torch.isfinite(channel)
                    valid = finite & (channel != 0) if self.nonzero else finite
                    if mask is not None and mask.shape[1:] == channel.shape:
                        valid = valid & mask[0]
                    values = channel[valid]
                    if values.numel() == 0:
                        continue

                    quantile_values = self._select_quantile_values(values)
                    lower = torch.quantile(quantile_values, self.lower / 100.0)
                    upper = torch.quantile(quantile_values, self.upper / 100.0)
                    clipped = torch.clamp(channel, min=lower, max=upper)
                    if self.log1p:
                        clipped = torch.log1p(torch.clamp(clipped, min=0.0))
                    norm_values = clipped[valid]
                    mean = norm_values.mean()
                    std = norm_values.std(unbiased=False)
                    if float(std.item()) <= self.eps:
                        normalized = clipped - mean
                    else:
                        normalized = (clipped - mean) / std
                    out[channel_idx] = torch.where(finite, normalized, torch.zeros_like(normalized))

                d[key] = out
            except Exception as exc:
                raw = d.get(key)
                raw_shape = tuple(raw.shape) if hasattr(raw, "shape") else "<no-shape>"
                raw_dtype = getattr(raw, "dtype", type(raw).__name__)
                raise RuntimeError(
                    f"PercentileClipNormalizeD failed for key='{key}', "
                    f"shape={raw_shape}, dtype={raw_dtype}, nonzero={self.nonzero}"
                ) from exc
        return d


class CreateDixonDerivedChannelsD(MapTransform):
    """Create Dixon-derived images in raw aligned MRI space before normalization."""

    def __init__(self, derived_channels: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys=["mri_in", "mri_out"], allow_missing_keys=allow_missing_keys)
        self.derived_channels = tuple(derived_channels)
        unsupported = set(self.derived_channels) - {"mri_sum", "mri_absdiff"}
        if unsupported:
            raise ValueError(f"Unsupported Dixon-derived channels: {sorted(unsupported)}")

    def __call__(self, data):
        d = dict(data)
        if not self.derived_channels:
            return d

        mri_in = _as_4d(d["mri_in"])
        mri_out = _match_spatial(_as_4d(d["mri_out"]), mri_in.shape[1:])
        if "mri_sum" in self.derived_channels:
            d["mri_sum"] = mri_in + mri_out
        if "mri_absdiff" in self.derived_channels:
            d["mri_absdiff"] = torch.abs(mri_in - mri_out)
        return d


class CreateInputChannelsD(MapTransform):
    def __init__(
        self,
        input_channels: Sequence[str] | None = None,
        train: bool = False,
        mri_dropout_prob: float = 0.0,
        topogram_dropout_prob: float = 0.0,
        mri_shift_prob: float = 0.0,
        mri_shift_voxels: int = 2,
    ):
        super().__init__(keys=["nacpet"], allow_missing_keys=False)
        self.input_channels = list(input_channels or DEFAULT_INPUT_CHANNELS)
        self.train = train
        self.mri_dropout_prob = mri_dropout_prob
        self.topogram_dropout_prob = topogram_dropout_prob
        self.mri_shift_prob = mri_shift_prob
        self.mri_shift_voxels = int(mri_shift_voxels)

    def _zeros_like_ref(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, *ref.shape[1:]), dtype=torch.float32, device=ref.device)

    def _get_body_like(self, d, ref: torch.Tensor) -> torch.Tensor:
        if "prediction_mask" in d:
            mask = _as_4d(d["prediction_mask"])
        elif "body_seg" in d:
            mask = _as_4d(d["body_seg"])
        else:
            mask = torch.isfinite(ref).float()
        return (_match_spatial(mask, ref.shape[1:]) > 0).float()

    def _get_channel(self, name: str, d, ref: torch.Tensor) -> torch.Tensor:
        if name in {"nacpet", "mri_in", "mri_out", "mri_sum", "mri_absdiff"}:
            if name not in d:
                raise KeyError(
                    f"Input channel '{name}' is missing. Dixon-derived channels must be created "
                    "before subject-level normalization."
                )
            return _match_spatial(_as_4d(d[name]), ref.shape[1:])
        if name == "topogram":
            if "topogram" not in d:
                return self._zeros_like_ref(ref)
            return _match_spatial(_as_4d(d["topogram"]), ref.shape[1:])
        if name in {"y_coord", "z_coord"}:
            # NIfTI arrays are loaded as x/y/z. If this is a cropped patch,
            # keep coordinates absolute in the original subject volume so
            # train and sliding-window inference use consistent positions.
            spatial_shape = tuple(int(v) for v in ref.shape[1:])
            crop_start = d.get("train_crop_start", d.get("val_crop_start"))
            original_shape = d.get("crop_original_shape")
            axis = 1 if name == "y_coord" else 2

            if torch.is_tensor(crop_start):
                coord_start = int(crop_start.detach().cpu().view(-1)[axis].item())
            else:
                coord_start = 0

            if torch.is_tensor(original_shape):
                coord_size = int(original_shape.detach().cpu().view(-1)[axis].item())
            else:
                coord_size = spatial_shape[axis]

            indices = torch.arange(
                coord_start,
                coord_start + spatial_shape[axis],
                dtype=torch.float32,
                device=ref.device,
            )
            if coord_size <= 1:
                coord = torch.zeros_like(indices)
            else:
                coord = 2.0 * indices / float(coord_size - 1) - 1.0
            if name == "y_coord":
                return coord.view(1, 1, -1, 1).expand((1, *spatial_shape)).clone()
            return coord.view(1, 1, 1, -1).expand((1, *spatial_shape)).clone()
        if name in {"body_mask", "prediction_mask"}:
            return self._get_body_like(d, ref)
        raise ValueError(f"Unknown input channel: {name}")

    def __call__(self, data):
        d = dict(data)
        ref = _as_4d(d["nacpet"])
        channels = [self._get_channel(name, d, ref) for name in self.input_channels]
        x = torch.cat(channels, dim=0).float()

        if self.train and self.mri_shift_prob > 0 and random.random() < self.mri_shift_prob:
            shifts = [random.randint(-self.mri_shift_voxels, self.mri_shift_voxels) for _ in range(3)]
            for idx, name in enumerate(self.input_channels):
                if name.startswith("mri_"):
                    x[idx] = torch.roll(x[idx], shifts=tuple(shifts), dims=(0, 1, 2))

        if self.train and self.mri_dropout_prob > 0 and random.random() < self.mri_dropout_prob:
            for idx, name in enumerate(self.input_channels):
                if name.startswith("mri_"):
                    x[idx].zero_()

        if self.train and self.topogram_dropout_prob > 0 and random.random() < self.topogram_dropout_prob:
            for idx, name in enumerate(self.input_channels):
                if name == "topogram":
                    x[idx].zero_()

        d["input"] = x
        return d


class DropKeysD(MapTransform):
    """Discard raw modality tensors after the model input has been assembled."""

    def __init__(self, keys, allow_missing_keys: bool = True):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d.pop(key, None)
        return d


def _input_channels_from_cfg(cfg) -> list[str]:
    return list(cfg.get("input", {}).get("channels", DEFAULT_INPUT_CHANNELS))


def _raw_image_keys_from_input_channels(input_channels: Sequence[str]) -> list[str]:
    required = {"nacpet"}
    channel_set = set(input_channels)

    if {"mri_in", "mri_sum", "mri_absdiff"} & channel_set:
        required.add("mri_in")
    if {"mri_out", "mri_sum", "mri_absdiff"} & channel_set:
        required.add("mri_out")
    if "topogram" in channel_set:
        required.add("topogram")

    return [key for key in BASE_IMAGE_KEYS + OPTIONAL_IMAGE_KEYS if key in required]


def _derived_input_keys_from_input_channels(input_channels: Sequence[str]) -> list[str]:
    channel_set = set(input_channels)
    return [name for name in ("mri_sum", "mri_absdiff") if name in channel_set]


def _load_keys_from_cfg(cfg, include_labels: bool) -> list[str]:
    keys = _raw_image_keys_from_input_channels(_input_channels_from_cfg(cfg))
    if cfg.get("model", {}).get("global_context", {}).get("enabled", False):
        for key in ("nacpet", "topogram"):
            if key not in keys:
                keys.append(key)
    if include_labels:
        keys = keys + LABEL_KEYS
    return keys


def _augmentation_cfg(cfg) -> dict:
    return cfg.get("augmentation", {})


def _percentile_pair(cfg, key: str, default: Sequence[float]) -> tuple[float, float]:
    values = cfg.get("preprocess", {}).get(key, default)
    if len(values) != 2:
        raise ValueError(f"Expected preprocess.{key} to contain [lower, upper], got {values}")
    return float(values[0]), float(values[1])


def _normalization_specs_from_cfg(cfg) -> dict[str, dict]:
    nac_lower, nac_upper = _percentile_pair(cfg, "nacpet_percentiles", [0.5, 99.5])
    mri_lower, mri_upper = _percentile_pair(cfg, "mri_percentiles", [0.5, 99.5])
    topo_lower, topo_upper = _percentile_pair(cfg, "topogram_percentiles", [0.5, 99.5])
    return {
        # Input-only rule: use finite positive PET voxels. Do not use prediction_mask/CT labels.
        "nacpet": {"lower": nac_lower, "upper": nac_upper, "positive": True, "log1p": True},
        # MRI zero background is excluded from stats, but no label-derived mask is used.
        "mri_in": {"lower": mri_lower, "upper": mri_upper, "nonzero": True},
        "mri_out": {"lower": mri_lower, "upper": mri_upper, "nonzero": True},
        # Derived channels are calculated from raw aligned Dixon images before this
        # subject-level normalization stage, then receive independent statistics.
        "mri_sum": {"lower": mri_lower, "upper": mri_upper, "nonzero": True},
        "mri_absdiff": {"lower": mri_lower, "upper": mri_upper, "nonzero": True},
        # Topogram is optional; if used, avoid zero padding dominating subject-level stats.
        "topogram": {"lower": topo_lower, "upper": topo_upper, "nonzero": True},
    }


def _normalization_keys_from_cfg(cfg) -> list[str]:
    input_channels = _input_channels_from_cfg(cfg)
    keys = _raw_image_keys_from_input_channels(input_channels) + _derived_input_keys_from_input_channels(input_channels)
    if cfg.get("model", {}).get("global_context", {}).get("enabled", False):
        for key in ("nacpet", "topogram"):
            if key not in keys:
                keys.append(key)
    return keys


def get_train_transforms(cfg):
    training = cfg["training"]
    aug = _augmentation_cfg(cfg)
    load_keys = _load_keys_from_cfg(cfg, include_labels=True)
    derived_keys = _derived_input_keys_from_input_channels(_input_channels_from_cfg(cfg))
    crop_keys = load_keys + derived_keys + ["official_eval_mask", "body_region_seg"]
    norm_keys = _normalization_keys_from_cfg(cfg)
    global_cfg = cfg.get("model", {}).get("global_context", {})
    supervise_global_aux = bool(global_cfg.get("supervise_auxiliary", True))
    data_cfg = cfg.get("data", {})
    spacing_mm = cfg.get("loss", {}).get("spacing_mm", [1.5234, 1.5234, 2.0])
    mu_fwhm_mm = float(cfg.get("loss", {}).get("mu_fwhm_mm", 4.0))
    exclusion_cm = cfg.get("loss", {}).get("official_exclusion_cm", 4.0)
    pet_supervision_cfg = cfg.get("pet_supervision", {})
    return Compose(
        [
            LoadImaged(keys=load_keys, allow_missing_keys=True),
            EnsureChannelFirstd(keys=load_keys, allow_missing_keys=True),
            PreserveBodyRegionLabelsD(),
            Lambdad(keys=["prediction_mask", "body_seg"], func=_binarize_mask, allow_missing_keys=True),
            CreateDixonDerivedChannelsD(derived_keys, allow_missing_keys=True),
            ComputeSubjectNormalizationStatsD(
                keys=norm_keys,
                specs=_normalization_specs_from_cfg(cfg),
                allow_missing_keys=True,
            ),
            CreateOfficialCTEvalMaskD(
                spacing_z_mm=float(spacing_mm[2]),
                exclusion_cm=float(exclusion_cm),
            ),
            CreateWholeBodyContextD(
                spatial_size=global_cfg.get("spatial_size", [96, 96, 128]),
                spacing_mm=spacing_mm,
            ),
            CreateExactPETSupervisionD(
                spatial_size=global_cfg.get("spatial_size", [96, 96, 128]),
                cache_dir=pet_supervision_cfg.get("cache_dir"),
                exclusion_cm=float(exclusion_cm),
                mask_supersample_factor=pet_supervision_cfg.get(
                    "mask_supersample_factor", 2
                ),
                enabled=pet_supervision_cfg.get("enabled", False),
            ),
            *(
                [
                    CreateWholeBodyPhysicsTargetsD(
                        spatial_size=global_cfg.get("spatial_size", [96, 96, 128]),
                        spacing_mm=spacing_mm,
                        fwhm_mm=mu_fwhm_mm,
                        cache_dir=data_cfg.get("physics_cache_dir"),
                        missing_policy=data_cfg.get("physics_cache_missing", "compute"),
                    )
                ]
                if supervise_global_aux
                else []
            ),
            RandomMaskCropSamplesD(
                keys=crop_keys,
                roi_size=training["patch_size"],
                num_samples=training["train_num_samples"],
                mask_key=training.get("crop_mask_key", "prediction_mask"),
                min_mask_fraction=training.get("min_mask_fraction", 0.35),
                max_crop_attempts=training.get("max_crop_attempts", 16),
                region_sampling=training.get("region_sampling", {}),
                allow_missing_keys=True,
            ),
            CreateNACActivityProxyD(),
            ApplySubjectNormalizationStatsD(
                keys=norm_keys,
                delete_stats=False,
                allow_missing_keys=True,
            ),
            CreateInputChannelsD(
                input_channels=_input_channels_from_cfg(cfg),
                train=True,
                mri_dropout_prob=aug.get("mri_dropout_prob", 0.0),
                topogram_dropout_prob=aug.get("topogram_dropout_prob", 0.0),
                mri_shift_prob=aug.get("mri_shift_prob", 0.0),
                mri_shift_voxels=aug.get("mri_shift_voxels", 2),
            ),
            DropKeysD(
                keys=norm_keys + PET_SUPERVISION_PATH_KEYS + ["_subject_norm_stats"],
                allow_missing_keys=True,
            ),
            EnsureTyped(
                keys=[
                    "input", "ct", "prediction_mask", "body_seg", "body_region_seg",
                    "organ_seg", "official_eval_mask", "activity_proxy",
                    "global_input", "global_target_mu_raw", "global_target_mu_smooth",
                    "global_prediction_mask", "global_body_mask", "global_official_eval_mask",
                    "global_organ_seg", "global_body_region_seg", "global_target_hu",
                    "global_activity_proxy", "global_spacing_mm",
                    "global_pet_eval_mask", "global_pet_body_mask",
                    "global_pet_region_masks", "global_activity_is_gt_pet",
                    "global_pet_mask_cache_hit",
                ],
                allow_missing_keys=True,
            ),
        ]
    )


def get_val_transforms(cfg):
    training = cfg["training"]
    load_keys = _load_keys_from_cfg(cfg, include_labels=True)
    derived_keys = _derived_input_keys_from_input_channels(_input_channels_from_cfg(cfg))
    crop_keys = load_keys + derived_keys + ["official_eval_mask", "body_region_seg"]
    norm_keys = _normalization_keys_from_cfg(cfg)
    global_cfg = cfg.get("model", {}).get("global_context", {})
    supervise_global_aux = bool(global_cfg.get("supervise_auxiliary", True))
    data_cfg = cfg.get("data", {})
    spacing_mm = cfg.get("loss", {}).get("spacing_mm", [1.5234, 1.5234, 2.0])
    mu_fwhm_mm = float(cfg.get("loss", {}).get("mu_fwhm_mm", 4.0))
    exclusion_cm = cfg.get("loss", {}).get("official_exclusion_cm", 4.0)
    pet_supervision_cfg = cfg.get("pet_supervision", {})
    return Compose(
        [
            LoadImaged(keys=load_keys, allow_missing_keys=True),
            EnsureChannelFirstd(keys=load_keys, allow_missing_keys=True),
            PreserveBodyRegionLabelsD(),
            Lambdad(keys=["prediction_mask", "body_seg"], func=_binarize_mask, allow_missing_keys=True),
            CreateDixonDerivedChannelsD(derived_keys, allow_missing_keys=True),
            ComputeSubjectNormalizationStatsD(
                keys=norm_keys,
                specs=_normalization_specs_from_cfg(cfg),
                allow_missing_keys=True,
            ),
            CreateOfficialCTEvalMaskD(
                spacing_z_mm=float(spacing_mm[2]),
                exclusion_cm=float(exclusion_cm),
            ),
            CreateWholeBodyContextD(
                spatial_size=global_cfg.get("spatial_size", [96, 96, 128]),
                spacing_mm=spacing_mm,
            ),
            CreateExactPETSupervisionD(
                spatial_size=global_cfg.get("spatial_size", [96, 96, 128]),
                cache_dir=pet_supervision_cfg.get("cache_dir"),
                exclusion_cm=float(exclusion_cm),
                mask_supersample_factor=pet_supervision_cfg.get(
                    "mask_supersample_factor", 2
                ),
                enabled=pet_supervision_cfg.get("enabled", False),
            ),
            *(
                [
                    CreateWholeBodyPhysicsTargetsD(
                        spatial_size=global_cfg.get("spatial_size", [96, 96, 128]),
                        spacing_mm=spacing_mm,
                        fwhm_mm=mu_fwhm_mm,
                        cache_dir=data_cfg.get("physics_cache_dir"),
                        missing_policy=data_cfg.get("physics_cache_missing", "compute"),
                    )
                ]
                if supervise_global_aux
                else []
            ),
            DeterministicSpatialCropSamplesD(
                keys=crop_keys,
                roi_size=training["patch_size"],
                num_samples=training["val_num_samples"],
                mask_key=cfg.get("validation", {}).get("crop_mask_key", "prediction_mask"),
                allow_missing_keys=True,
            ),
            CreateNACActivityProxyD(),
            ApplySubjectNormalizationStatsD(
                keys=norm_keys,
                delete_stats=False,
                allow_missing_keys=True,
            ),
            CreateInputChannelsD(input_channels=_input_channels_from_cfg(cfg), train=False),
            DropKeysD(
                keys=norm_keys + PET_SUPERVISION_PATH_KEYS + ["_subject_norm_stats"],
                allow_missing_keys=True,
            ),
            EnsureTyped(
                keys=[
                    "input", "ct", "prediction_mask", "body_seg", "body_region_seg",
                    "organ_seg", "official_eval_mask", "activity_proxy",
                    "global_input", "global_target_mu_raw", "global_target_mu_smooth",
                    "global_prediction_mask", "global_body_mask", "global_official_eval_mask",
                    "global_organ_seg", "global_body_region_seg", "global_target_hu",
                    "global_activity_proxy", "global_spacing_mm",
                    "global_pet_eval_mask", "global_pet_body_mask",
                    "global_pet_region_masks", "global_activity_is_gt_pet",
                    "global_pet_mask_cache_hit",
                ],
                allow_missing_keys=True,
            ),
        ]
    )


def get_inference_transforms(cfg=None):
    cfg = cfg or {}
    load_keys = _load_keys_from_cfg(cfg, include_labels=False)
    derived_keys = _derived_input_keys_from_input_channels(_input_channels_from_cfg(cfg))
    norm_keys = _normalization_keys_from_cfg(cfg)
    global_cfg = cfg.get("model", {}).get("global_context", {})
    spacing_mm = cfg.get("loss", {}).get("spacing_mm", [1.5234, 1.5234, 2.0])
    return Compose(
        [
            LoadImaged(keys=load_keys, allow_missing_keys=True),
            EnsureChannelFirstd(keys=load_keys, allow_missing_keys=True),
            CreateDixonDerivedChannelsD(derived_keys, allow_missing_keys=True),
            ComputeSubjectNormalizationStatsD(
                keys=norm_keys,
                specs=_normalization_specs_from_cfg(cfg),
                allow_missing_keys=True,
            ),
            CreateWholeBodyContextD(
                spatial_size=global_cfg.get("spatial_size", [96, 96, 128]),
                spacing_mm=spacing_mm,
            ),
            ApplySubjectNormalizationStatsD(keys=norm_keys, allow_missing_keys=True),
            CreateInputChannelsD(input_channels=_input_channels_from_cfg(cfg), train=False),
            DropKeysD(keys=norm_keys, allow_missing_keys=True),
            EnsureTyped(keys=["input", "global_input", "global_spacing_mm"], allow_missing_keys=True),
        ]
    )
