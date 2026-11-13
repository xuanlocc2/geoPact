from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib

from dataset import get_dataset
from pet_supervision import PET_REGION_NAMES, load_or_create_exact_pet_supervision
from train import load_config, prepare_pet_supervision_sampling, split_dataset
from unet import build_model


EXPECTED_SUBJECT_COUNT = 75
EXPECTED_MONITOR_IDS = {"sub-000", "sub-001"}
EXPECTED_PET_IDS = {
    "sub-000",
    "sub-001",
    "sub-002",
    "sub-005",
    "sub-006",
    "sub-008",
    "sub-013",
    "sub-014",
}
EXPECTED_MODEL_PARAMETERS = 20_579_395


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _build_model(cfg):
    metadata_cfg = cfg.get("metadata", {})
    model_cfg = cfg["model"]
    return build_model(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_channels=model_cfg["base_channels"],
        dropout=model_cfg["dropout"],
        metadata_dim=len(metadata_cfg.get("keys", [])),
        use_metadata_film=metadata_cfg.get("use_film", False),
        model_name=model_cfg["name"],
        geometry_in_channels=model_cfg.get("geometry_in_channels"),
        geometry_channel_indices=model_cfg.get("geometry_channel_indices"),
        topogram_channel_index=model_cfg.get("topogram_channel_index", -1),
        topogram_gate_init_bias=model_cfg.get("topogram_gate_init_bias", -2.0),
        topogram_conditioned=model_cfg.get("topogram_conditioned"),
        mri_channel_indices=model_cfg.get("mri_channel_indices"),
        mri_base_channels=model_cfg.get("mri_base_channels"),
        mri_gate_init_bias=model_cfg.get("mri_gate_init_bias", -2.0),
        global_context=model_cfg.get("global_context"),
    )


def _build_exact_pet_caches(train_data, cfg) -> None:
    pet_cfg = cfg["pet_supervision"]
    compact_shape = cfg["model"]["global_context"]["spatial_size"]
    exclusion_cm = float(cfg["loss"].get("official_exclusion_cm", 4.0))
    pet_items = [item for item in train_data if item.get("has_pet_supervision", False)]

    for index, item in enumerate(pet_items, start=1):
        ct_img = nib.load(item["ct"])
        result = load_or_create_exact_pet_supervision(
            subject_id=item["subject_id"],
            pet_paths={
                key: item[key]
                for key in ("pet_gt", "pet_body_seg", "pet_organ_seg", "pet_tissue_seg")
            },
            source_ct_shape=ct_img.shape[:3],
            source_ct_affine=ct_img.affine,
            compact_shape=compact_shape,
            cache_dir=pet_cfg["cache_dir"],
            exclusion_cm=exclusion_cm,
            mask_supersample_factor=int(pet_cfg.get("mask_supersample_factor", 2)),
        )
        region_sums = result["region_masks"].reshape(len(PET_REGION_NAMES), -1).sum(dim=1)
        _require(bool((region_sums > 0).all()), f"Empty compact PET region for {item['subject_id']}")
        _require(float(result["activity_suv"].sum()) > 0.0, f"Empty GT SUV for {item['subject_id']}")
        print(
            f"PET cache [{index}/{len(pet_items)}] {item['subject_id']}: "
            f"cache_hit={bool(result['cache_hit'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast V4.5 production preflight")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_v4_5.yaml")))
    parser.add_argument(
        "--build-pet-cache",
        action="store_true",
        help="Load/resample all eight PET-labelled subjects before training.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["data"]["data_dir"])
    _require(data_dir.is_dir(), f"Training data directory does not exist: {data_dir}")

    all_data = get_dataset(str(data_dir))
    _require(
        len(all_data) == EXPECTED_SUBJECT_COUNT,
        f"Expected {EXPECTED_SUBJECT_COUNT} complete subjects, found {len(all_data)}",
    )
    train_data, monitor_data = split_dataset(all_data, cfg)
    train_ids = {item["subject_id"] for item in train_data}
    monitor_ids = {item["subject_id"] for item in monitor_data}
    pet_ids = {
        item["subject_id"] for item in train_data if item.get("has_pet_supervision", False)
    }
    _require(len(train_data) == 75 and len(train_ids) == 75, "Train split is not 75 unique subjects")
    _require(monitor_ids == EXPECTED_MONITOR_IDS, f"Unexpected train-monitor IDs: {sorted(monitor_ids)}")
    _require(monitor_ids.issubset(train_ids), "Train-monitor subjects do not receive gradients")
    _require(pet_ids == EXPECTED_PET_IDS, f"Unexpected PET-labelled IDs: {sorted(pet_ids)}")

    sampled = prepare_pet_supervision_sampling(train_data, monitor_data, cfg)
    _require(len(sampled) == 83, "Expected 75 unique records plus 8 PET replicas")
    _require(
        sum(bool(item.get("has_pet_supervision", False)) for item in sampled) == 16,
        "Expected exactly sixteen PET-supervised appearances per epoch",
    )
    replica_ids = [
        item["subject_id"] for item in sampled if item.get("pet_oversample_replica")
    ]
    _require(
        set(replica_ids) == EXPECTED_PET_IDS and len(replica_ids) == len(EXPECTED_PET_IDS),
        "Each PET-labelled subject must receive exactly one replica",
    )

    train_patch = tuple(int(value) for value in cfg["training"]["patch_size"])
    infer_patch = tuple(int(value) for value in cfg["inference"]["patch_size"])
    _require(train_patch == infer_patch, f"Train/inference ROI mismatch: {train_patch} != {infer_patch}")
    _require(cfg["model"].get("output_units") == "hu", "Model output must be HU")
    ray_cfg = cfg["loss"]["pet_evaluator_proxy"].get("ray_activity_weighting", {})
    _require(bool(ray_cfg.get("enabled", False)), "Controlled GT-PET ray weighting is disabled")
    _require(float(ray_cfg.get("gt_pet_alpha", -1.0)) == 0.15, "GT-PET ray alpha must be 0.15")
    _require(float(ray_cfg.get("nacpet_alpha", -1.0)) == 0.0, "NAC-PET ray alpha must remain zero")
    _require(
        [float(value) for value in ray_cfg.get("weight_clip", [])] == [0.8, 1.2],
        "Controlled ray weights must be clipped to [0.8, 1.2]",
    )

    model = _build_model(cfg)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _require(
        parameter_count == EXPECTED_MODEL_PARAMETERS,
        f"Unexpected model parameter count: {parameter_count}",
    )

    if args.build_pet_cache:
        _build_exact_pet_caches(train_data, cfg)

    print("V4.5 PREFLIGHT PASSED")
    print(f"data={data_dir}")
    print(f"train={len(train_data)} train_monitor={len(monitor_data)} PET={len(pet_ids)}")
    print(f"records_per_epoch={len(sampled)} train/infer_ROI={train_patch}")
    print(f"model_parameters={parameter_count}")
    print(f"output={cfg['output']['output_dir']}")


if __name__ == "__main__":
    main()
