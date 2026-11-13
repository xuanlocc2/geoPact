from __future__ import annotations

import json
import os
from typing import Dict, List


REQUIRED_FEATURE_KEYS = ("nacpet", "mri_in", "mri_out")
REQUIRED_LABEL_KEYS = ("ct", "prediction_mask")
PET_LABEL_FILENAMES = {
    "pet_gt": "pet.nii.gz",
    "pet_body_seg": "body_seg.nii.gz",
    "pet_organ_seg": "organ_seg.nii.gz",
    "pet_tissue_seg": "tissue_seg.nii.gz",
}


def _assert_exists(paths: Dict[str, str], required_keys) -> None:
    for key in required_keys:
        path = paths[key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file for '{key}': {path}")


def _add_if_exists(paths: Dict[str, str], key: str, path: str) -> None:
    if os.path.exists(path):
        paths[key] = path


def get_subject_features(features_dir: str) -> Dict[str, str]:
    paths = {
        "nacpet": os.path.join(features_dir, "nacpet.nii.gz"),
        "mri_in": os.path.join(features_dir, "mri_combined_in_phase.nii.gz"),
        "mri_out": os.path.join(features_dir, "mri_combined_out_phase.nii.gz"),
    }
    _assert_exists(paths, REQUIRED_FEATURE_KEYS)
    _add_if_exists(paths, "topogram", os.path.join(features_dir, "topogram.nii.gz"))
    _add_if_exists(paths, "metadata_json", os.path.join(features_dir, "metadata.json"))
    return paths


def get_subject_ct_labels(ct_label_dir: str) -> Dict[str, str]:
    paths = {
        "ct": os.path.join(ct_label_dir, "ct.nii.gz"),
        "prediction_mask": os.path.join(ct_label_dir, "prediction_mask.nii.gz"),
    }
    _assert_exists(paths, REQUIRED_LABEL_KEYS)
    _add_if_exists(paths, "body_seg", os.path.join(ct_label_dir, "body_seg.nii.gz"))
    _add_if_exists(paths, "organ_seg", os.path.join(ct_label_dir, "organ_seg.nii.gz"))
    return paths


def get_subject_pet_labels(pet_label_dir: str) -> Dict[str, str]:
    """Return PET-space evaluator targets only when the complete set exists."""
    candidate = {
        key: os.path.join(pet_label_dir, filename)
        for key, filename in PET_LABEL_FILENAMES.items()
    }
    present = {key: os.path.exists(path) for key, path in candidate.items()}
    if any(present.values()) and not all(present.values()):
        missing = [key for key, exists in present.items() if not exists]
        raise FileNotFoundError(
            f"Incomplete PET supervision under {pet_label_dir}; missing {missing}"
        )
    return candidate if all(present.values()) else {}


def load_metadata(metadata_path: str | None) -> Dict[str, float]:
    if not metadata_path or not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    sex = raw.get("sex", "")
    sex_value = 1.0 if str(sex).upper().startswith("M") else 0.0
    height = float(raw.get("height", 0.0) or 0.0)
    weight = float(raw.get("weight", 0.0) or 0.0)
    bmi = weight / (height * height) if height > 0 else 0.0
    return {
        "sex": sex_value,
        "age": float(raw.get("age", 0.0) or 0.0),
        "height": height,
        "weight": weight,
        "bmi": bmi,
    }


def get_dataset(data_dir: str) -> List[Dict[str, str]]:
    subjects: List[Dict[str, str]] = []
    for sub in sorted(os.listdir(data_dir)):
        subject_dir = os.path.join(data_dir, sub)
        if not os.path.isdir(subject_dir):
            continue

        features_dir = os.path.join(subject_dir, "features")
        ct_label_dir = os.path.join(subject_dir, "ct-label")
        pet_label_dir = os.path.join(subject_dir, "pet-label")
        if not os.path.isdir(features_dir) or not os.path.isdir(ct_label_dir):
            continue

        item = {"subject_id": sub, "subject_dir": subject_dir}
        item.update(get_subject_features(features_dir))
        item.update(get_subject_ct_labels(ct_label_dir))
        item.update(get_subject_pet_labels(pet_label_dir))
        item["has_pet_supervision"] = bool("pet_gt" in item)
        item.update({f"meta_{k}": v for k, v in load_metadata(item.get("metadata_json")).items()})
        subjects.append(item)

    if not subjects:
        raise RuntimeError(f"No training subjects found under {data_dir}")

    return subjects
