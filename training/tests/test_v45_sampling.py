from __future__ import annotations

import pytest

from train import prepare_pet_supervision_sampling, split_dataset


PET_IDS = [
    "sub-000", "sub-001", "sub-002", "sub-005",
    "sub-006", "sub-008", "sub-013", "sub-014",
]


def _record(subject_id: str, has_pet: bool = False):
    return {"subject_id": subject_id, "has_pet_supervision": has_pet}


def test_pet_sampling_reaches_target_without_dropping_subjects():
    train = [_record(subject_id, subject_id in PET_IDS) for subject_id in PET_IDS]
    train.extend(_record(f"sub-other-{index:02d}") for index in range(67))
    val = [_record("sub-000", True), _record("sub-001", True)]
    config = {
        "pet_supervision": {
            "enabled": True,
            "expected_train_subjects": PET_IDS,
            "expected_val_subjects": ["sub-000", "sub-001"],
            "require_expected_subjects": True,
            "oversample": {"enabled": True, "target_fraction": 0.19},
        }
    }

    expanded = prepare_pet_supervision_sampling(train, val, config)
    assert len(expanded) == 83
    assert {item["subject_id"] for item in train}.issubset(
        {item["subject_id"] for item in expanded}
    )
    positive_count = sum(bool(item["has_pet_supervision"]) for item in expanded)
    assert positive_count == 16
    assert 0.19 <= positive_count / len(expanded) < 0.20
    replicas = [item for item in expanded if item.get("pet_oversample_replica")]
    assert len(replicas) == len(PET_IDS)
    assert {item["subject_id"] for item in replicas} == set(PET_IDS)


def test_train_monitor_subjects_are_included_in_gradient_split():
    all_data = [_record("sub-000", True), _record("sub-001", True), _record("sub-002", True)]
    config = {
        "data": {
            "val_subjects": ["sub-000", "sub-001"],
            "include_val_subjects_in_train": True,
        }
    }

    train, monitor = split_dataset(all_data, config)
    assert [item["subject_id"] for item in train] == ["sub-000", "sub-001", "sub-002"]
    assert [item["subject_id"] for item in monitor] == ["sub-000", "sub-001"]


def test_pet_sampling_fails_early_when_expected_label_is_missing():
    train = [_record(subject_id, True) for subject_id in PET_IDS[:-1]]
    val = [_record("sub-000", True), _record("sub-001", True)]
    config = {
        "pet_supervision": {
            "enabled": True,
            "expected_train_subjects": PET_IDS,
            "expected_val_subjects": ["sub-000", "sub-001"],
            "require_expected_subjects": True,
        }
    }

    with pytest.raises(RuntimeError, match="sub-014"):
        prepare_pet_supervision_sampling(train, val, config)
