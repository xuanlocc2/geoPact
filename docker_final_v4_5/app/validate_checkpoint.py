from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from predict import build_model_from_checkpoint, normalize_checkpoint_config


EXPECTED_CHANNELS = [
    "nacpet",
    "z_coord",
    "topogram",
    "y_coord",
    "mri_in",
    "mri_out",
    "mri_absdiff",
]
EXPECTED_MODEL = "gated_topogram_conditioned_mri_resunet3d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: validate_checkpoint.py CHECKPOINT")

    path = Path(sys.argv[1])
    if not path.is_file() or path.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"Checkpoint is missing or unexpectedly small: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("Expected a training checkpoint containing model_state_dict")
    if not isinstance(checkpoint.get("config"), dict):
        raise RuntimeError("Checkpoint has no embedded config")

    cfg = normalize_checkpoint_config(checkpoint["config"])
    channels = cfg.get("input", {}).get("channels")
    model_cfg = cfg.get("model", {})
    failures = []

    if channels != EXPECTED_CHANNELS:
        failures.append(f"input channels are {channels}, expected {EXPECTED_CHANNELS}")
    if model_cfg.get("name") != EXPECTED_MODEL:
        failures.append(f"model is {model_cfg.get('name')}, expected {EXPECTED_MODEL}")
    if model_cfg.get("in_channels") != len(EXPECTED_CHANNELS):
        failures.append(f"in_channels is {model_cfg.get('in_channels')}, expected 7")
    if model_cfg.get("output_units") != "hu":
        failures.append(f"output_units is {model_cfg.get('output_units')}, expected hu")
    if not model_cfg.get("global_context", {}).get("enabled", False):
        failures.append("whole-body global_context is not enabled")
    if failures:
        raise RuntimeError("Wrong release checkpoint: " + "; ".join(failures))

    model = build_model_from_checkpoint(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    del model, checkpoint

    print(f"Checkpoint OK: {path}")
    print(f"SHA256: {sha256(path)}")
    print(f"Model: {EXPECTED_MODEL}")
    print(f"Input channels: {EXPECTED_CHANNELS}")
    print(f"Parameters: {parameters:,}")


if __name__ == "__main__":
    main()
