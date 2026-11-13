"""Export weights and portable inference settings without private training state."""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("Source and destination must differ; keep the training checkpoint intact.")
    if args.destination.exists():
        parser.error("Destination already exists; choose a new output path.")

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    # Only these sections are read by the released inference path. Training
    # state contains machine-specific paths and is not needed for prediction.
    portable_config = {
        key: copy.deepcopy(config[key])
        for key in ("input", "model", "metadata", "preprocess", "inference")
        if key in config
    }
    portable_config["training"] = {"patch_size": config["training"]["patch_size"]}
    portable_config["loss"] = {"spacing_mm": config.get("loss", {}).get("spacing_mm", [1.5234, 1.5234, 2.0])}
    exported = {
        "epoch": checkpoint["epoch"],
        "model_state_dict": checkpoint["model_state_dict"],
        "config": portable_config,
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, args.destination)
    with args.destination.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    print(f"Exported inference checkpoint: {args.destination}")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
