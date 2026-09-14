from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise RuntimeError("Checkpoint does not contain an embedded config")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(checkpoint["config"], handle, sort_keys=False, allow_unicode=True)
    print(f"Exported checkpoint config: {args.output}")


if __name__ == "__main__":
    main()
