from __future__ import annotations

import faulthandler
import logging
from pathlib import Path
import sys
import time

import torch

from predict import predict


FEATURES_DIR = Path("/data/features")
OUTPUT_CT = Path("/data/output/ct.nii.gz")
CHECKPOINT = Path("/app/checkpoints/model.pth")
PATCH_SIZE = (160, 160, 192)
OVERLAP = 0.5

REQUIRED_FEATURES = (
    "nacpet.nii.gz",
    "mri_combined_in_phase.nii.gz",
    "mri_combined_out_phase.nii.gz",
    "topogram.nii.gz",
)

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)
LOGGER = logging.getLogger("bicmac")


def _log_cuda_memory(prefix: str) -> None:
    if not torch.cuda.is_available():
        return
    mib = 1024**2
    LOGGER.info(
        "%s cuda_allocated_mib=%.1f cuda_reserved_mib=%.1f "
        "cuda_peak_allocated_mib=%.1f cuda_peak_reserved_mib=%.1f",
        prefix,
        torch.cuda.memory_allocated() / mib,
        torch.cuda.memory_reserved() / mib,
        torch.cuda.max_memory_allocated() / mib,
        torch.cuda.max_memory_reserved() / mib,
    )


def main() -> int:
    faulthandler.enable(all_threads=True)
    started = time.perf_counter()
    stage = "startup"

    try:
        LOGGER.info("status=starting stage=%s", stage)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for final inference")
        torch.cuda.reset_peak_memory_stats()
        properties = torch.cuda.get_device_properties(0)
        LOGGER.info(
            "runtime python=%s torch=%s torch_cuda=%s gpu=%s total_vram_mib=%.1f",
            sys.version.split()[0],
            torch.__version__,
            torch.version.cuda,
            properties.name,
            properties.total_memory / 1024**2,
        )

        stage = "preflight"
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"Missing baked checkpoint: {CHECKPOINT}")
        missing = [name for name in REQUIRED_FEATURES if not (FEATURES_DIR / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing required feature files: {missing}")
        for name in REQUIRED_FEATURES:
            path = FEATURES_DIR / name
            LOGGER.info("input name=%s size_mib=%.1f", name, path.stat().st_size / 1024**2)

        OUTPUT_CT.parent.mkdir(parents=True, exist_ok=True)
        if OUTPUT_CT.exists():
            OUTPUT_CT.unlink()
        LOGGER.info(
            "contract features=%s output=%s roi=%s overlap=%.3f",
            FEATURES_DIR,
            OUTPUT_CT,
            PATCH_SIZE,
            OVERLAP,
        )

        stage = "inference"
        predict(
            features_dir=str(FEATURES_DIR),
            output_ct=str(OUTPUT_CT),
            checkpoint=str(CHECKPOINT),
            patch_size_override=PATCH_SIZE,
            overlap_override=OVERLAP,
        )

        stage = "output_validation"
        if not OUTPUT_CT.is_file() or OUTPUT_CT.stat().st_size == 0:
            raise RuntimeError(f"Inference did not create a valid output: {OUTPUT_CT}")

        _log_cuda_memory("status=success")
        LOGGER.info(
            "status=success stage=complete output_size_mib=%.1f elapsed_seconds=%.2f",
            OUTPUT_CT.stat().st_size / 1024**2,
            time.perf_counter() - started,
        )
        return 0
    except Exception:
        _log_cuda_memory("status=failed")
        LOGGER.exception(
            "status=failed stage=%s elapsed_seconds=%.2f",
            stage,
            time.perf_counter() - started,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
