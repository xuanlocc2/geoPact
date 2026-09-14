from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--prediction", required=True)
    args = parser.parse_args()

    reference_path = Path(args.reference)
    prediction_path = Path(args.prediction)
    reference = nib.load(reference_path)
    prediction = nib.load(prediction_path)
    array = prediction.get_fdata(dtype=np.float32)

    checks = {
        "shape": prediction.shape == reference.shape,
        "dtype": prediction.get_data_dtype() == np.dtype(np.float32),
        "finite": bool(np.isfinite(array).all()),
        "affine": bool(np.allclose(prediction.affine, reference.affine, atol=1e-5, rtol=1e-6)),
        "spacing": bool(
            np.allclose(
                prediction.header.get_zooms()[:3],
                reference.header.get_zooms()[:3],
                atol=1e-6,
                rtol=1e-6,
            )
        ),
        "orientation": nib.aff2axcodes(prediction.affine) == nib.aff2axcodes(reference.affine),
        "hu_range": float(array.min()) >= -1000.001 and float(array.max()) <= 2000.001,
    }

    reference_qform, reference_qcode = reference.get_qform(coded=True)
    prediction_qform, prediction_qcode = prediction.get_qform(coded=True)
    reference_sform, reference_scode = reference.get_sform(coded=True)
    prediction_sform, prediction_scode = prediction.get_sform(coded=True)
    checks["qform"] = int(prediction_qcode) == int(reference_qcode) and (
        int(reference_qcode) == 0
        or bool(np.allclose(prediction_qform, reference_qform, atol=1e-5, rtol=1e-6))
    )
    checks["sform"] = int(prediction_scode) == int(reference_scode) and (
        int(reference_scode) == 0
        or bool(np.allclose(prediction_sform, reference_sform, atol=1e-5, rtol=1e-6))
    )

    for name, passed in checks.items():
        print(f"{name}: {'OK' if passed else 'FAILED'}")
    print(f"shape: {prediction.shape}")
    print(f"spacing: {prediction.header.get_zooms()[:3]}")
    print(f"orientation: {nib.aff2axcodes(prediction.affine)}")
    print(f"HU min/max/mean: {float(array.min()):.3f} / {float(array.max()):.3f} / {float(array.mean()):.3f}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Output QC failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
