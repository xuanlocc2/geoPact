from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from dataset import get_subject_features, load_metadata
from transforms import DEFAULT_INPUT_CHANNELS, get_inference_transforms
from unet import build_model


DEFAULT_METADATA_KEYS = ["sex", "age", "height", "weight", "bmi"]
MODEL_PATH = Path(__file__).parent / "outputs/checkpoints/best_model.pth"
DEFAULT_PATCH_SIZE = (96, 96, 96)
SW_BATCH = 1
OVERLAP = 0.5


def load_checkpoint(checkpoint_path: Path, device):
    # Load on CPU first so the state dictionary does not duplicate model VRAM.
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt.get("config")
    return ckpt, None


def normalize_checkpoint_config(cfg):
    if cfg is None:
        return {
            "input": {"channels": DEFAULT_INPUT_CHANNELS},
            "model": {"in_channels": 3, "out_channels": 1, "base_channels": 32, "dropout": 0.2, "output_units": "normalized"},
            "training": {"patch_size": list(DEFAULT_PATCH_SIZE)},
        }
    cfg.setdefault("input", {"channels": DEFAULT_INPUT_CHANNELS})
    cfg.setdefault("training", {"patch_size": list(DEFAULT_PATCH_SIZE)})
    return cfg


def build_model_from_checkpoint(cfg):
    model_cfg = cfg.get("model", {})
    return build_model(
        in_channels=model_cfg.get("in_channels", len(cfg.get("input", {}).get("channels", DEFAULT_INPUT_CHANNELS))),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("base_channels", 32),
        dropout=model_cfg.get("dropout", 0.2),
        metadata_dim=len(cfg.get("metadata", {}).get("keys", DEFAULT_METADATA_KEYS)),
        use_metadata_film=cfg.get("metadata", {}).get("use_film", False),
        model_name=model_cfg.get("name", "mdixon_resunet3d"),
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


def get_metadata_tensor(subject: dict, cfg, device):
    metadata_cfg = cfg.get("metadata", {})
    if not metadata_cfg.get("use_film", False):
        return None
    values = load_metadata(subject.get("metadata_json"))
    keys = metadata_cfg.get("keys", DEFAULT_METADATA_KEYS)
    vector = [float(values.get(key, 0.0)) for key in keys]
    return torch.tensor(vector, dtype=torch.float32, device=device).view(1, -1)


def save_prediction_nifti(pred_hu: np.ndarray, reference_path: Path, output_path: Path) -> None:
    """Save CT in the NAC-PET physical grid and fail on any metadata drift."""
    reference = nib.load(reference_path)
    array = np.asarray(pred_hu, dtype=np.float32)
    if array.shape != reference.shape:
        raise RuntimeError(f"Prediction/reference shape mismatch: {array.shape} vs {reference.shape}")
    if not np.isfinite(array).all():
        raise RuntimeError("Prediction contains NaN or Inf values")
    if float(array.min()) < -1000.001 or float(array.max()) > 2000.001:
        raise RuntimeError(f"Prediction HU outside submission range: [{array.min()}, {array.max()}]")

    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    header["cal_min"] = float(array.min())
    header["cal_max"] = float(array.max())
    image = nib.Nifti1Image(array, reference.affine, header)

    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    if qform is not None:
        image.set_qform(qform, int(qcode))
    if sform is not None:
        image.set_sform(sform, int(scode))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, output_path)

    saved = nib.load(output_path)
    saved_array = saved.get_fdata(dtype=np.float32)
    checks = {
        "shape": saved.shape == reference.shape,
        "dtype": saved.get_data_dtype() == np.dtype(np.float32),
        "finite": bool(np.isfinite(saved_array).all()),
        "affine": bool(np.allclose(saved.affine, reference.affine, atol=1e-5, rtol=1e-6)),
        "spacing": bool(
            np.allclose(saved.header.get_zooms()[:3], reference.header.get_zooms()[:3], atol=1e-6, rtol=1e-6)
        ),
        "orientation": nib.aff2axcodes(saved.affine) == nib.aff2axcodes(reference.affine),
        "hu_range": float(saved_array.min()) >= -1000.001 and float(saved_array.max()) <= 2000.001,
    }
    saved_qform, saved_qcode = saved.get_qform(coded=True)
    saved_sform, saved_scode = saved.get_sform(coded=True)
    checks["qform"] = int(saved_qcode) == int(qcode) and (
        int(qcode) == 0 or bool(np.allclose(saved_qform, qform, atol=1e-5, rtol=1e-6))
    )
    checks["sform"] = int(saved_scode) == int(scode) and (
        int(scode) == 0 or bool(np.allclose(saved_sform, sform, atol=1e-5, rtol=1e-6))
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Saved NIfTI failed QC checks: {', '.join(failed)}")

    print(
        "NIfTI QC: "
        f"shape={saved.shape}, dtype={saved.get_data_dtype()}, "
        f"spacing={saved.header.get_zooms()[:3]}, orientation={nib.aff2axcodes(saved.affine)}, "
        f"qform={int(saved_qcode)}, sform={int(saved_scode)}"
    )


def predict(
    features_dir: str,
    output_ct: str,
    checkpoint: str,
    patch_size_override: tuple[int, int, int] | None = None,
    overlap_override: float | None = None,
):
    features_path = Path(features_dir)
    for filename in (
        "nacpet.nii.gz",
        "mri_combined_in_phase.nii.gz",
        "mri_combined_out_phase.nii.gz",
    ):
        required_path = features_path / filename
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, cfg = load_checkpoint(Path(checkpoint), device)
    cfg = normalize_checkpoint_config(cfg)
    model = build_model_from_checkpoint(cfg)
    model.load_state_dict(state_dict)
    del state_dict
    model.to(device)
    model.eval()

    print("Loading features...")
    subject = get_subject_features(str(features_path))
    data = get_inference_transforms(cfg)(subject)
    # Keep the complete seven-channel subject on CPU. MONAI transfers only the
    # active windows to sw_device, saving several GiB of inference VRAM.
    x = data["input"].unsqueeze(0).float().cpu()
    global_input = data.get("global_input")
    if global_input is not None:
        global_input = global_input.unsqueeze(0).to(device)
    metadata = get_metadata_tensor(subject, cfg, device)
    expected_channels = cfg["model"].get("in_channels", len(cfg["input"]["channels"]))
    assert x.ndim == 5, x.shape
    assert x.shape[1] == expected_channels, (x.shape, expected_channels)
    if not bool(torch.isfinite(x).all()):
        raise RuntimeError("Preprocessed inference input contains NaN or Inf")
    print(f"Input shape: {list(x.shape)}")

    patch_size = patch_size_override or tuple(
        cfg.get("inference", {}).get("patch_size", cfg.get("training", {}).get("patch_size", DEFAULT_PATCH_SIZE))
    )
    overlap = overlap_override if overlap_override is not None else cfg.get("inference", {}).get("overlap", OVERLAP)
    print(f"Sliding window inference, roi_size={patch_size}...")

    global_context = None
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        if getattr(model, "global_context_enabled", False):
            global_context = model.prepare_global_context(global_input)

    def predictor(window):
        window = window.to(device, non_blocking=True)
        meta = metadata
        if meta is not None and meta.shape[0] != window.shape[0]:
            meta = meta.expand(window.shape[0], -1)
        return model(window, metadata=meta, global_context=global_context)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        pred = sliding_window_inference(
            x,
            roi_size=patch_size,
            sw_batch_size=cfg.get("inference", {}).get("sw_batch_size", SW_BATCH),
            predictor=predictor,
            overlap=overlap,
            mode="gaussian",
            progress=False,
            sw_device=device,
            device=torch.device("cpu"),
        )

    output_units = cfg.get("model", {}).get("output_units", "hu")
    if output_units == "normalized":
        pred_hu = pred.cpu().float().clamp(0.0, 1.0).numpy()[0, 0] * 3000.0 - 1000.0
    else:
        pred_hu = pred.cpu().float().numpy()[0, 0]
    print("Prediction HU range before clipping:", float(pred_hu.min()), float(pred_hu.max()))
    if not np.isfinite(pred_hu).all():
        raise RuntimeError("Model prediction contains NaN or Inf before clipping")
    pred_hu = np.clip(pred_hu, -1000.0, 2000.0).astype(np.float32)

    out_path = Path(output_ct)
    save_prediction_nifti(pred_hu, features_path / "nacpet.nii.gz", out_path)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--output_ct", required=True)
    parser.add_argument("--checkpoint", default=str(MODEL_PATH))
    parser.add_argument(
        "--patch_size",
        nargs=3,
        type=int,
        metavar=("X", "Y", "Z"),
        help="Override checkpoint sliding-window ROI without changing the checkpoint config.",
    )
    parser.add_argument("--overlap", type=float, default=None, help="Optional sliding-window overlap override.")
    args = parser.parse_args()

    if args.patch_size is not None and any(value <= 0 for value in args.patch_size):
        parser.error("--patch_size values must be positive")
    if args.overlap is not None and not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must be in [0, 1)")

    predict(
        args.features_dir,
        args.output_ct,
        args.checkpoint,
        patch_size_override=tuple(args.patch_size) if args.patch_size is not None else None,
        overlap_override=args.overlap,
    )


if __name__ == "__main__":
    main()
