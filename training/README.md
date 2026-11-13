# GeoPACT V4.5 training

This is the archived V4.5 training implementation for the released epoch-150
model: training loop, losses, PET evaluator surrogate, preprocessing, model,
checkpointing, and full-volume monitoring. The original source snapshot is
`v4.5-train75-exact-pet-controlled-ray-dgx`, recipe revision
`exact-pet-supervision-19pct-controlled-ray-dgx-20260806`.

The source is kept together here so local imports resolve to the matching
training snapshot. Use `docker_final_v4_5/` for released inference: its predictor
contains deployment-specific memory and output checks added after the training
snapshot. This directory is not part of the inference Docker build.

## Environment

Use Python 3.11 with a hardware-appropriate PyTorch installation. For a CUDA
12.8 host, an example environment matching the inference dependency pins is:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r training/requirements.txt
```

Run commands from the repository root. The archived recipe used DGX Spark
ARM64; select a native PyTorch wheel appropriate for that machine if using it.
The environment above is a portable setup, not a claim that the full training
run has been repeated on x86-64. Training memory use differs from inference;
the inference A40 resource target is not a training hardware guarantee.

## Data layout

Obtain authorized challenge data and PET supervision resources from the
organizers; see [data access](../DATA_ACCESS.md). Do not upload the data to GitHub.

```text
/absolute/path/to/train/
  sub-000/
    features/
      nacpet.nii.gz
      mri_combined_in_phase.nii.gz
      mri_combined_out_phase.nii.gz
      topogram.nii.gz
    ct-label/
      ct.nii.gz
      prediction_mask.nii.gz
      body_seg.nii.gz
      organ_seg.nii.gz
    pet-label/                  # complete set on PET-supervised subjects only
      pet.nii.gz
      body_seg.nii.gz
      organ_seg.nii.gz
      tissue_seg.nii.gz
  sub-001/
    ...
```

Retain official registration, intensity conventions, label values, and NIfTI
geometry. `ct.nii.gz` and `prediction_mask.nii.gz` are required by the loader;
retain the official body/organ masks for the original objective and monitoring.
The PET label set must be complete when present. It is used only for
supervision, never as model input. Optional feature `metadata.json` is not used
for model conditioning in this recipe.

The original run uses 75 unique labelled subjects. PET supervision is expected
for `sub-000`, `sub-001`, `sub-002`, `sub-005`, `sub-006`, `sub-008`, `sub-013`,
and `sub-014`. Each is replicated once in sampling: 83 records and 16 PET
appearances per epoch, with eight crops per record.

## Configure, check, and train

Set three absolute local paths in Bash:

```bash
export GEOPACT_DATA_DIR=/absolute/path/to/train
export GEOPACT_OUTPUT_DIR=/absolute/path/to/outputs/geopact-v45
export GEOPACT_CACHE_DIR=/absolute/path/to/cache/geopact-v45
export MPLBACKEND=Agg
mkdir -p "$GEOPACT_OUTPUT_DIR" "$GEOPACT_CACHE_DIR"
python training/preflight_v45.py --config training/config_v4_5.yaml
python training/preflight_v45.py --config training/config_v4_5.yaml --build-pet-cache
python training/train.py --config training/config_v4_5.yaml
```

On PowerShell, set variables with `$env:GEOPACT_DATA_DIR = 'E:/path/to/train'`
and similarly for output/cache; activate with `.venv\Scripts\Activate.ps1`.
Alternatively, copy the YAML to `training/config.local.yaml` (ignored by Git) and replace the three
`${GEOPACT_...}` values with absolute paths. Paths are relative to the working
directory if not absolute. Use a fresh output directory for a new run; the
training script initializes log files when not resuming.

The preflight checks the original 75-case split, eight PET-labelled cases,
sampling, ROI, ray weights, and parameter count. `--build-pet-cache` additionally
loads PET volumes and builds the compact supervision cache. For another cohort
or ablation, use a separate config and output directory; edit subject lists,
PET requirements, and split settings deliberately. The original-data preflight
is intentionally specific to the released recipe.

The released config disables optional Hugging Face uploads and contains no
remote destination. Training does not require an account or runtime network.
Do not enable uploads for restricted data without appropriate authorization.

## Resume and outputs

```bash
python training/train.py --config training/config_v4_5.yaml --resume auto
# Or specify a full training checkpoint:
python training/train.py --config training/config_v4_5.yaml \
  --resume /absolute/path/to/outputs/geopact-v45/checkpoints/last_model.pth
```

`auto` uses the last checkpoint if present; otherwise it starts a new run.
Use trusted full training checkpoints from this pipeline. The public GitHub
Release checkpoint is inference-only and lacks optimizer/scheduler/loss state
and recipe metadata; it is not a training-resume checkpoint.

Outputs include `effective_config.yaml`, `checkpoints/`, `logs/`, `plots/`, and
periodic full-volume monitoring outputs. Checkpoints retain the embedded config,
epoch, model, optimizer, scheduler, loss-reference state, and AMP scaler.
Checkpoints, generated images, and caches must stay outside Git history.

For separate full-volume CT/proxy evaluation on authorized labelled data:

```bash
python training/full_volume_validate.py \
  --config training/config_v4_5.yaml \
  --checkpoint /absolute/path/to/checkpoints/last_model.pth \
  --val_dir /absolute/path/to/held-out-labelled-data \
  --output_dir /absolute/path/to/evaluation
```

This evaluates CT and a differentiable PET response surrogate. Actual PET
reconstruction and challenge scoring still require the organizers' tools.

## Recipe provenance and interpretation

The architecture, preprocessing, augmentation, optimization, ROI, and inference
settings were compared with the original epoch-150 checkpoint. The emergency
checkpoint interval is set to the checkpoint's value (0), rather than the
archived YAML's later value (25). Paths are now local environment variables,
and optional remote uploads are disabled. The training/preflight default config
paths now resolve beside their scripts, and unset config variables fail early
with a descriptive error.

The archived YAML records loss phases starting at epochs 1, 31, 91, and 118.
At epoch 91 the PET whole-body/organ/brain weights are 0.15/0.15/0.05;
from epoch 118 they are 0.20/0.20/0.015. This preserves the documented tuning
history. The final checkpoint's embedded config instead has the final values
assigned to its epoch-91 phase and does not fully record that history.

Both `sub-000` and `sub-001` are included in training and reused as monitoring
cases (`include_val_subjects_in_train: true`). Their metrics are training
diagnostics, not held-out generalization estimates. The epoch-118 adjustment
was guided by public validation, so that set also informed model development.
For independent generalization estimates use a separate held-out cohort.

A fresh 150-epoch GPU run has not been repeated for this publication. The
archived code/config provide the implementation and recorded recipe, not a
promise of bit-identical weights across hardware or restarts. See
[validation notes](../VALIDATION.md) for checks performed.

## Tests and license

```bash
python -m pip install -r training/requirements-dev.txt
python -m pytest -c training/pytest.ini training/tests -q
```

Tests use synthetic arrays and need no challenge data. They cover PET sampling,
exact PET masks/cache, controlled ray weighting, and loss backpropagation.
Project code is covered by the root [MIT License](../LICENSE). Third-party
resources and challenge data retain their own terms.
