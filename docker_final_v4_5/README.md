# GeoPACT V4.5 inference

Public inference implementation for Team DreamLab. This generates pseudo-CT;
PET reconstruction is a separate step. Weights are the epoch-150 model exported
without optimizer state or private training paths.

## Requirements

- Recommended: Linux x86-64, NVIDIA GPU, Docker, NVIDIA Container Toolkit.
  The Docker entrypoint requires CUDA.
- Docker pins PyTorch 2.11.0, CUDA 12.8, and cuDNN 9. The host driver must
  support that container runtime. Other packages are pinned in requirements.txt.
- Challenge target: one NVIDIA A40, 128 GB host RAM, 300 seconds per subject.
  This is a target, not a guarantee on other hardware. Full volumes need much
  more RAM than small crops; the original local full-subject test exceeded
  a Docker Desktop VM with approximately 8 GB RAM.
- Build and checkpoint download require internet; inference runs offline.

## 1. Download weights

Run from the repository root using Bash:

```bash
mkdir -p docker_final_v4_5/checkpoints
curl --fail --location --retry 3 \
  https://github.com/xuanlocc2/geoPact/releases/download/v4.5-e150/model.pth \
  --output docker_final_v4_5/checkpoints/model.pth
(cd docker_final_v4_5 && sha256sum --check model.sha256)
```

Public inference export SHA-256:
`1384551db67b1d34d51ea7c08b923712e276e2dd17a8f9ffeae800c71a7b8027`.
This differs from the original training file's hash; model tensors are unchanged.
The exporter is `app/export_inference_checkpoint.py`. Keep original training
checkpoints separately. Only load checkpoints from trusted sources.

## 2. Prepare inputs

Use one subject's official BIC-MAC feature directory:

```text
/path/to/sub-004/features/
  nacpet.nii.gz
  mri_combined_in_phase.nii.gz
  mri_combined_out_phase.nii.gz
  topogram.nii.gz
```

All four files are required for this model. No CT, segmentation, or reconstructed
PET labels are needed. Optional metadata.json is not used for conditioning by
this checkpoint (metadata.use_film=false).

Preserve official intensity conventions, orientation, spacing, and registration.
The topogram is stored as NIfTI with its singleton dimension as in challenge data.
Arbitrary clinical DICOM or unregistered cohorts are not drop-in inputs; review
app/transforms.py before adapting another dataset.

## 3. Build and run Docker

Run from the repository root:

```bash
docker build -t geopact:v4.5-e150 docker_final_v4_5
FEATURES_DIR=/absolute/path/to/sub-004/features
OUTPUT_DIR=/absolute/path/to/output/sub-004
mkdir -p "$OUTPUT_DIR"
docker run --rm --gpus all --network none --memory 128g \
  --mount "type=bind,source=$FEATURES_DIR,target=/data/features,readonly" \
  --mount "type=bind,source=$OUTPUT_DIR,target=/data/output" \
  geopact:v4.5-e150
```

For challenge timing checks on Linux, prefix docker run with `timeout 300`.
Ensure the Docker host/VM actually has the requested memory available.
Build-time validation checks the seven-channel architecture, embedded config,
HU output, global context, and strict weight loading.

The container embeds the checkpoint and uses ROI 160 x 160 x 192, sliding-window
batch 1, overlap 0.5. It writes `$OUTPUT_DIR/ct.nii.gz`, replacing any existing
file at that exact location. Use a separate output directory per subject.

Output is float32 HU clipped to [-1000, 2000], on the NAC-PET grid. Shape,
affine, spacing, orientation, qform, and sform are preserved and checked after
saving. Logs go to stderr; failures return nonzero exit status.

## 4. Native Python alternative

For the pinned environment, use Python 3.11 and PyTorch 2.11.0 for your hardware.
Example for CUDA 12.8, in Bash from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r docker_final_v4_5/requirements.txt
python docker_final_v4_5/app/validate_checkpoint.py docker_final_v4_5/checkpoints/model.pth
python docker_final_v4_5/app/predict.py \
  --features_dir /absolute/path/to/sub-004/features \
  --output_ct /absolute/path/to/output/sub-004/ct.nii.gz \
  --checkpoint docker_final_v4_5/checkpoints/model.pth \
  --patch_size 160 160 192 --overlap 0.5
```

On Windows activate with `.venv\Scripts\Activate.ps1`; use single-line commands
or PowerShell continuation syntax. Native predict.py falls back to CPU without
CUDA, but CPU execution is not expected to meet challenge timing.

## 5. Check output

In the native Python environment:

```bash
python docker_final_v4_5/app/validate_output.py \
  --reference /absolute/path/to/sub-004/features/nacpet.nii.gz \
  --prediction /absolute/path/to/output/sub-004/ct.nii.gz
```

Or use the built image without a host Python environment:

```bash
docker run --rm --network none --entrypoint python \
  --mount "type=bind,source=$FEATURES_DIR,target=/data/features,readonly" \
  --mount "type=bind,source=$OUTPUT_DIR,target=/data/output,readonly" \
  geopact:v4.5-e150 /app/validate_output.py \
  --reference /data/features/nacpet.nii.gz \
  --prediction /data/output/ct.nii.gz
```

These checks validate geometry, dtype, finite values, and HU range, not accuracy.
PET reconstruction and challenge scoring require the organizers' separate tools
and authorized data access.

## Validation limits and troubleshooting

The original local record reports a successful cropped Docker smoke test.
A full-subject attempt exceeded local Docker VM RAM. That record does not
establish full-subject A40 qualification or clinical generalization.

- Missing checkpoint: download to checkpoints/model.pth and verify the hash.
- Missing input: check all four filenames and the mounted directory.
- CUDA unavailable: check driver, Container Toolkit, and --gpus all.
- Exit 137: inspect host/container RAM limits; this can indicate OOM.
- CUDA OOM: use sufficient VRAM. Changing ROI may change predictions and no
  longer matches the fixed Docker recipe.

A complete training entrypoint is not included. See the root README for method,
data access, license, and support.
