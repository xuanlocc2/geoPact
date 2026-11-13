# GeoPACT: Geometry-Anchored PET-Aware Pseudo-CT

GeoPACT is Team DreamLab's solution for the BIC-MAC Challenge. It generates whole-body pseudo-CT from NAC-PET, Dixon MRI, and a two-dimensional topogram for PET attenuation correction.

The method achieved a tied first average rank on the challenge private leaderboard. Its design emphasizes downstream PET quality as well as attenuation-map accuracy.

## Method at a glance

- **Geometry anchor:** NAC-PET and absolute z-coordinate form the primary 3D geometry stream.
- **Controlled multimodal fusion:** topogram and Dixon MRI use separate encoders and gated residual corrections instead of early concatenation.
- **Coarse global context:** a compact whole-body encoder modulates the local U-Net at its coarse scales through FiLM.
- **PET-aware supervision:** local attenuation errors are composed in a compact full-field-of-view grid, then evaluated by a differentiable attenuation-response surrogate.
- **Evaluator-shaped losses:** training combines raw and smoothed attenuation losses with whole-body, organ-bias, and brain-outlier PET-response objectives.

## Repository guide

| Location | Contents |
|---|---|
| [`experiments/BICMAC_V4_5_METHOD_TECHNICAL_PAPER_SOURCE.md`](experiments/BICMAC_V4_5_METHOD_TECHNICAL_PAPER_SOURCE.md) | Complete V4.5 method specification: preprocessing, architecture, losses, optimization, and inference. |
| [`docker_final_v4_5/Dockerfile`](docker_final_v4_5/Dockerfile) | Portable Docker build definition. |
| [`docker_final_v4_5/README.md`](docker_final_v4_5/README.md) | Docker and native Python inference instructions. |
| [`docker_final_v4_5/app/`](docker_final_v4_5/app/) | Inference, model, preprocessing, and PET-supervision source modules. |
| [`challenge-codebase/`](challenge-codebase/) | Official BIC-MAC challenge codebase, tracked as a Git submodule. |
| [`DATA_ACCESS.md`](DATA_ACCESS.md) | How to obtain official data and resources without committing private paths or artifacts. |
| [`training/README.md`](training/README.md) | V4.5 training loop, config, losses, preflight, resume, and tests. |

## Data and external resources

Challenge data, CT labels, reconstructed PET labels, and private infrastructure documentation are not distributed here. Obtain the official data, evaluator, and reconstruction resources directly from the BIC-MAC Challenge organizers under their data-use agreement. See [`DATA_ACCESS.md`](DATA_ACCESS.md).

The **V4.5 epoch-150 inference checkpoint** is available through [GitHub Releases](https://github.com/xuanlocc2/geoPact/releases/tag/v4.5-e150). It contains trained weights and portable inference settings; optimizer state and private training paths are omitted.

## Quick start: inference

Use Bash on a Linux NVIDIA GPU host with Docker and NVIDIA Container Toolkit. See the [inference guide](docker_final_v4_5/README.md) for input details, native Python usage, and output validation.

```bash
git clone --recurse-submodules https://github.com/xuanlocc2/geoPact.git
cd geoPact
mkdir -p docker_final_v4_5/checkpoints
curl --fail --location --retry 3 \
  https://github.com/xuanlocc2/geoPact/releases/download/v4.5-e150/model.pth \
  --output docker_final_v4_5/checkpoints/model.pth
(cd docker_final_v4_5 && sha256sum --check model.sha256)
docker build -t geopact:v4.5-e150 docker_final_v4_5
```

Set absolute paths to one subject's features and a writable output directory:

```bash
FEATURES_DIR=/absolute/path/to/sub-004/features
OUTPUT_DIR=/absolute/path/to/output/sub-004
mkdir -p "$OUTPUT_DIR"
docker run --rm --gpus all --network none --memory 128g \
  --mount "type=bind,source=$FEATURES_DIR,target=/data/features,readonly" \
  --mount "type=bind,source=$OUTPUT_DIR,target=/data/output" \
  geopact:v4.5-e150
```

Required inputs: `nacpet.nii.gz`, `mri_combined_in_phase.nii.gz`, `mri_combined_out_phase.nii.gz`, and `topogram.nii.gz`, in official BIC-MAC format. Output: `ct.nii.gz`, float32 pseudo-CT in HU clipped to [-1000, 2000], on the NAC-PET grid. PET reconstruction is a separate step using the organizers' tools.

## Training and validation status

The repository includes the [V4.5 training pipeline](training/README.md), configuration, losses, preflight, checkpoint resume, and synthetic tests, alongside the [method specification](experiments/BICMAC_V4_5_METHOD_TECHNICAL_PAPER_SOURCE.md). Training requires authorized challenge data and PET supervision resources. See the training guide for setup and commands.

The final recipe trains on all 75 labelled cases and reuses two training cases for monitoring; those monitoring metrics are not held-out validation. The published config preserves the documented epoch-118 loss adjustment. A complete 150-epoch GPU run has not been repeated for this publication.

The challenge deployment target is one NVIDIA A40, 128 GB host RAM, and 300 seconds per subject. This is a target, not a runtime guarantee on other hardware. The original local record reports a successful cropped Docker smoke test; full-subject A40 qualification is not established by that record.

See [`VALIDATION.md`](VALIDATION.md) for checks performed on the public export and their limitations.

## What is intentionally excluded

Git history excludes checkpoint binaries, medical-image data, prediction outputs, virtual environments, temporary files, server-specific paths, credentials, and private runbooks. Download the public checkpoint from Releases instead of adding it to Git.

## License and support

Project code, documentation, and the released inference weights are available under [MIT](LICENSE). Third-party dependencies and the `challenge-codebase` submodule retain their own terms. This license does not grant access to challenge data or change its data-use agreement.

For setup or download problems, [open a GitHub issue](https://github.com/xuanlocc2/geoPact/issues). Include the release tag, environment, command, and error log with private paths and subject information removed.

## Team

**DreamLab**
