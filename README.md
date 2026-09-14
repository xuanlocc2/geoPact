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
| [`docker_final_v4_5/app/`](docker_final_v4_5/app/) | Portable inference, training, model, and PET-supervision source modules. |
| [`challenge-codebase/`](challenge-codebase/) | Official BIC-MAC challenge codebase, tracked as a Git submodule. |
| [`DATA_ACCESS.md`](DATA_ACCESS.md) | How to obtain official data and resources without committing private paths or artifacts. |

## Data and external resources

This repository does not include challenge data, CT labels, reconstructed PET labels, trained weights, or private infrastructure documentation. Obtain the official data, evaluator, and reconstruction resources directly from the BIC-MAC Challenge organizers under their data-use agreement. See [`DATA_ACCESS.md`](DATA_ACCESS.md).

## Reproducibility

1. Clone the repository with submodules:

   ```bash
   git clone --recurse-submodules https://github.com/xuanlocc2/geoPact.git
   ```

2. Obtain the official BIC-MAC data and challenge resources.
3. Create a local Python environment from the relevant experiment or Docker requirements.
4. Configure data locations locally through command-line arguments, environment variables, or an untracked configuration file.
5. Start with the V4.5 method source and the portable Docker application code.

## What is intentionally excluded

The repository ignores trained checkpoints, medical-image data, prediction outputs, virtual environments, temporary files, server-specific paths, credentials, and private runbooks. This keeps the public repository lightweight and avoids distributing restricted or machine-specific artifacts.

## Team

**DreamLab**
