# Public inference release validation

Release: `v4.5-e150`. Checked on 2026-11-13.

## Checks performed for this release

- Python syntax parsing passed for all 11 public application modules.
- The public inference export contains only epoch, model state, and portable
  inference configuration. Optimizer/scheduler state and private training paths
  were omitted; the original training file was kept intact.
- All 236 state-dictionary tensors exactly match the original epoch-150 file.
- Input, model, metadata, preprocessing, inference, patch size, and spacing
  settings used by inference match the original checkpoint.
- `validate_checkpoint.py` passed strict state loading: seven input channels,
  HU output, enabled global context, and 20,579,395 parameters.
- The public export's SHA-256 matches `docker_final_v4_5/model.sha256`.
- `validate_output.py` passed all checks on an existing 128-cubed cropped
  prediction: shape, float32 dtype, finite values, affine, spacing, orientation,
  HU range, qform, and sform. This was validation of a saved output, not a new
  inference run.

## Environment and limits

These release checks used Windows, Python 3.13, CPU PyTorch 2.7.1, and MONAI
1.6.0. This differs from the pinned Docker deployment environment. CUDA was
unavailable and Docker Engine was not running, so a fresh Docker build and
GPU inference were not performed during this release preparation.

The original local build record reports a successful cropped Docker smoke test
and a full-subject host-memory failure in a roughly 8 GB Docker VM. These checks
do not establish full-subject A40 runtime, memory qualification, prediction
accuracy on external cohorts, or clinical generalization. The challenge target
remains one A40, 128 GB host RAM, and 300 seconds per subject.

## Training source publication (2026-11-13)

The archived V4.5 training pipeline is now provided under `training/`.

- All 13 training/test Python files passed syntax parsing.
- Training, preflight, and full-volume validation CLI imports and `--help`
  completed successfully.
- Eight synthetic tests passed: sampling/split handling, missing PET-label
  detection, controlled ray weighting, soft PET regions, resampling/cache on a
  flipped physical grid, and finite nonzero loss backpropagation.
- The training model loaded the public epoch-150 state dictionary strictly
  (20,579,395 parameters).
- Config environment expansion and missing-variable errors were checked;
  remote uploads are disabled in the published config.
- The configured PET loss weights were checked at epochs 1, 31, 91, 117, 118,
  and 150, preserving the archived epoch-118 tuning phase.
- Architecture, preprocessing, augmentation, optimization, and inference
  settings were compared with the original checkpoint. The known differences
  in path/upload settings and recorded loss history are explained in the
  training guide.

These checks used the same local CPU environment described above. The first
test attempt hit a Windows temporary-directory permission error (seven tests
passed); rerunning with a fresh temporary directory inside the workspace passed
all eight. No complete-dataset preflight, new full training run, or new GPU
training/inference was performed. Tests do not establish training accuracy,
resource requirements, or bit-identical reproduction.
