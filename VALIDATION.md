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
