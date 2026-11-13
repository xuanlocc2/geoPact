# Data and Resource Access

This repository contains source code and method documentation. It does not distribute challenge data, CT labels, reconstructed PET labels, or private infrastructure notes.

## Public model weights

Download `model.pth` from the [V4.5 epoch-150 release](https://github.com/xuanlocc2/geoPact/releases/tag/v4.5-e150). Place it at `docker_final_v4_5/checkpoints/model.pth` and verify it against [`model.sha256`](docker_final_v4_5/model.sha256). The checkpoint contains weights and portable inference settings, with training state and private paths removed. The released weights are available under the project MIT license; no medical images are bundled.

If the download is unavailable, [open an issue](https://github.com/xuanlocc2/geoPact/issues) to contact the maintainers. Challenge organizers supply the data and official resources, not this team's checkpoint.

## Challenge data and evaluation

The public [training pipeline](training/README.md) additionally needs official
CT targets/masks and complete PET supervision labels for the configured eight
PET-labelled subjects. Its original-data preflight checks 75 complete subjects.
No training labels, patient metadata, or supervision caches are published.

To reproduce an experiment, obtain the following resources from the BIC-MAC Challenge organizers and use them under the competition's data-use agreement:

1. The official training and validation data supplied by the challenge.
2. The official challenge codebase and evaluator.
3. Any official reconstruction environment or baseline assets required by the challenge instructions.

Configure local paths through command-line arguments, environment variables, or an untracked local configuration file. Do not commit local paths, credentials, server-specific runbooks, or downloaded data and model artifacts.
