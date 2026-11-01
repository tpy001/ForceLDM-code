# pi0_forceldm validation notes

The [main README](README.md) is the user guide for installation, data processing, robot adaptation, training, and serving. This document summarizes the recorded preparation and GPU smoke tests. It does not report a complete training reproduction or robot task success rate.

## Recorded environment

| Component | Recorded version |
| --- | --- |
| OS | Ubuntu 22.04, Linux x86_64 |
| GPU | One NVIDIA A800-SXM4-80GB |
| Python / uv | 3.12.13 / 0.13.6 |
| JAX / jaxlib / Flax | 0.5.3 / 0.5.3 / 0.10.2 |
| PyTorch | 2.7.1+cu126 |
| TorchCodec / Transformers / Diffusers | 0.4.0 / 4.53.2 / 0.37.0 |
| NumPy / huggingface-hub / tyro | 1.26.4 / 0.36.2 / 1.0.10 |
| LeRobot | Source revision `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |
| System FFmpeg | 4.4.2 |

Use the checked-in `uv.lock`. The driver's reported CUDA capability is distinct from the CUDA version of installed Python wheels.

## Pinned data and assets

- Dataset: `PeiYuan001/forceldm-dataset`, revision `ee424c789ee8e43c95d34d08ba88c9f6b39870f6`.
- Dataset metadata: LeRobot v2.1, 474 episodes, 362,339 frames, 30 Hz, six tasks. The downloaded snapshot contained 2,394 files totaling 9,560,254,738 bytes.
- Base model: official JAX `gs://openpi-assets/checkpoints/pi0_base`, downloaded to `checkpoints/pi0_base`. The recorded download check covered 33 files totaling 12,014,440,489 bytes, including size and MD5 comparisons.
- VAE: `stabilityai/sdxl-vae`, revision `6f5909a7e596173e25d4e97b07fd19cdf9611c76`.
- Tokenizer: `gs://big_vision/paligemma_tokenizer.model`, cached using `OPENPI_DATA_HOME`.

The prepared snapshot already includes calibrated force and two flow video streams. Its original state has 27 values; the robot adapter selects the first seven. The current wrench is consistent with state values `[7:13]`. Flow targets are RGB visualizations, not physical displacement tensors.

## Data and normalization checks

Recorded checks covered a real episode, action chunks `(32, 7)`, effort windows `(42, 6)`, two current RGB views, and two flow views. Robot/model transforms produced padded state `(32,)`, actions `(32, 32)`, effort `(42, 6)`, 48 prompt tokens, and 224 x 224 RGB images.

Full normalization used the original state/action/effort transforms and batch ordering. With batch size 16 and the original loader's drop-last behavior, 362,336 frames were used and three trailing frames were dropped. Initial eight-frame smoke statistics were separate temporary assets and must not be used for training.

`scripts/compute_norm_stats_no_video.py` skips unused video decoding and checks numeric-reader equivalence at selected episode boundaries/interior frames. It is specialized to the default data layout, local cache, 30 Hz, and 32-step horizon. For another robot, prefer `scripts/compute_norm_stats.py --config-name <config>` unless you adapt those assumptions.

Normalization paths:

```text
assets/pi0_forceldm/PeiYuan001/forceldm-dataset/norm_stats.json
assets/pi0_forceldm_noise/PeiYuan001/forceldm-dataset/norm_stats.json
```

The noise variant can reuse the baseline statistics only while both data pipelines are identical.

## Single-GPU smoke tests: November 1, 2026

Both configurations used two current RGB views, batch size 16, FSDP size 1, EMA disabled, the pinned dataset and weights, full normalization, and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`.

| Configuration | Optimization steps | Sampled GPU peak | JAX active allocation peak |
| --- | ---: | ---: | ---: |
| `pi0_forceldm` | 3 | 71.97 GiB | 55.16 GiB |
| `pi0_forceldm_noise` | 3 | 71.94 GiB | 55.21 GiB |

Each test completed forward/backward passes and optimizer updates with finite logged metrics and no out-of-memory error. The smoke schedule compressed training to three steps so the noise variant exercised strengths 0, 0.15, and 0.3. Normal configs retain 30,000 steps, with noise ramping between 30% and 70% progress.

Sampled GPU peaks include JAX preallocation and are not directly comparable to active allocation peaks. A100 80GB was not directly tested. These measurements do not cover three RGB cameras, enabled EMA, multi-GPU scaling, inference latency, or robot evaluation.

## Historical checkpoints and resume

An earlier training run was stopped at the user's request. A recorded saved checkpoint was at step 15,000 under the old config directory:

```text
checkpoints/pi0_latent_flow_multiview/forceldm_multiview_bs16_seed42_offline_20261031/15000
```

That checkpoint is an external local artifact, not included in this repository. Logged steps after a save do not imply a newer resumable checkpoint exists. The model/config names were subsequently unified to `pi0_forceldm`; new runs use `checkpoints/pi0_forceldm/<exp_name>/`.

Serving can use the current compatible config with an explicitly supplied historical checkpoint path. `--resume` resolves the current config/experiment directory; it does not discover an old run from its renamed model. Verify compatibility and map or migrate the complete training state before resuming an old run. No migration or resumed training is implied by these notes.

The recorded runs used local W&B offline logging. Model weights, datasets, credentials, private logs, and offline run files are excluded from Git. No independent evaluation split or robot success-rate result was established by these checks.
