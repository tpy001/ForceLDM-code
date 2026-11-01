<h1 align="center">ForceLDM: A Force-aware Latent Dynamics Model for Contact-Rich Manipulation</h1>

<p align="center">
  <a href="https://tpy001.github.io/ForceLDM/"><img src="https://img.shields.io/badge/Project-Page-3273DC?logo=googlechrome&amp;logoColor=white" alt="ForceLDM project page"></a>
  <a href="https://huggingface.co/datasets/PeiYuan001/forceldm-dataset"><img src="https://img.shields.io/badge/Data-Hugging%20Face-FFD21E?logo=huggingface&amp;logoColor=black" alt="Data on Hugging Face"></a>
  <a href="https://github.com/tpy001/forcewam-275ab3be/tree/forceldm"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github&amp;logoColor=white" alt="Code on GitHub"></a>
  <!-- Add the arXiv URL here when the paper is available. -->
  <img src="https://img.shields.io/badge/arXiv-Coming%20soon-B31B1B?logo=arxiv&amp;logoColor=white" alt="arXiv paper coming soon" title="Paper link coming soon">
</p>

A pi0-based student–teacher policy built on [openpi](https://github.com/Physical-Intelligence/openpi). The teacher uses future force and optical flow during training. Inference uses current RGB, robot state, and force history.

## Installation

Use Linux, an NVIDIA GPU, and Python 3.11–3.12. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone --branch forceldm https://github.com/tpy001/forcewam-275ab3be.git
cd forcewam-275ab3be
sudo apt-get install -y ffmpeg libgl1 libegl1 libglib2.0-0
GIT_LFS_SKIP_SMUDGE=1 uv sync --locked --python 3.12
source .venv/bin/activate
```

Use the pinned LeRobot version in `uv.lock`.

## Model weights

Download the pi0 base checkpoint, tokenizer, and flow VAE:

```bash
mkdir -p checkpoints
python - <<'PY'
import gcsfs
from huggingface_hub import snapshot_download
from openpi.shared import download

fs = gcsfs.GCSFileSystem(token="anon", session_kwargs={"trust_env": True})
fs.get("openpi-assets/checkpoints/pi0_base", "checkpoints/pi0_base", recursive=True)
download.maybe_download("gs://big_vision/paligemma_tokenizer.model")
snapshot_download(
    "stabilityai/sdxl-vae",
    revision="6f5909a7e596173e25d4e97b07fd19cdf9611c76",
    allow_patterns=["config.json", "diffusion_pytorch_model.bin"],
    local_dir="checkpoints/sdxl-vae",
)
PY
```

## Data preparation

### Download the prepared dataset

[PeiYuan001/forceldm-dataset](https://huggingface.co/datasets/PeiYuan001/forceldm-dataset) already contains force and flow fields. Download it and proceed to normalization:

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    "PeiYuan001/forceldm-dataset", repo_type="dataset",
    revision="ee424c789ee8e43c95d34d08ba88c9f6b39870f6",
    local_dir=Path("~/.cache/huggingface/lerobot/PeiYuan001/forceldm-dataset").expanduser(),
)
PY
```

### Prepare your own dataset

Convert synchronized recordings to LeRobot v2.1 at `~/.cache/huggingface/lerobot/<owner>/<dataset>`. Include episode/task metadata, FPS, trajectory Parquet files, and camera videos. A raw-recording converter is not included. The source data needs RGB, state, actions, and calibrated wrench measurements:

| Field | Default format |
| --- | --- |
| `observation.images.head_camera` | Head/external RGB |
| `observation.images.wrist_left_camera` | Wrist RGB |
| `observation.state` | First 7 values used as robot state |
| `action` | 7 values per frame |
| `observation.effort` | Calibrated `[Fx, Fy, Fz, Mx, My, Mz]`, N/Nm |

Task metadata provides the language prompt. Keep sensor timing, coordinate frames, and units consistent with deployment.

If wrench measurements are stored inside the state vector, adapt and run:

```bash
python scripts/add_force.py --repo_id YOUR_ORG/YOUR_DATASET
```

This modifies the dataset in place and extracts wrench from `state[7:13]`; adjust the slice for your sensor. It does not perform calibration. The script also generates `observation.is_contact`, which is unused by the current model loss. If your dataset omits this label, remove its entry from the repack mapping in `LeRobotForceLDMDataConfig.create()`.

### Compute optical flow

Compute RAFT flow from frame `t` to `t+32`, clamped to the episode end:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/compute_lerobot_future_flow_video.py \
  --repo-id YOUR_ORG/YOUR_DATASET --output-dir work/future_flow \
  --image-keys observation.images.head_camera observation.images.wrist_left_camera \
  --stream-names base_0_rgb left_wrist_0_rgb \
  --future-step 32 --device cuda

python scripts/add_flow_videos_to_lerobot.py \
  --repo-id YOUR_ORG/YOUR_DATASET \
  --flow-videos-dir work/future_flow/videos --mode copy
```

These commands generate and register `observation.future_flow.base_0_rgb` and `observation.future_flow.left_wrist_0_rgb`. They are preprocessing outputs, not required source fields. Both are needed for teacher training. Change image keys for your cameras. For a preview, use `--episodes 0 --limit-frames 16` with a separate output directory; register only complete videos.

### Compute normalization

After setting your dataset and robot transforms in the config:

```bash
JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py --config-name pi0_forceldm
```

Statistics are saved under `assets/<config>/<dataset_id>/norm_stats.json`. Recompute after changing data, calibration, or transforms. Do not use LeRobot's `meta/stats.json` as model normalization.

## Adapt to your robot

Copy `pi0_forceldm` in [config.py](src/openpi/training/config.py) to a new config name and update:

| Setting | Change |
| --- | --- |
| `repo_id` | Your LeRobot dataset ID |
| `camera_mapping` | Model RGB slots → dataset/robot camera fields |
| `LeRobotForceLDMDataConfig.create()` | Repack custom state, action, force, and generated flow keys; remove the contact entry if absent |
| [ForceLDMInputs / ForceLDMOutputs](src/openpi/policies/yuanluo_policy.py) | State selection and action output; currently 7 dimensions |
| `use_delta_joint_actions` / delta mask | Default: first 6 actions relative to state, last action absolute |
| Model and effort settings | Action horizon, wrench dimension, history length and offsets |

Camera mapping example:

```python
camera_mapping={
    "base_0_rgb": "observation.images.front_camera",
    "left_wrist_0_rgb": "observation.images.wrist_camera",
    # "right_wrist_0_rgb": "observation.images.side_camera",
}
```

The default model pads state/actions to 32 dimensions. Training reads 32 action steps and wrench offsets `[-36,-32,...,-4,0,1,...,32]`: 10 history/current samples plus 32 future samples. Update offsets and flow timing together when changing FPS or horizon. Use the same config for normalization, training, and serving.

## Training

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python scripts/train.py pi0_forceldm \
  --exp-name my_run --model.flow-vae-name checkpoints/sdxl-vae \
  --batch-size 16 --fsdp-devices 1
```

Defaults: 30,000 steps, horizon 32, EMA disabled. Checkpoints go to `checkpoints/pi0_forceldm/my_run/<step>/`, saved every 15,000 steps. Add `--resume` to continue the same run. Use `--no-wandb-enabled` to disable W&B.

### Multiple GPUs

Use one JAX process with all GPUs visible:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python scripts/train.py pi0_forceldm \
  --exp-name my_run_4gpu --model.flow-vae-name checkpoints/sdxl-vae \
  --batch-size 16 --fsdp-devices 4
```

Batch size is global and must be divisible by GPU count. `fsdp-devices` must divide GPU count: `1` replicates parameters; `4` shards across four GPUs. Multi-node training requires code changes.

The optional `pi0_forceldm_noise` config adds student query noise. Compute its normalization with `--config-name pi0_forceldm_noise`, then substitute that config name in the training command.

## Inference server

Use an existing checkpoint step directory and its matching config:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py \
  --policy.config pi0_forceldm \
  --policy.dir checkpoints/pi0_forceldm/my_run/15000 \
  --port 8000
```

Install the client in the robot environment:

```bash
pip install -e packages/openpi-client
```

Call it with real sensor data:

```python
from openpi_client import websocket_client_policy

policy = websocket_client_policy.WebsocketClientPolicy(host="SERVER_IP", port=8000)
actions = policy.infer({
    "observation.images.head_camera": head_rgb,        # uint8 HWC RGB
    "observation.images.wrist_left_camera": wrist_rgb, # uint8 HWC RGB
    "observation.state": state,                       # float32 (7,)
    "observation.effort": wrench_history,             # float32 (10, 6)
    "prompt": "Insert the plug into the socket.",
})["actions"]                                         # (32, 7)
```

Use camera keys from your config. Wrench history follows `[-36,-32,...,-4,0]` at the training FPS; pad missing initial history with the earliest measurement. Future force and flow are unnecessary at inference. The robot runtime must execute the returned action chunk at the correct control rate and enforce controller limits.

## Acknowledgments

We thank [openpi](https://github.com/Physical-Intelligence/openpi) and [TA-VLA](https://github.com/ZZongzheng0918/TA-VLA) for their open-source contributions, which this project builds upon.

## Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@inproceedings{tang2026forceldm,
  title     = {Force{LDM}: A Force-aware Latent Dynamics Model for Contact-Rich Manipulation},
  author    = {Tang, Peiyuan and Li, Yao and Zhang, Xiaodong and Zhang, Wuyang and Xia, Qin and Zhang, Yanyong and Yang, Zijiang James},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=0j9L12vSKx}
}
```
