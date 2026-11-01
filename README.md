# pi0_forceldm

**pi0_forceldm** 是基于 openpi / π0 的双专家模型，使用未来力与双视角光流进行蒸馏。数据集为 [PeiYuan001/forceldm-dataset](https://huggingface.co/datasets/PeiYuan001/forceldm-dataset)。

环境安装、固定依赖和权重、数据下载、归一化、单卡训练、离线 W&B 以及实际问题见 [复现指南](README_REPRODUCE.md)。

基础训练配置为 `pi0_forceldm`：batch size 16，30,000 步，action horizon 32，EMA 关闭，学生 future-query 噪声关闭。

噪声实验配置为 `pi0_forceldm_noise`：学生 future-query 噪声最大强度为 0.3，在训练进度 30%～70% 线性增加，之后维持 0.3。30,000 步时约前 9,000 步不加噪，9,000～21,000 步逐渐增加；其他超参数与基础配置相同。

```bash
# 数据与归一化处理相同，可以复用基础配置已经计算好的统计。
mkdir -p assets/pi0_forceldm_noise/PeiYuan001/forceldm-dataset
cp assets/pi0_forceldm/PeiYuan001/forceldm-dataset/norm_stats.json \
   assets/pi0_forceldm_noise/PeiYuan001/forceldm-dataset/norm_stats.json

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python scripts/train.py pi0_forceldm_noise \
  --exp-name forceldm_noise_seed42 --seed 42 \
  --model.flow-vae-name checkpoints/sdxl-vae --log-interval 100
```

EMA 功能完整保留，包括配置、权重更新以及 checkpoint 保存和恢复；当前配置的 `ema_decay=None` 仅表示本次实验关闭，可通过 `--ema-decay 0.99` 开启。

```bash
# 先按复现指南准备环境、数据、基础权重、SDXL VAE 和归一化统计。
# 在已经分配的 GPU 上执行；每次新实验使用不同名称。
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python scripts/train.py pi0_forceldm \
  --exp-name forceldm_multiview --seed 42 \
  --model.flow-vae-name checkpoints/sdxl-vae --log-interval 100
```

总 loss 为学生 action loss + 教师 action loss + 0.5 × force distillation loss + 0.5 × flow distillation loss。两种蒸馏分别归一化后加权。

保留的入口：

- `scripts/train.py`：训练与 checkpoint 恢复（`--resume`）。
- `scripts/compute_norm_stats.py`、`scripts/compute_norm_stats_no_video.py`：归一化统计。
- `scripts/compute_lerobot_future_flow_video.py`、`scripts/add_flow_videos_to_lerobot.py`、`scripts/add_force.py`：力/光流数据准备；指定数据集已提供这些字段，无需重复处理。
- `scripts/serve_policy.py`：通过指定 config 和 checkpoint 提供策略服务。

```bash
python scripts/serve_policy.py \
  --policy.config pi0_forceldm \
  --policy.dir checkpoints/pi0_forceldm/forceldm_multiview/15000
```

代码源自 [Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi) 和 USTC forceWAM 分支。许可证见 [LICENSE](LICENSE)。

## 两路或三路 RGB 相机

在所用训练配置的 `data=LeRobotForceLDMDataConfig(...)` 中配置 `camera_mapping`。键是模型相机槽位，值是数据集 / 推理输入中的实际字段名。默认保持头部和左腕两路；三路示例：

```python
camera_mapping={
    "base_0_rgb": "observation.images.head_camera",
    "left_wrist_0_rgb": "observation.images.wrist_left_camera",
    "right_wrist_0_rgb": "observation.images.wrist_right_camera",
},
```

第三路也可以来自侧面相机：将最后一行的值改为自己的字段名，例如 `observation.images.side_camera`。训练数据和推理输入须提供所配置的字段；缺失相机会报错。两路配置不会添加第三路占位图像。

第三路会经过重打包、resize、观测预处理和共享视觉编码，进入学生与教师共用的视觉前缀。相机数增加会增加视觉 token 数和显存需求。当前教师的未来光流监督仍使用头部和左腕两路；增加第三路当前 RGB 不要求增加第三路光流。

训练与推理使用同一份相机映射。模型内部支持的槽位为 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`；槽位名称只作内部映射，实际相机字段名可自行配置。当前数据集只有两路 RGB，所以默认训练配置仍保留两路。

## 单卡 80GB 配置验证（2026-11-01）

目标配置为单卡 A100 80GB：默认两路 RGB、batch 16、FSDP 1、EMA 关闭，使用本指南的固定依赖、基础权重、SDXL VAE 和全量归一化统计。基础配置和 noise 配置已在单张 **A800-SXM4-80GB（SM 8.0）** 上完成以下短测；没有直接使用 A100 实测。

| 配置 | batch | 优化步数 | nvidia-smi 采样峰值 | JAX 活跃分配峰值 |
| --- | ---: | ---: | ---: | ---: |
| `pi0_forceldm` | 16 | 3 | 71.97 GiB | 55.16 GiB |
| `pi0_forceldm_noise` | 16 | 3 | 71.94 GiB | 55.21 GiB |

每种配置均执行完整前向、反向和优化器更新，3 步指标均为有限值，无 OOM。短测将总步数临时压缩为 3，以覆盖 noise 强度 0、0.15、0.3；仓库正式配置仍为 30,000 步，保留原 30%～70% 噪声调度。测试使用 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`；nvidia-smi 数值包含 JAX 预分配，和活跃张量峰值口径不同。

[NVIDIA 列出 A100 的计算能力为 8.0](https://developer.nvidia.com/cuda/gpus)，与本次实测 A800 的 SM 版本相同；结合相同的 80GB 显存，判断默认配置兼容单卡 A100 80GB。这是基于同架构运行结果的兼容性判断。三路 RGB 或开启 EMA 未纳入本次 GPU 显存验证；两路/三路映射及视觉 token、注意力连通性已有 7 项测试通过。

此次短测作业和 worker 均已结束，没有恢复原长训练。原 checkpoint、日志和离线 W&B 保留；没有上传训练 loss 或离线文件。
