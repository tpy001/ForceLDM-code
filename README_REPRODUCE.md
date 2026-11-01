# pi0_forceldm 复现指南与实际问题记录

本指南记录 `forceWAM` 分支在 Ailab-Server2 上的环境准备。**已验证环境和数据链路，并已验证单卡 batch16 离线训练（现已按用户要求停止）；尚未执行独立评测，不能据此报告论文复现指标。**

## 1. 代码、环境与目录

```bash
git clone https://github.com/tpy001/forcewam-275ab3be.git
cd forcewam-275ab3be
```

原始代码提交为 `6125718acd3a4ebd99b7fa39a070b7055204d77e`，分支为 `forceWAM`，保留原始 15 次提交。初始环境准备将双视角配置（现名 `pi0_forceldm`）的数据集 ID 指向用户指定的数据集；同步依赖锁文件；增加本指南入口与本地工具目录的忽略规则。

本次服务器目录：`/data/data/tpy/forcewam-275ab3be`。其他机器可以使用自己的克隆目录，以下命令从仓库根目录执行。

| 项目 | 本次实测 |
| --- | --- |
| 系统 | Ubuntu 22.04 / Linux x86_64 |
| GPU | NVIDIA A800-SXM4-80GB，一张，编号 0 |
| NVIDIA 驱动 | 595.71.05 |
| Python | 3.12.13，独立 `.venv` |
| uv | 0.13.6 |
| JAX / jaxlib / Flax | 0.5.3 / 0.5.3 / 0.10.2 |
| PyTorch | 2.7.1+cu126 |
| TorchCodec / Transformers / Diffusers | 0.4.0 / 4.53.2 / 0.37.0 |
| NumPy / huggingface-hub / tyro | 1.26.4 / 0.36.2 / 1.0.10 |
| LeRobot | 0.1.0，源码固定到 `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |
| 系统 FFmpeg | 4.4.2 |

`nvidia-smi` 中的 CUDA 13.2 是驱动可支持的 CUDA 版本。实际 PyTorch wheel 使用 CUDA 12.6，项目 JAX 使用 CUDA 12 wheel；不要仅因显示 13.2 就修改依赖中的 CUDA 版本。

## 2. 安装系统库和 Python 环境

项目要求 Python `>=3.11,<3.13`。本机默认 `python3` 是 3.10.12，不能直接使用；选择已安装的 `/usr/bin/python3.12`，不替换系统 Python。

```bash
# root 用户执行；普通用户需由管理员安装或使用 sudo。
apt-get update
apt-get install -y --no-install-recommends ffmpeg libgl1 libegl1 libglib2.0-0

# uv 安装在项目自己的工具目录，不修改其他项目环境。
mkdir -p .repro
python3.12 -m pip install --target .repro/tools 'uv==0.13.6'
export GIT_LFS_SKIP_SMUDGE=1
export UV_CACHE_DIR="$PWD/.repro/uv-cache"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
.repro/tools/bin/uv sync --locked --python /usr/bin/python3.12
source .venv/bin/activate
.repro/tools/bin/uv pip check --python .venv/bin/python
```

使用本仓库更新后的 `uv.lock`。**原始代码中的锁文件与 `pyproject.toml` 不一致**，原样 `uv sync --locked` 会报 `The lockfile ... needs to be updated`。本次已先备份原锁文件，然后运行 `uv lock --python /usr/bin/python3.12` 再 `uv sync --locked`。关键模型依赖保持项目指定版本；没有为安装而升级 JAX、Flax 或 PyTorch。

本机的备份及安装日志位于 `.repro/uv.lock.original`、`.repro/install-resolved.log`；这类机器相关文件不提交到 Git。

## 3. 数据集下载与版本

数据集：[PeiYuan001/forceldm-dataset](https://huggingface.co/datasets/PeiYuan001/forceldm-dataset)。本次固定的 revision：

```text
ee424c789ee8e43c95d34d08ba88c9f6b39870f6
```

该版本元信息：LeRobot v2.1，474 episodes，362,339 frames，30 Hz，6 类任务；共有 2,394 个仓库文件，合计 9,560,254,738 字节。仅有原始训练 split，不能直接把训练集当作独立测试集。

```bash
export HF_HOME="$PWD/.cache/huggingface"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export HF_HUB_DISABLE_XET=1
export OPENPI_DATA_HOME="$PWD/.cache/openpi"

python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="PeiYuan001/forceldm-dataset",
    repo_type="dataset",
    revision="ee424c789ee8e43c95d34d08ba88c9f6b39870f6",
    local_dir=Path(os.environ["HF_LEROBOT_HOME"]) / "PeiYuan001/forceldm-dataset",
    max_workers=4,
)
PY
```

必须先设置这些变量再导入 LeRobot。训练配置不传 `root`，LeRobot 会从 `HF_LEROBOT_HOME/<repo_id>` 查找文件；将文件下载到其他目录不会自动改变训练的数据位置。以上变量在新的 shell、SSH 会话或作业 worker 中都需要重新设置。

服务器直连 Hugging Face 失败后，本次下载使用服务器已有的私有代理配置；没有修改全局 Git/系统代理。其他机器按自己的网络设置使用 `HTTPS_PROXY` 等变量，或直接联网。不要把代理账号、密码或访问 token 写入 README、脚本正文或 Git。

### 已核对的数据映射

| 数据字段 | 含义 / 单样本读取结果 |
| --- | --- |
| `observation.images.head_camera` | RGB，360×640×3，AV1 视频 |
| `observation.images.wrist_left_camera` | RGB，360×640×3，AV1 视频 |
| `observation.future_flow.base_0_rgb` | 224×224×3，H.264 光流可视化 |
| `observation.future_flow.left_wrist_0_rgb` | 224×224×3，H.264 腕部光流可视化 |
| `observation.state` | 原始 27 维；策略输入取前 7 维 |
| `action` | 7 维；配置读取未来 32 步 |
| `observation.effort` | 已标定世界系 `[Fx,Fy,Fz,Mx,My,Mz]`，单位 N/Nm |
| 力历史 | `[-36,-32,...,-4,0,1,...,32]`，42×6 |
| `observation.is_contact` | 原始接触标签；数据集未按标定后的力重新生成标签 |

力输入的前 10 帧是含当前帧的历史，后 32 帧是未来监督；时间单位是帧，以 30 Hz 转为秒。已核对样本中 `state[7:13]` 与当前帧的标定 wrench 一致。光流是 RGB 可视化，不能解释为具有真实位移单位的二维向量。

## 4. Tokenizer 和基础权重

配置使用的基础权重为官方 `gs://openpi-assets/checkpoints/pi0_base`，本地预期位置是 `checkpoints/pi0_base/params`。它不是 `pi05_base`，也不是 PyTorch 导出的权重。

其他机器可通过项目已安装的 gcsfs 获取：

```bash
mkdir -p checkpoints
python - <<'PY'
import gcsfs
from openpi.shared import download
fs = gcsfs.GCSFileSystem(token="anon", session_kwargs={"trust_env": True})
fs.get("openpi-assets/checkpoints/pi0_base", "checkpoints/pi0_base", recursive=True)
print(download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon", "session_kwargs": {"trust_env": True}}))
PY
```

Google Storage 也需要可用的网络通路。下载中断不能仅凭目录存在就视为完成。本次从官方 GCS 清单固定对象 generation，下载 33 个文件，共 12,014,440,489 字节，并逐文件核对大小和 MD5；清单保存在服务器 `.repro/pi0-base-manifest.json`。

Tokenizer 使用同一服务器已有的公共 `paligemma_tokenizer.model` 缓存，复制至本项目 `OPENPI_DATA_HOME/big_vision/`。训练和验证使用本项目新建的 `.venv`；初始数据下载进程仅复用了已有下载器，没有修改其环境。

### 光流编码器 SDXL VAE

`pi0_forceldm` 还依赖 [stabilityai/sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae)。仅下载 π₀ 权重并不够。此处固定 revision `6f5909a7e596173e25d4e97b07fd19cdf9611c76`，下载到本项目，后续训练显式传入本地路径：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="stabilityai/sdxl-vae",
    revision="6f5909a7e596173e25d4e97b07fd19cdf9611c76",
    allow_patterns=["config.json", "diffusion_pytorch_model.bin"],
    local_dir="checkpoints/sdxl-vae",
)
PY
```

已实测通过代码原本使用的 `FlaxAutoencoderKL.from_pretrained(..., from_pt=True)` 转换和 CPU 编码。224×224 图像得到 `(1,28,28,4)` 的 NHWC latent；不要把 Flax 输出误按 PyTorch 的 NCHW 布局断言。Diffusers 0.37.0 会输出 Flax 弃用提示，本次使用锁定版本，不据此改写模型实现。

## 5. 归一化统计与后续训练入口

本次准备选择现有 `pi0_forceldm`，因为该数据集同时具有两路 RGB、两路 future flow 及六维标定力。它是可运行配置的接入准备，**不代表已经确认论文全部训练超参数或完成复现**。其他旧实验配置已在仓库精简时删除。

训练前必须用当前标定数据重新计算模型输入对应的归一化统计：

```bash
# 从仓库根目录运行，先 source .venv/bin/activate 并设置第 3 节环境变量。
JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py \
  --config-name pi0_forceldm
```

输出预期为：

```text
assets/pi0_forceldm/PeiYuan001/forceldm-dataset/norm_stats.json
```

不要直接把数据集 `meta/stats.json` 复制为模型统计。模型取 7 维 state，默认将前 6 维 action 转为相对于当前 state 的差值，还读取时间窗口内的 effort，因此应使用仓库脚本计算。也不要复用旧的未标定 wrench 模型统计。`--max-frames` 仅适合冒烟检查；全量训练前应生成全量统计。该脚本会读取视频，耗时取决于 CPU、存储与解码速度。

下面是 GPU 作业的训练入口示例，应放在已获分配、挂载好项目数据的作业/worker 中执行：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python scripts/train.py pi0_forceldm \
  --exp-name forceldm_multiview_seed42 \
  --seed 42 \
  --model.flow-vae-name checkpoints/sdxl-vae \
  --no-wandb-enabled
```

示例沿用原配置的 30,000 步、batch size 16、action horizon 32、EMA 关闭、FSDP 1；已验证单卡 batch16 能实际执行优化；尚未完成全部 30,000 步及独立评测。使用新的实验目录，不默认添加 `--overwrite`。需要接入 W&B 时再按实际日志方案启用。

## 6. 实际问题及解决方法

| 实际遇到的问题 | 处理与复现要点 |
| --- | --- |
| 系统默认 Python 3.10 不满足要求 | 显式选择 Python 3.12，安装到独立 `.venv` |
| `uv` 不在 PATH | 项目内安装；pip `--target` 后的可执行文件在 `.repro/tools/bin/uv`，不是 `.repro/tools/uv/uv` |
| 原始锁文件与声明不一致，`--locked` 失败 | 备份并重新 lock；本仓库已提供更新后的锁文件 |
| 缺少 FFmpeg 与 OpenGL 库 | 安装 ffmpeg、libgl1、libegl1、libglib2.0-0；实际验证了 cv2、TorchCodec 和 AV1/H.264 解码 |
| `apt-get update` 的 NVIDIA/PPA 外网源超时 | 内网 Ubuntu 镜像可用，最终使用已有镜像索引完成安装；不需要改动 CUDA 驱动 |
| 并发 apt 命令发生 dpkg 锁冲突 | 等待当前安装结束后再检查；不要删除 dpkg 锁文件 |
| Hugging Face 直连失败 | 使用已有私有服务器代理，并固定 revision；不在仓库记录凭据 |
| 原配置指向 `llly/all_0409_stage_flow` | 仅修改双视角光流配置为指定数据集，验证实际字段和时间窗口 |
| README 引用了本分支不存在的 examples/docs 等路径 | 以当前分支实际脚本为准，本指南不依赖那些缺失路径 |
| `.gitmodules` 有条目但本提交没有对应 gitlink | `git submodule update` 不会自动得到 LIBERO；本次真实机器人数据训练不需要 LIBERO 仿真环境 |
| 原 `train.sh` 含旧机器绝对环境路径、GPU 1/4/6/7 及 `--overwrite` | 不直接照搬；使用本项目环境和实际分配的 GPU 编号 |
| 数据集没有独立测试 split | 另行明确划分和评测方案，不能把训练损失当作任务成功率 |

## 7. 验证结果与边界

已实测通过：

- 230 个已安装包的依赖一致性检查。
- PyTorch 与 JAX 的 A800 CUDA 小规模矩阵运算。
- OpenCV、TorchCodec、训练配置、data loader、pi0_forceldm 模型模块导入。
- `scripts/train.py pi0_forceldm --help` 和 `scripts/compute_norm_stats.py --help`。
- Episode 0 的真实样本读取：32×7 action、42×6 effort、两路 RGB 和两路光流；当前 wrench 一致性检查。
- 数据重打包、动作差分、图像 resize、tokenizer 和 padding：state 为 32 维，actions 为 32×32，effort 为 42×6，prompt 为 48 tokens，两路 RGB 为 224×224×3。此检查未应用尚未计算的归一化统计。

- 完整数据 2,394 个文件已下载，逐文件存在性和大小检查通过。
- 官方 π₀ 基础权重 33 个对象逐文件 MD5 与大小校验通过，Orbax checkpoint 元信息可读。
- 本地 SDXL VAE 的 PyTorch→Flax 转换和 CPU 编码通过。
- 原归一化脚本以 8 帧完成冒烟检查，且归一化后的训练 dataloader 返回 `(2,32)` state 和 `(2,32,32)` actions。**该统计仅存在 `.repro/smoke-assets/`，随后已通过下述全量统计脚本生成正式训练统计。**

机器上的过程记录集中在 `.repro/`，包括 `environment-report.json`、`verification.log`、`model-input.log`、`norm-smoke-verified.log`、`vae-prepare-verified.log`、安装和资源下载日志。准备过程已结束，没有遗留下载或自检进程。首次在线 W&B 作业在认证阶段退出；随后已改为 offline 并执行优化、生成 checkpoint，详见第 9 节；未计算任务成功率。

## 8. 单卡训练准备记录（2026-10-31）

- 按 `pi0_forceldm` 的默认 **batch size 16**、30,000 步、seed 42、FSDP 1 提交；没有改用 batch 4，也没有增加梯度累积。
- 为查看前几步，日志间隔设为 1；W&B project 为 `forceldm`，实验名为 `forceldm_multiview_bs16_seed42_20261031`。训练记录总 loss、student/teacher action loss、future force/flow distillation loss 和 gradient norm。
- 全量统计已经完成。`scripts/compute_norm_stats_no_video.py` 缓存相同的数值列，保持 LeRobot 轨迹内索引截断、原动作差分、原 RunningStats 和 batch 16 顺序，只跳过不参与这三类统计的视频解码。先在轨迹起点、内部、交界和终点对照原 LeRobot reader，检查数值完全一致。
- 统计使用 362,336 帧，末尾 3 帧按原 dataloader 的 `drop_last=True` 规则丢弃；正式输出已写入 `assets/pi0_forceldm/PeiYuan001/forceldm-dataset/norm_stats.json`。可用以下命令重新生成：

```bash
source .venv/bin/activate
python scripts/compute_norm_stats_no_video.py
```

- 数值 reader 缓存的轨迹边界使用独立 NumPy 拷贝，统计阶段使用单进程，避免多进程共享 Torch storage 后原 NumPy 视图失效；该调整不改变训练的 dataloader。
- 新 Worker 镜像的 apt 索引为空，首次启动出现 `Unable to locate package ffmpeg`。启动脚本改为先用内网 Ubuntu jammy 源更新索引，再安装系统库；不修改 NVIDIA 驱动。
- API key 仅放在权限为 600 的私有认证文件，由 worker 读取并调用 `wandb.login(..., verify=True)`；密钥不写入作业 spec、README 或 Git。开发节点验证返回 403，worker 会分别尝试直连和服务器代理。只有认证成功才进入训练，失败时退出释放 GPU，不能把写入 netrc 等同于已验证登录成功。

作业名称：`tpy-forcewam-bs16-1031-r1`；单 GPU、16 CPU、128 GiB 内存、32 GiB shm。共享日志位于 `.repro/clusterx/logs/tpy-forcewam-bs16-1031-r1.log`，spec/receipt 保存在 `.repro/clusterx/specs/`。作业提交或 Worker RUNNING 都不等于已有训练 loss，须查看日志中的实际 step 和 W&B run。


### 本次启动最终状态

` tpy-forcewam-bs16-1031-r1 ` 的作业和 Worker 最终均为 FAILED。系统库安装已修复；API key 直连验证超时，经服务器代理验证返回 403 / AuthenticationError，因此按入口检查退出，未进入训练。没有训练 loss、已创建的在线 W&B run 或训练 checkpoint；本次两个 Worker 均已结束。不能将此状态称作训练成功或 W&B 登录成功。下一步需要确认可用的 W&B API key、对应服务地址和到该服务的认证网络通路。


## 9. 按用户要求改用本地 W&B（不上传）

在线认证不可用后，用户明确选择 W&B offline。本次作业设置 `WANDB_MODE=offline`，保留 `wandb_enabled=True`，不调用 login、不读 API key、不执行任何 `wandb sync`。离线模式仍记录原来的 loss、梯度范数和图像；`--log-interval 1` 使前几步可逐步查看。

- 作业：`tpy-forcewam-bs16-off-1031`，单张 A800 80GB。
- 默认 batch size：16（日志已实际确认 `local_batch_size: 16`），FSDP 1，无梯度累积，目标 30,000 步。
- 实验目录：`checkpoints/pi0_latent_flow_multiview/forceldm_multiview_bs16_seed42_offline_20261031/`。
- 离线 run：`.repro/wandb/wandb/offline-run-20261031_144840-ri7302ns/`。
- 纯文本训练日志：`.repro/clusterx/logs/tpy-forcewam-bs16-off-1031.log`。

```bash
# 在同一项目根目录，已设置第3节环境变量并激活 .venv 后：
WANDB_MODE=offline WANDB_DIR="$PWD/.repro/wandb" \
python scripts/train.py pi0_forceldm \
  --exp-name forceldm_bs16_seed42_offline \
  --project-name forceldm \
  --model.flow-vae-name "$PWD/checkpoints/sdxl-vae" \
  --seed 42 --log-interval 1 --wandb-enabled
```

上述是已分配 GPU 的 worker 内执行的训练入口；不要对同一个正在运行的实验重复启动。查看文本 loss：

```bash
tail -f .repro/clusterx/logs/tpy-forcewam-bs16-off-1031.log
# 仅查看已有 step 指标：
grep 'Step [0-9]' .repro/clusterx/logs/tpy-forcewam-bs16-off-1031.log
```

W&B offline 生成本地 run 文件，不等同于自动提供本地网页仪表盘。此次不会上传离线文件或 loss，也不会把它们提交到公开 GitHub。模型初始化和首步 XLA/cuDNN 编译可能较久，`slow_operation_alarm` 消息本身不代表训练已失败；须区分初始化日志与真正的 `Step ... loss=...`。


### 当前状态与精简范围（2026-11-01）

训练任务已按用户要求停止，任务状态为 STOPPED，GPU worker 已终止。保留已有 checkpoint、离线 W&B、数据和日志；停止操作不会额外保存 checkpoint。最近完整 checkpoint 为第 15,000 步，后续日志中的优化步骤不等同于可恢复的 checkpoint。

仓库保留 `pi0_forceldm` 基础配置及下述噪声变体。已移除 LIBERO / IsaacLab、旧 SEER / 独立 TaVLA 模型、probe 实验和其他旧配置；保留 π0 基础组件、力/光流数据处理、归一化、训练、checkpoint 与策略服务。pi0_forceldm 的两项辅助函数改为从 π0 模块导入，与原 TaVLA 版本逐函数 AST 对比一致。batch 16、30,000 步、EMA 关闭、噪声配置和损失权重保持原值。

本地训练数据、模型权重、私有认证、日志和离线 W&B 均由 `.gitignore` 排除，不应提交到公开仓库。代码清理没有创建 commit 或 push。

EMA 的配置、权重更新、checkpoint 保存和恢复功能完整保留；当前实验配置使用 `ema_decay=None`，需要时可通过 `--ema-decay 0.99` 开启。

## 10. 方法统一命名为 pi0_forceldm

- 方法和基础训练配置：`pi0_forceldm`；噪声变体：`pi0_forceldm_noise`。
- 模型实现：`src/openpi/models/pi0_forceldm.py`，类名 `Pi0ForceLDM`。
- 模型配置：`Pi0ForceLDMConfig`；带力观测的数据结构位于 `model_forceldm.py`。
- 训练、归一化、数据加载、策略服务和本文命令均使用新名称。仅更改名称，不修改网络结构、参数树、损失、超参数和 EMA 功能。
- 新实验默认写到 `checkpoints/pi0_forceldm/<exp_name>/`，归一化统计位于 `assets/pi0_forceldm/PeiYuan001/forceldm-dataset/`。

服务器已有统计从旧配置目录复制到新目录，逐文件 SHA-256 校验一致。旧统计、checkpoint、日志和离线 W&B 原路径保留；没有重启训练。

历史实验的 checkpoint 仍在 `checkpoints/pi0_latent_flow_multiview/forceldm_multiview_bs16_seed42_offline_20261031/15000`。策略服务可同时指定新配置名和旧 checkpoint 路径。`--resume` 按当前配置名和实验名查找目录，不能只改名称就自动找到旧实验；后续续训需要显式建立目录映射或迁移完整实验目录（包括训练状态），此次没有执行迁移。

## 11. 开启学生 future-query 噪声

使用 `pi0_forceldm_noise`，保留 `pi0_forceldm` 作为无噪声对照。参数为 `student_future_query_noise_scale_max=0.3`、`student_future_query_noise_start_ratio=0.3`、`student_future_query_noise_end_ratio=0.7`；调度使用 `step / (num_train_steps - 1)`。因此 30,000 步时约第 9,000 步才开始注入，约第 21,000 步达到最大强度，并非从 step 0 就加噪。

噪声添加到学生的未来力和光流 query，按各自 query 的 RMS 缩放；0.3 是相对噪声系数。默认推理不加噪。batch 16、EMA 关闭、损失权重、数据映射和其他训练参数沿用基础配置，EMA 功能仍可开启。

两种配置的数据处理完全相同，可将 `assets/pi0_forceldm/PeiYuan001/forceldm-dataset/norm_stats.json` 复制到 `assets/pi0_forceldm_noise/PeiYuan001/forceldm-dataset/norm_stats.json`，也可运行 `python scripts/compute_norm_stats.py --config-name pi0_forceldm_noise` 重新计算。服务器已复制并校验统计一致。新 checkpoint 默认写入 `checkpoints/pi0_forceldm_noise/<exp_name>/`，不覆盖基础实验。完整启动示例见主 README；此次仅添加配置，没有重启训练。

## 12. 固定使用光流蒸馏

已删除旧 RGB 消融开关 `use_future_rgb_instead_of_flow`、`future_rgb_step` 及相应数据采样、未来 RGB 观测字段和模型分支。`pi0_forceldm` 和 `pi0_forceldm_noise` 均固定使用 `flow_img` / `wrist_flow_img` 作为教师的双视角未来视觉监督；两路当前 RGB 相机输入继续保留。噪声配置、损失权重、归一化统计、模型参数树以及 EMA 功能保持原样。

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
