# forceWAM / ForceLDM 复现指南与实际问题记录

本指南记录 `forceWAM` 分支在 Ailab-Server2 上的环境准备。**已验证环境和单样本数据链路；尚未执行正式训练或评测，不能据此报告论文复现指标。**

## 1. 代码、环境与目录

```bash
git clone https://github.com/tpy001/forcewam-275ab3be.git
cd forcewam-275ab3be
```

原始代码提交为 `6125718acd3a4ebd99b7fa39a070b7055204d77e`，分支为 `forceWAM`，保留原始 15 次提交。环境准备对代码的改动仅为：将 `pi0_latent_flow_multiview` 的数据集 ID 指向用户指定的数据集；同步依赖锁文件；增加本指南入口与本地工具目录的忽略规则。

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

`pi0_latent_flow_multiview` 还依赖 [stabilityai/sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae)。仅下载 π₀ 权重并不够。此处固定 revision `6f5909a7e596173e25d4e97b07fd19cdf9611c76`，下载到本项目，后续训练显式传入本地路径：

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

本次准备选择现有 `pi0_latent_flow_multiview`，因为该数据集同时具有两路 RGB、两路 future flow 及六维标定力。它是可运行配置的接入准备，**不代表已经确认论文全部训练超参数或完成复现**。其余原始配置仍保留各自的数据集 ID，不能不经检查直接替换使用。

训练前必须用当前标定数据重新计算模型输入对应的归一化统计：

```bash
# 从仓库根目录运行，先 source .venv/bin/activate 并设置第 3 节环境变量。
JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py \
  --config-name pi0_latent_flow_multiview
```

输出预期为：

```text
assets/pi0_latent_flow_multiview/PeiYuan001/forceldm-dataset/norm_stats.json
```

不要直接把数据集 `meta/stats.json` 复制为模型统计。模型取 7 维 state，默认将前 6 维 action 转为相对于当前 state 的差值，还读取时间窗口内的 effort，因此应使用仓库脚本计算。也不要复用旧的未标定 wrench 模型统计。`--max-frames` 仅适合冒烟检查；全量训练前应生成全量统计。该脚本会读取视频，耗时取决于 CPU、存储与解码速度。

下面是**后续 GPU 作业的训练入口示例**，应放在已获分配、挂载好项目数据的作业/worker 中执行；本次没有直接在开发机启动训练，也没有提交 GPU 作业：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python scripts/train.py pi0_latent_flow_multiview \
  --exp-name forceldm_multiview_seed42 \
  --seed 42 \
  --model.flow-vae-name checkpoints/sdxl-vae \
  --no-wandb-enabled
```

示例沿用原配置的 30,000 步、batch size 16、action horizon 32、EMA 关闭、FSDP 1；尚未通过完整模型训练验证显存与收敛。使用新的实验目录，不默认添加 `--overwrite`。需要接入 W&B 时再按实际日志方案启用。

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
- OpenCV、TorchCodec、训练配置、data loader、latent-flow 模型模块导入。
- `scripts/train.py pi0_latent_flow_multiview --help` 和 `scripts/compute_norm_stats.py --help`。
- Episode 0 的真实样本读取：32×7 action、42×6 effort、两路 RGB 和两路光流；当前 wrench 一致性检查。
- 数据重打包、动作差分、图像 resize、tokenizer 和 padding：state 为 32 维，actions 为 32×32，effort 为 42×6，prompt 为 48 tokens，两路 RGB 为 224×224×3。此检查未应用尚未计算的归一化统计。

- 完整数据 2,394 个文件已下载，逐文件存在性和大小检查通过。
- 官方 π₀ 基础权重 33 个对象逐文件 MD5 与大小校验通过，Orbax checkpoint 元信息可读。
- 本地 SDXL VAE 的 PyTorch→Flax 转换和 CPU 编码通过。
- 原归一化脚本以 8 帧完成冒烟检查，且归一化后的训练 dataloader 返回 `(2,32)` state 和 `(2,32,32)` actions。**该统计仅存在 `.repro/smoke-assets/`，正式 `assets/` 中尚未生成全量统计。**

机器上的过程记录集中在 `.repro/`，包括 `environment-report.json`、`verification.log`、`model-input.log`、`norm-smoke-verified.log`、`vae-prepare-verified.log`、安装和资源下载日志。准备过程已结束，没有遗留下载或自检进程。尚未提交训练作业、生成训练 checkpoint 或计算任务成功率；下一步应先生成全量归一化统计，再确定作业资源与实验方案。