升级到 CUDA 12.8 与 PyTorch 2.8 的说明

目标
- 在具有 CUDA 11.8 的机器上，升级/切换到 CUDA 12.8 并使用与其兼容的 PyTorch 2.8 环境（本文件作为迁移与运行提示）。

注意
- 你当前机器的驱动必须支持 CUDA 12.x。驱动与 CUDA 版本不兼容可能会导致运行失败。
- 如果你不能升级系统 CUDA，建议使用 conda 创建一个含 CUDA 12.8 的环境（安装带有 cudatoolkit=12.8 的 PyTorch）或使用官方 PyTorch+CUDA 12.8 的 pip 轮子（根据官方命令）。

建议的 conda 环境命令（推荐）

1) 创建 conda 环境并安装 PyTorch 2.8 + CUDA 12.8

```bash
conda create -n setr_py28 python=3.10 -y
conda activate setr_py28
# 安装 PyTorch 2.8 + torchvision + torchaudio (cudatoolkit 12.8)
conda install pytorch==2.8.0 torchvision pytorch-cuda=12.8 -c pytorch -c nvidia -y
# 其余依赖
pip install -r requirements.txt
pip install einops
```

2) 或使用 pip 安装（若官方发布了对应 pip/whl）：

```bash
python -m pip install torch==2.8.0+cu128 torchvision --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

如何用 4 张 GPU 启动训练（推荐使用 torchrun）

在项目根目录下：

```bash
# 使用 torchrun 启动 4 GPU
torchrun --nproc_per_node=4 vessel_segmentation/train_distributed.py --gpus 4 --cfg vitb16_frozen --flag vitb16_cuda12_py28
```

如果你在使用 ssh 或多节点集群，请参考 torchrun/torch.distributed 文档配置 MASTER_ADDR, MASTER_PORT, NODE_RANK 等环境变量。

快速检查
- 运行一个小脚本来检查 torch 和 cuda 信息：

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

常见问题
- 如果 CUDA 驱动不兼容：回退到系统驱动或使用对应旧版 PyTorch 或使用容器（NVIDIA NGC / Docker image）。

"""
