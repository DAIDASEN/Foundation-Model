# SETR MLA 分布式训练指南 (ViT-Base/16 冻结编码器)

## 架构说明

本项目实现了 SETR (SEgmentation TRansformer) 模型，采用 Multi-Level Aggregation (MLA) 解码器进行语义分割。

### ViT-Base/16 编码器规格
根据预训练权重 `VFM_Fundus_weights.pth` 的架构：
- **模型**: ViT-Base
- **Patch size**: 16x16
- **Hidden dimension**: 768
- **Transformer层数**: 12
- **注意力头数**: 12
- **MLP维度**: 3072 (4x hidden_dim)
- **位置编码**: 197 (14x14 patches + 1 cls token)
- **输入分辨率**: 224x224

## 环境要求

- Python 3.7+
- PyTorch (支持 CUDA 11.8)
- 4 x NVIDIA GPU (例如: RTX 3060)
- 依赖包: 见 `requirements.txt`

## 安装依赖

```bash
cd SETR-main
pip install -r requirements.txt
```

## 准备预训练权重

将 ViT-Base/16 预训练权重文件放置在 SETR-main 目录下，命名为：
```
SETR-main/VFM_Fundus_weights.pth
```

该权重文件应包含以下组件的参数：
- Patch embedding (patch_embed.*)
- Position embeddings (pos_embed)
- Transformer blocks (blocks.*.*)
- Layer normalization (norm.*)

## 训练配置

项目提供了两个配置类：

### 1. ConfigViTB16Frozen (推荐)
专门为 ViT-Base/16 冻结训练设计：
- 输入尺寸: 224x224
- Patch size: 16
- Hidden size: 768
- Transformer层数: 12
- 注意力头数: 12
- **冻结 Transformer 编码器**
- 仅训练 MLA 解码器和分割头

### 2. Config (默认)
原始配置，适用于较小模型或从头训练

## 分布式训练

### 方式 1: 使用提供的脚本 (推荐)

```bash
# 4 GPU 训练，使用 ViT-Base/16 冻结配置
bash train_vitb16_4gpus.sh exp_vitb16_frozen
```

### 方式 2: 直接使用 Python 命令

```bash
# 基础训练命令
python3 -m vessel_segmentation.train_distributed \
  --gpus 4 \
  --cfg vitb16_frozen \
  --batch_size 4 \
  --epochs 100 \
  --flag exp_vitb16_frozen

# 从检查点恢复训练
python3 -m vessel_segmentation.train_distributed \
  --gpus 4 \
  --cfg vitb16_frozen \
  --batch_size 4 \
  --resume vessel_segmentation/checkpoints/exp_vitb16_frozen/model_epoch_50.pth \
  --flag exp_vitb16_frozen
```

### 参数说明

- `--gpus`: GPU 数量 (默认: 6，建议设置为 4)
- `--cfg`: 配置选择 (`default` 或 `vitb16_frozen`)
- `--batch_size`: 每个 GPU 的批次大小 (总批次 = batch_size × gpus)
- `--epochs`: 训练轮数
- `--flag`: 实验名称，用于区分不同的训练运行
- `--resume`: 恢复训练的检查点路径
- `--master_port`: 分布式训练主端口 (可选，默认自动选择)

## 训练监控

### TensorBoard

```bash
tensorboard --logdir vessel_segmentation/logs/exp_vitb16_frozen
```

监控指标：
- Training/Validation Loss
- Dice Score
- IoU (Intersection over Union)
- Accuracy
- Learning Rate

## 输出文件

训练过程会生成以下文件：

```
vessel_segmentation/
├── checkpoints/
│   └── exp_vitb16_frozen/
│       ├── best_model.pth          # 最佳模型 (基于验证集 Dice)
│       ├── final_model.pth         # 最终模型
│       └── model_epoch_*.pth       # 定期保存的检查点
└── logs/
    └── exp_vitb16_frozen/
        └── events.out.tfevents.*   # TensorBoard 日志
```

## 模型架构详情

### 编码器 (冻结)
- ViT-Base/16 Transformer
- 从预训练权重加载
- 训练时参数不更新 (requires_grad=False)

### 解码器 (可训练)
- Multi-Level Aggregation (MLA)
  - 从 Transformer 的多个层提取特征
  - 1×1 卷积投影到统一通道数
  - 特征融合
- 上采样解码器
  - 根据 patch_size 动态确定上采样步数
  - 卷积 + BatchNorm + ReLU + Upsample
- 分割头
  - 1×1 卷积输出类别预测

## 性能优化建议

### 批次大小调整
- 对于 RTX 3060 (12GB VRAM)，建议每 GPU batch_size=4-6
- 总有效批次大小 = batch_size × num_gpus
- 例如：4 GPUs × batch_size 4 = 有效批次 16

### 学习率
- 默认学习率: 1e-4
- 使用 Cosine Annealing 调度器
- 可在 `config.py` 中调整

### 数据增强
- 训练集：随机翻转、旋转、缩放
- 验证集：仅调整大小和归一化

## 常见问题

### Q1: CUDA 版本不匹配
确保 PyTorch 编译时使用的 CUDA 版本与系统 CUDA 版本兼容。
```bash
python3 -c "import torch; print(torch.version.cuda)"
```

### Q2: 显存不足 (Out of Memory)
- 减小 `--batch_size`
- 减小输入尺寸 (在 config.py 中修改 `input_size`)
- 使用梯度累积

### Q3: 预训练权重加载失败
- 检查权重文件路径是否正确
- 确认权重文件架构与配置匹配
- 查看日志中的加载信息

### Q4: 分布式训练端口冲突
使用 `--master_port` 指定不同的端口：
```bash
--master_port 12357
```

## 引用

如果使用本代码，请引用 SETR 论文：
```
@inproceedings{zheng2021rethinking,
  title={Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers},
  author={Zheng, Sixiao and Lu, Jiachen and Zhao, Hengshuang and Zhu, Xiatian and Luo, Zekun and Wang, Yabiao and Fu, Yanwei and Feng, Jiaya and Xiang, Tao and Torr, Philip HS and others},
  booktitle={CVPR},
  year={2021}
}
```
