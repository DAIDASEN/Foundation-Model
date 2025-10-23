# SETR ViT-Base/16 冻结训练 - 快速上手指南

## 概述

本项目实现了基于 SETR (SEgmentation TRansformer) 的血管分割模型，支持：
- ✓ ViT-Base/16 预训练编码器加载和冻结
- ✓ 4 GPU 分布式训练（CUDA 11.8）
- ✓ MLA (Multi-Level Aggregation) 解码器训练
- ✓ 自动位置编码插值（支持不同输入尺寸）

## 环境准备

### 1. 安装依赖

```bash
cd SETR-main
pip install -r requirements.txt
```

### 2. 准备预训练权重

将 ViT-Base/16 预训练权重放置在 SETR-main 目录：

```bash
# 方式1: 直接复制
cp /path/to/your/VFM_Fundus_weights.pth SETR-main/

# 方式2: 使用符号链接（推荐）
bash setup_weights.sh /path/to/your/VFM_Fundus_weights.pth
```

**权重文件要求**：
- 文件名: `VFM_Fundus_weights.pth`
- 架构: ViT-Base/16 (768 dim, 12 layers, 12 heads)
- 包含组件: patch_embed, pos_embed, blocks, norm

### 3. 准备数据集

将数据集放置在项目根目录下的 `data/FIVES_vessel/` 或修改 `config.py` 中的 `data_root`：

```
data/FIVES_vessel/
├── images/
│   ├── train/
│   │   ├── image1.png
│   │   └── ...
│   └── test/
│       └── ...
└── annotations/
    ├── train/
    │   ├── image1.png  # 对应的掩码
    │   └── ...
    └── test/
        └── ...
```

## 验证配置

运行测试脚本确保环境正确配置：

```bash
python3 test_vitb16_config.py
```

应该看到：
- ✓ 配置参数验证通过
- ✓ 模型创建成功
- ✓ 前向传播测试通过
- ✓ 分布式设置检查完成

## 开始训练

### 快速开始（4 GPU 训练）

```bash
bash train_vitb16_4gpus.sh my_experiment
```

### 自定义训练参数

```bash
python3 -m vessel_segmentation.train_distributed \
  --gpus 4 \
  --cfg vitb16_frozen \
  --batch_size 4 \
  --epochs 100 \
  --flag my_experiment \
  --master_port 12356
```

### 查看训练配置示例

```bash
# 查看 4 GPU 配置
python3 training_examples.py --gpus 4 --batch_size 4

# 查看 2 GPU 配置
python3 training_examples.py --gpus 2 --batch_size 6
```

## 监控训练

### TensorBoard

```bash
tensorboard --logdir vessel_segmentation/logs/my_experiment
```

在浏览器打开 http://localhost:6006 查看：
- Loss 曲线
- Dice Score
- IoU
- Accuracy
- Learning Rate

### 检查点位置

```
vessel_segmentation/checkpoints/my_experiment/
├── best_model.pth          # 最佳模型
├── final_model.pth         # 最终模型
└── model_epoch_*.pth       # 定期检查点
```

## 恢复训练

从检查点恢复：

```bash
python3 -m vessel_segmentation.train_distributed \
  --gpus 4 \
  --cfg vitb16_frozen \
  --resume vessel_segmentation/checkpoints/my_experiment/model_epoch_50.pth \
  --flag my_experiment
```

## 预期结果

### 模型参数

- **总参数**: ~86M
- **冻结参数**: ~85M (编码器)
- **可训练参数**: ~1M (解码器 + 头)

### 显存占用（每GPU）

- Batch size 4: ~6 GB
- Batch size 6: ~8 GB
- 推荐: RTX 3060 (12GB)

### 训练时间估算

- 4 GPU × Batch 4 = 有效批次 16
- 每 epoch: ~5-10分钟（取决于数据集大小）
- 100 epochs: ~8-16小时

## 常见问题

### Q: 找不到预训练权重文件

**A**: 确保 `VFM_Fundus_weights.pth` 在 SETR-main 目录下：
```bash
ls -lh SETR-main/VFM_Fundus_weights.pth
```

如果没有，请运行：
```bash
bash setup_weights.sh /path/to/weights.pth
```

### Q: CUDA Out of Memory

**A**: 减小批次大小：
```bash
--batch_size 2  # 或更小
```

### Q: 分布式训练端口冲突

**A**: 指定不同端口：
```bash
--master_port 12357
```

### Q: 想使用所有 GPU

**A**: 不指定 --gpus 参数（默认会使用所有可用GPU）或：
```bash
--gpus $(nvidia-smi -L | wc -l)
```

### Q: 训练太慢

**A**: 检查：
1. 使用了足够大的批次
2. 数据加载器工作进程数（config.py 中的 num_workers）
3. 启用了 pin_memory

## 性能优化建议

### 1. 批次大小调优

| GPU配置 | 建议batch_size | 总批次 |
|---------|----------------|--------|
| 2 × RTX 3060 | 6 | 12 |
| 4 × RTX 3060 | 4 | 16 |
| 8 × RTX 3060 | 2-3 | 16-24 |

### 2. 数据加载优化

在 `config.py` 中设置：
```python
num_workers = 4  # 根据CPU核心数调整
```

### 3. 学习率调整

如果总批次大小改变，可能需要调整学习率：
```python
# config.py
learning_rate = 1e-4 * (total_batch_size / 16)
```

## 下一步

1. **评估模型**: 参见 `evaluate.py`
2. **预测**: 参见 `predict.py`
3. **可视化**: 参见 `visualize.py`

详细文档：
- [完整中文训练指南](TRAINING_GUIDE_CN.md)
- [Complete English Guide](TRAINING_GUIDE.md)

## 技术支持

遇到问题？
1. 运行 `python3 test_vitb16_config.py` 检查配置
2. 查看 [TRAINING_GUIDE_CN.md](TRAINING_GUIDE_CN.md) 完整文档
3. 检查 TensorBoard 日志
4. 查看训练输出的错误信息

---

**祝训练顺利！** 🚀
