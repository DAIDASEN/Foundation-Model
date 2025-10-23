# SETR Vessel Segmentation

基于 SETR (Segmentation Transformer) 的血管分割项目，支持分布式训练。

## 🚀 快速开始

### 环境要求
- 4 × NVIDIA GPU (如 RTX 3060)
- CUDA 11.8
- Python 3.7+

### ViT-Base/16 冻结训练 (推荐)

使用预训练的 ViT-Base/16 编码器，仅训练 MLA 解码器：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备预训练权重 (VFM_Fundus_weights.pth)
bash setup_weights.sh /path/to/VFM_Fundus_weights.pth

# 3. 测试配置
python3 test_vitb16_config.py

# 4. 开始训练 (4 GPU)
bash train_vitb16_4gpus.sh my_experiment
```

## 📚 文档

- **[快速上手指南 (中文)](QUICKSTART_CN.md)** - 新手必读
- **[完整训练指南 (中文)](TRAINING_GUIDE_CN.md)** - 详细配置说明
- **[Complete Training Guide (English)](TRAINING_GUIDE.md)** - Full documentation

## 📊 训练监控

```bash
tensorboard --logdir vessel_segmentation/logs/my_experiment
```

## 🔧 配置示例

查看不同 GPU 配置的训练参数：

```bash
python3 training_examples.py --gpus 4 --batch_size 4
```

## 训练

```bash
python3 -m vessel_segmentation.train_distributed \
  --gpus 6 \
  --batch_size 6 \
  --epochs 100 \
  --flag exp1
```
- 输出检查点：`vessel_segmentation/checkpoints/exp1/`
- 输出日志（TensorBoard）：`vessel_segmentation/logs/exp1/`
- 断点续训：
```bash
python3 -m vessel_segmentation.train_distributed \
  --gpus 6 \
  --resume vessel_segmentation/checkpoints/exp1/model_epoch_30.pth \
  --flag exp1
```

## 预测（批量生成掩码）
将预测输出按 flag 分目录保存（masks/ 和 probabilities/）：

```bash
python3 -m vessel_segmentation.predict \
  --image_dir data/FIVES_vessel/images/test \
  --output_dir outputs \
  --checkpoint vessel_segmentation/checkpoints/exp1/best_model.pth \
  --image_size 512 \
  --batch_size 8 \
  --device cuda \
  --save_probability \
  --flag exp1
```
- 输出目录：`outputs/exp1/`
  - 掩码：`outputs/exp1/masks/*.png`
  - 概率图（可选）：`outputs/exp1/probabilities/*.png`

## 可视化（高亮血管叠加）
将可视化输出按 flag 分目录保存（overlay/、comparison/、masks/）：

```bash
python3 -m vessel_segmentation.visualize \
  --image_dir data/FIS/images/test \
  --output_dir viz \
  --checkpoint vessel_segmentation/checkpoints/exp1/best_model.pth \
  --image_size 512 \
  --batch_size 4 \
  --device cuda \
  --color 0,255,0 \
  --alpha 0.6 \
  --flag exp1
```
- 输出目录：`viz/exp1/`
  - 叠加图：`viz/exp1/overlay/*.jpg`
  - 对比图：`viz/exp1/comparison/*.jpg`
  - 掩码（可选）：`viz/exp1/masks/*.png`
