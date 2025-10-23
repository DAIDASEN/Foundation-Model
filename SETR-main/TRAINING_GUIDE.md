# SETR MLA Distributed Training Guide (ViT-Base/16 Frozen Encoder)

## Architecture Overview

This project implements the SETR (SEgmentation TRansformer) model with Multi-Level Aggregation (MLA) decoder for semantic segmentation.

### ViT-Base/16 Encoder Specifications
Based on the pretrained weights architecture `VFM_Fundus_weights.pth`:
- **Model**: ViT-Base
- **Patch size**: 16x16
- **Hidden dimension**: 768
- **Transformer layers**: 12
- **Attention heads**: 12
- **MLP dimension**: 3072 (4x hidden_dim)
- **Position embeddings**: 197 (14x14 patches + 1 cls token)
- **Input resolution**: 224x224

## Requirements

- Python 3.7+
- PyTorch (with CUDA 11.8 support)
- 4 x NVIDIA GPU (e.g., RTX 3060)
- Dependencies: see `requirements.txt`

## Installation

```bash
cd SETR-main
pip install -r requirements.txt
```

## Prepare Pretrained Weights

Place the ViT-Base/16 pretrained weights file in the SETR-main directory, named:
```
SETR-main/VFM_Fundus_weights.pth
```

The weights file should contain parameters for:
- Patch embedding (patch_embed.*)
- Position embeddings (pos_embed)
- Transformer blocks (blocks.*.*)
- Layer normalization (norm.*)

## Training Configuration

The project provides two configuration classes:

### 1. ConfigViTB16Frozen (Recommended)
Designed specifically for ViT-Base/16 frozen training:
- Input size: 224x224
- Patch size: 16
- Hidden size: 768
- Transformer layers: 12
- Attention heads: 12
- **Freeze Transformer encoder**
- Train only MLA decoder and segmentation head

### 2. Config (Default)
Original configuration for smaller models or training from scratch

## Distributed Training

### Method 1: Use Provided Script (Recommended)

```bash
# 4 GPU training with ViT-Base/16 frozen config
bash train_vitb16_4gpus.sh exp_vitb16_frozen
```

### Method 2: Direct Python Command

```bash
# Basic training command
python3 -m vessel_segmentation.train_distributed \
  --gpus 4 \
  --cfg vitb16_frozen \
  --batch_size 4 \
  --epochs 100 \
  --flag exp_vitb16_frozen

# Resume from checkpoint
python3 -m vessel_segmentation.train_distributed \
  --gpus 4 \
  --cfg vitb16_frozen \
  --batch_size 4 \
  --resume vessel_segmentation/checkpoints/exp_vitb16_frozen/model_epoch_50.pth \
  --flag exp_vitb16_frozen
```

### Parameter Explanation

- `--gpus`: Number of GPUs (default: 6, recommended: 4)
- `--cfg`: Configuration choice (`default` or `vitb16_frozen`)
- `--batch_size`: Batch size per GPU (total batch = batch_size × gpus)
- `--epochs`: Number of training epochs
- `--flag`: Experiment name for distinguishing different runs
- `--resume`: Checkpoint path for resuming training
- `--master_port`: Master port for distributed training (optional, auto-selected by default)

## Training Monitoring

### TensorBoard

```bash
tensorboard --logdir vessel_segmentation/logs/exp_vitb16_frozen
```

Monitored metrics:
- Training/Validation Loss
- Dice Score
- IoU (Intersection over Union)
- Accuracy
- Learning Rate

## Output Files

Training generates the following files:

```
vessel_segmentation/
├── checkpoints/
│   └── exp_vitb16_frozen/
│       ├── best_model.pth          # Best model (based on validation Dice)
│       ├── final_model.pth         # Final model
│       └── model_epoch_*.pth       # Periodic checkpoints
└── logs/
    └── exp_vitb16_frozen/
        └── events.out.tfevents.*   # TensorBoard logs
```

## Model Architecture Details

### Encoder (Frozen)
- ViT-Base/16 Transformer
- Loaded from pretrained weights
- Parameters not updated during training (requires_grad=False)

### Decoder (Trainable)
- Multi-Level Aggregation (MLA)
  - Extracts features from multiple Transformer layers
  - 1×1 convolution projection to unified channels
  - Feature fusion
- Upsampling decoder
  - Dynamically determines upsampling steps based on patch_size
  - Conv + BatchNorm + ReLU + Upsample
- Segmentation head
  - 1×1 convolution for class prediction

## Performance Optimization Tips

### Batch Size Tuning
- For RTX 3060 (12GB VRAM), recommended batch_size=4-6 per GPU
- Total effective batch size = batch_size × num_gpus
- Example: 4 GPUs × batch_size 4 = effective batch 16

### Learning Rate
- Default learning rate: 1e-4
- Uses Cosine Annealing scheduler
- Can be adjusted in `config.py`

### Data Augmentation
- Training set: Random flip, rotation, scaling
- Validation set: Resize and normalization only

## Troubleshooting

### Q1: CUDA Version Mismatch
Ensure PyTorch's CUDA version is compatible with your system CUDA:
```bash
python3 -c "import torch; print(torch.version.cuda)"
```

### Q2: Out of Memory
- Reduce `--batch_size`
- Reduce input size (modify `input_size` in config.py)
- Use gradient accumulation

### Q3: Pretrained Weights Loading Failed
- Check if weights file path is correct
- Confirm weights architecture matches configuration
- Review loading information in logs

### Q4: Distributed Training Port Conflict
Specify a different port using `--master_port`:
```bash
--master_port 12357
```

## Testing

Before training, verify the configuration:
```bash
python3 test_vitb16_config.py
```

This will test:
- Configuration parameters
- Model creation and architecture
- Forward pass
- Distributed training setup

## Citation

If you use this code, please cite the SETR paper:
```
@inproceedings{zheng2021rethinking,
  title={Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers},
  author={Zheng, Sixiao and Lu, Jiachen and Zhao, Hengshuang and Zhu, Xiatian and Luo, Zekun and Wang, Yabiao and Fu, Yanwei and Feng, Jiaya and Xiang, Tao and Torr, Philip HS and others},
  booktitle={CVPR},
  year={2021}
}
```
