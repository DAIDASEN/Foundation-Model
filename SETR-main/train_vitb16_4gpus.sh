#!/bin/bash
# Training script for SETR with ViT-Base/16 frozen encoder on 4 GPUs
# This script uses the ConfigViTB16Frozen configuration which matches ViT-Base/16 architecture
# 
# Requirements:
# - 4 NVIDIA GPUs (e.g., RTX 3060)
# - CUDA 11.8
# - VFM_Fundus_weights.pth file in SETR-main directory (ViT-Base/16 pretrained weights)
#
# Usage:
#   bash train_vitb16_4gpus.sh [experiment_name]
#
# Example:
#   bash train_vitb16_4gpus.sh exp_vitb16_frozen

EXPERIMENT_NAME=${1:-"vitb16_frozen_4gpu"}

python3 -m vessel_segmentation.train_distributed \
  --gpus 4 \
  --cfg vitb16_frozen \
  --batch_size 4 \
  --epochs 100 \
  --flag "$EXPERIMENT_NAME" \
  --master_port 12356

# Output will be saved to:
# - Checkpoints: vessel_segmentation/checkpoints/$EXPERIMENT_NAME/
# - Logs (TensorBoard): vessel_segmentation/logs/$EXPERIMENT_NAME/
#
# To resume training:
# python3 -m vessel_segmentation.train_distributed \
#   --gpus 4 \
#   --cfg vitb16_frozen \
#   --batch_size 4 \
#   --resume vessel_segmentation/checkpoints/$EXPERIMENT_NAME/model_epoch_XX.pth \
#   --flag "$EXPERIMENT_NAME"
