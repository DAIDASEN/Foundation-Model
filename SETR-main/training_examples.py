#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：演示如何使用不同的配置进行训练
"""

import argparse


def print_training_example(gpus=4, batch_size=4, cfg='vitb16_frozen'):
    """打印训练命令示例"""
    
    total_batch = gpus * batch_size
    
    print("=" * 70)
    print(f"训练配置示例 ({cfg})")
    print("=" * 70)
    print(f"\nGPU数量: {gpus}")
    print(f"每GPU批次大小: {batch_size}")
    print(f"总有效批次大小: {total_batch} (= {gpus} GPUs × {batch_size} batch_size)")
    
    # 根据配置显示模型信息
    if cfg == 'vitb16_frozen':
        print("\n模型配置: ViT-Base/16 冻结编码器")
        print("  - 输入分辨率: 224×224")
        print("  - Patch size: 16×16")
        print("  - Hidden dim: 768")
        print("  - Transformer层数: 12")
        print("  - 注意力头数: 12")
        print("  - 冻结编码器: 是")
        print("  - 训练解码器: 是")
    else:
        print("\n模型配置: 默认配置")
        print("  - 输入分辨率: 512×512")
        print("  - Patch size: 32×32")
        print("  - Hidden dim: 768")
        print("  - Transformer层数: 6")
        print("  - 注意力头数: 12")
        print("  - 冻结编码器: 是")
    
    # 内存估算
    if cfg == 'vitb16_frozen':
        # ViT-Base/16 with 224x224 input
        est_memory_per_sample = 1.5  # GB approx
    else:
        # Default config with 512x512 input
        est_memory_per_sample = 3.0  # GB approx
    
    est_memory = batch_size * est_memory_per_sample
    print(f"\n预估显存需求 (每GPU): ~{est_memory:.1f} GB")
    print(f"推荐GPU: RTX 3060 (12GB) 或更好")
    
    print("\n训练命令:")
    print("-" * 70)
    print(f"python3 -m vessel_segmentation.train_distributed \\")
    print(f"  --gpus {gpus} \\")
    print(f"  --cfg {cfg} \\")
    print(f"  --batch_size {batch_size} \\")
    print(f"  --epochs 100 \\")
    print(f"  --flag exp_{cfg}")
    print("-" * 70)
    
    print("\n或使用快捷脚本:")
    print("-" * 70)
    if cfg == 'vitb16_frozen':
        print(f"bash train_vitb16_4gpus.sh exp_{cfg}")
    else:
        print(f"python3 -m vessel_segmentation.train_distributed --gpus {gpus} --batch_size {batch_size} --flag exp_{cfg}")
    print("-" * 70)
    
    print("\n输出目录:")
    print(f"  - 检查点: vessel_segmentation/checkpoints/exp_{cfg}/")
    print(f"  - 日志: vessel_segmentation/logs/exp_{cfg}/")
    
    print("\n监控训练:")
    print(f"  tensorboard --logdir vessel_segmentation/logs/exp_{cfg}")
    
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(description='显示训练配置示例')
    parser.add_argument('--gpus', type=int, default=4, help='GPU数量')
    parser.add_argument('--batch_size', type=int, default=4, help='每GPU批次大小')
    parser.add_argument('--cfg', type=str, default='vitb16_frozen', 
                        choices=['default', 'vitb16_frozen'],
                        help='配置选择')
    args = parser.parse_args()
    
    print("\n")
    print_training_example(args.gpus, args.batch_size, args.cfg)
    
    # 显示其他配置对比
    if args.cfg == 'vitb16_frozen':
        print("\n提示: 想查看默认配置的示例？运行:")
        print("  python3 training_examples.py --cfg default")
    else:
        print("\n提示: 想查看 ViT-Base/16 冻结配置的示例？运行:")
        print("  python3 training_examples.py --cfg vitb16_frozen")
    
    print("\n不同GPU配置示例:")
    print("  2 GPUs: python3 training_examples.py --gpus 2 --batch_size 6")
    print("  4 GPUs: python3 training_examples.py --gpus 4 --batch_size 4")
    print("  8 GPUs: python3 training_examples.py --gpus 8 --batch_size 2")
    print()


if __name__ == '__main__':
    main()
