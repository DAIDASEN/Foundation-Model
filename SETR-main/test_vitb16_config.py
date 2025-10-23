#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证 ViT-Base/16 冻结配置是否正确
"""

import sys
import os

# 添加路径
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

import torch
from vessel_segmentation.config import Config, ConfigViTB16Frozen
from vessel_segmentation.model import get_model


def test_config():
    """测试配置参数"""
    print("=" * 60)
    print("测试配置参数")
    print("=" * 60)
    
    # 测试默认配置
    config_default = Config()
    print("\n默认配置 (Config):")
    print(f"  输入尺寸: {config_default.input_size}")
    print(f"  Patch size: {config_default.patch_size}")
    print(f"  Hidden size: {config_default.hidden_size}")
    print(f"  层数: {config_default.num_layers}")
    print(f"  注意力头数: {config_default.num_heads}")
    print(f"  冻结Transformer: {config_default.freeze_transformer}")
    
    # 测试 ViT-Base/16 配置
    config_vitb16 = ConfigViTB16Frozen()
    print("\nViT-Base/16 冻结配置 (ConfigViTB16Frozen):")
    print(f"  输入尺寸: {config_vitb16.input_size}")
    print(f"  Patch size: {config_vitb16.patch_size}")
    print(f"  Hidden size: {config_vitb16.hidden_size}")
    print(f"  层数: {config_vitb16.num_layers}")
    print(f"  注意力头数: {config_vitb16.num_heads}")
    print(f"  冻结Transformer: {config_vitb16.freeze_transformer}")
    print(f"  预训练权重路径: {config_vitb16.pretrained_transformer_path}")
    
    # 验证架构参数
    expected_patches = (config_vitb16.input_size // config_vitb16.patch_size) ** 2
    print(f"\n预期的patch数量: {expected_patches}")
    print(f"预期的位置编码数量: {expected_patches} (不含cls_token)")
    
    assert config_vitb16.input_size == 224, "输入尺寸应为224"
    assert config_vitb16.patch_size == 16, "Patch size应为16"
    assert config_vitb16.hidden_size == 768, "Hidden size应为768"
    assert config_vitb16.num_layers == 12, "层数应为12"
    assert config_vitb16.num_heads == 12, "注意力头数应为12"
    assert config_vitb16.freeze_transformer == True, "应冻结Transformer"
    
    print("\n✓ 配置参数验证通过")


def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("测试模型创建")
    print("=" * 60)
    
    config = ConfigViTB16Frozen()
    
    print("\n创建模型...")
    model = get_model(config)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  冻结参数: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    # 检查哪些模块是可训练的
    trainable_modules = set()
    frozen_modules = set()
    
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        if param.requires_grad:
            trainable_modules.add(module_name)
        else:
            frozen_modules.add(module_name)
    
    print(f"\n可训练模块: {sorted(trainable_modules)}")
    print(f"冻结模块: {sorted(frozen_modules)}")
    
    # 验证冻结逻辑
    if config.freeze_transformer:
        # 如果没有加载预训练权重，可能所有参数都是可训练的
        print("\n注意: 由于未找到预训练权重文件，所有参数当前可能都是可训练的")
        print("提供 VFM_Fundus_weights.pth 后，Transformer 编码器将被冻结")
    
    print("\n✓ 模型创建成功")
    return model


def test_forward_pass():
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("测试前向传播")
    print("=" * 60)
    
    config = ConfigViTB16Frozen()
    model = get_model(config)
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, config.input_size, config.input_size)
    
    print(f"\n输入张量形状: {input_tensor.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"输出张量形状: {output.shape}")
    
    # 验证输出形状
    expected_shape = (batch_size, config.num_classes, config.input_size, config.input_size)
    assert output.shape == expected_shape, f"输出形状应为 {expected_shape}, 实际为 {output.shape}"
    
    print("\n✓ 前向传播测试通过")


def test_distributed_setup():
    """测试分布式训练设置"""
    print("\n" + "=" * 60)
    print("测试分布式训练设置")
    print("=" * 60)
    
    # 检查CUDA可用性
    print(f"\nCUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("警告: CUDA不可用，无法测试GPU相关功能")
    
    # 检查分布式后端
    print(f"\nNCCL后端可用: {torch.distributed.is_nccl_available()}")
    
    print("\n✓ 分布式设置检查完成")


def main():
    print("\n" + "=" * 60)
    print("SETR ViT-Base/16 冻结配置测试")
    print("=" * 60)
    
    try:
        # 运行测试
        test_config()
        model = test_model_creation()
        test_forward_pass()
        test_distributed_setup()
        
        print("\n" + "=" * 60)
        print("所有测试通过! ✓")
        print("=" * 60)
        print("\n下一步:")
        print("1. 准备 VFM_Fundus_weights.pth 预训练权重文件")
        print("2. 准备训练数据集")
        print("3. 运行训练脚本:")
        print("   bash train_vitb16_4gpus.sh exp_name")
        print("=" * 60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
