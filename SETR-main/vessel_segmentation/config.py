# -*- coding: utf-8 -*-

class Config:
    # 数据集配置
    data_root = 'data/FIVES_vessel'
    train_image_dir = 'images/train'
    train_mask_dir = 'annotations/train'
    test_image_dir = 'images/test'
    test_mask_dir = 'annotations/test'
    
    # 模型配置
    num_classes = 2
    input_size = 512 
    patch_size = 32
    hidden_size = 768  #(input/patch)^2*3
    num_layers = 6
    num_heads = 12
    freeze_transformer = True
    pretrained_transformer_path = 'VFM_Fundus_weights.pth'
    
    # 训练配置
    batch_size = 6 
    num_epochs = 100  
    learning_rate = 1e-4
    weight_decay = 1e-4
    train_val_split = 0.9
    
    # 优化器配置
    optimizer = 'adam'
    lr_scheduler = 'cosine'
    
    # 保存配置
    save_dir = 'vessel_segmentation/checkpoints'
    log_dir = 'vessel_segmentation/logs'
    save_interval = 5
    
    # 其他配置
    num_workers = 4
    device = 'cuda'
    seed = 42


class ConfigViTB16Frozen(Config):
    """
    与 VFM_Fundus_weights.pth 对齐的配置：
    - ViT-B/16 骨干（patch_size=16, hidden=768, heads=12, 推荐 12 层）
    - 冻结 Transformer，仅训练 MLA/解码器/头
    - 其余数据与优化参数沿用父类默认
    注意：使用该配置时建议切换保存目录，避免覆盖其他实验。
    """
    patch_size = 16
    num_layers = 12
    freeze_transformer = True
    # 可选：为区分实验，可设置默认子目录（也可通过 --flag 覆盖）
    save_dir = 'vessel_segmentation/checkpoints/vitb16_frozen'
    log_dir = 'vessel_segmentation/logs/vitb16_frozen'
