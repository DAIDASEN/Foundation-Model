import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from vessel_segmentation.model import VesselSETR
from vessel_segmentation.config import Config


class PredictionDataset(Dataset):
    """预测数据集"""
    def __init__(self, image_dir, image_size=512):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        # 获取所有图片文件
        self.image_files = sorted([
            f for f in self.image_dir.glob('*')
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"在 {image_dir} 中未找到图片文件")
        
        print(f"找到 {len(self.image_files)} 张待预测图片")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # 读取图片
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # 调整大小用于模型推理
        image_resized = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # 转换为tensor并归一化（与训练时保持一致）
        image_tensor = torch.from_numpy(np.array(image_resized)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # [C, H, W]
        
        # 使用与训练时相同的归一化
        from torchvision import transforms as T
        normalize = T.Normalize(mean=[0.33, 0.15, 0.06], std=[0.24, 0.12, 0.05])
        image_tensor = normalize(image_tensor)
        
        return {
            'image': image_tensor,
            'original_size': original_size,
            'filename': img_path.name
        }


def predict(
    image_dir,
    output_dir,
    checkpoint_path,
    image_size=512,
    batch_size=8,
    device='cuda',
    resize_to_original=True,
    save_probability=False
):
    """
    对图片进行批量预测
    
    Args:
        image_dir: 输入图片目录
        output_dir: 输出目录
        checkpoint_path: 模型checkpoint路径
        image_size: 模型输入尺寸
        batch_size: 批次大小
        device: 设备
        resize_to_original: 是否将预测结果调整回原始尺寸
        save_probability: 是否保存概率图
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mask_dir = output_dir / 'masks'
    mask_dir.mkdir(exist_ok=True)
    
    if save_probability:
        prob_dir = output_dir / 'probabilities'
        prob_dir.mkdir(exist_ok=True)
    
    # 加载数据集
    dataset = PredictionDataset(image_dir, image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    config = Config()
    model = VesselSETR(
        img_size=config.input_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        embed_dim=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads
    )
    
    # 加载权重
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint info:")
        if 'epoch' in checkpoint:
            print(f"  - Epoch: {checkpoint['epoch']}")
        if 'val_dice' in checkpoint:
            print(f"  - Val Dice: {checkpoint['val_dice']:.4f}")
        if 'val_iou' in checkpoint:
            print(f"  - Val IoU: {checkpoint['val_iou']:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"\n开始预测...")
    print(f"输入目录: {image_dir}")
    print(f"输出目录: {output_dir}")
    print(f"批次大小: {batch_size}")
    print(f"设备: {device}")
    print(f"调整回原始尺寸: {resize_to_original}")
    print(f"=" * 60)
    
    total_images = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测中"):
            images = batch['image'].to(device)
            original_sizes = batch['original_size']
            filenames = batch['filename']
            
            # 预测
            outputs = model(images)
            
            # 获取预测类别
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # 获取概率（softmax）
            if save_probability:
                probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            # 处理每张图片
            for i in range(len(filenames)):
                pred_mask = predictions[i]
                filename = filenames[i]
                original_size = (original_sizes[0][i].item(), original_sizes[1][i].item())
                
                # 调整mask尺寸
                if resize_to_original:
                    mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize(original_size, Image.NEAREST)
                    pred_mask_final = (np.array(mask_pil) > 127).astype(np.uint8)
                else:
                    pred_mask_final = pred_mask.astype(np.uint8)
                
                # 保存mask (0或255)
                mask_path = mask_dir / filename.replace('.jpg', '.png')
                Image.fromarray((pred_mask_final * 255).astype(np.uint8)).save(mask_path)
                
                # 保存概率图
                if save_probability:
                    prob_map = probabilities[i]
                    if resize_to_original:
                        prob_pil = Image.fromarray((prob_map * 255).astype(np.uint8))
                        prob_pil = prob_pil.resize(original_size, Image.BILINEAR)
                        prob_map = np.array(prob_pil).astype(np.float32) / 255.0
                    
                    prob_path = prob_dir / filename.replace('.jpg', '.png')
                    Image.fromarray((prob_map * 255).astype(np.uint8)).save(prob_path)
                
                total_images += 1
    
    print(f"\n✅ 预测完成！")
    print(f"   处理图片数: {total_images}")
    print(f"   预测mask保存在: {mask_dir}")
    if save_probability:
        print(f"   概率图保存在: {prob_dir}")
    
    # 生成统计信息
    print(f"\n📊 统计信息:")
    vessel_pixels = []
    for mask_file in mask_dir.glob('*.png'):
        mask = np.array(Image.open(mask_file))
        vessel_ratio = (mask > 127).sum() / mask.size
        vessel_pixels.append(vessel_ratio)
    
    if vessel_pixels:
        print(f"   平均血管占比: {np.mean(vessel_pixels)*100:.2f}%")
        print(f"   最小血管占比: {np.min(vessel_pixels)*100:.2f}%")
        print(f"   最大血管占比: {np.max(vessel_pixels)*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='血管分割预测')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='输入图片目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--image_size', type=int, default=512,
                        help='模型输入尺寸')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--no_resize', action='store_true',
                        help='不将预测结果调整回原始尺寸')
    parser.add_argument('--save_probability', action='store_true',
                        help='保存概率图')
    parser.add_argument('--flag', type=str, default=None,
                        help='实验标识flag, 若提供则在输出目录下创建该子目录用于区分保存')
    
    args = parser.parse_args()
    
    # 处理flag子目录
    resolved_output_dir = args.output_dir
    if args.flag:
        resolved_output_dir = str(Path(args.output_dir) / args.flag)

    predict(
        image_dir=args.image_dir,
        output_dir=resolved_output_dir,
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        resize_to_original=not args.no_resize,
        save_probability=args.save_probability
    )


if __name__ == '__main__':
    main()
