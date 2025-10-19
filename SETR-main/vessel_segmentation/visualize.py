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


class VisualizationDataset(Dataset):
    """可视化数据集"""
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
        
        print(f"找到 {len(self.image_files)} 张图片")
    
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
        
        # 保存原图用于可视化
        original_image = np.array(image)
        
        return {
            'image': image_tensor,
            'original_image': original_image,
            'original_size': original_size,
            'filename': img_path.name
        }


def overlay_mask_on_image(image, mask, color=(255, 255, 255), alpha=0.5):
    """
    将mask叠加到原图上
    
    Args:
        image: 原图 numpy array [H, W, 3]
        mask: 分割mask numpy array [H, W], 值为0或1
        color: 高亮颜色 (R, G, B)
        alpha: 透明度 0-1
    
    Returns:
        叠加后的图片 numpy array [H, W, 3]
    """
    # 确保image和mask尺寸匹配
    if image.shape[:2] != mask.shape:
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((image.shape[1], image.shape[0]), PILImage.NEAREST)
        mask = np.array(mask_pil) > 127
    
    # 创建彩色mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = color
    
    # 叠加
    output = image.copy()
    overlay_region = mask == 1
    output[overlay_region] = (
        alpha * colored_mask[overlay_region] + 
        (1 - alpha) * image[overlay_region]
    ).astype(np.uint8)
    
    return output


def create_side_by_side(original, overlay, prediction_only=None):
    """
    创建对比图：原图 | 叠加图 | 预测mask
    
    Args:
        original: 原图
        overlay: 叠加图
        prediction_only: 预测mask（可选）
    
    Returns:
        拼接后的图片
    """
    if prediction_only is not None:
        # 三列布局
        result = np.hstack([original, overlay, prediction_only])
    else:
        # 两列布局
        result = np.hstack([original, overlay])
    
    return result


def visualize_predictions(
    image_dir,
    output_dir,
    checkpoint_path,
    annotation_dir=None,
    image_size=512,
    batch_size=4,
    device='cuda',
    overlay_color=(255, 0, 0),
    overlay_alpha=0.5,
    save_individual=True,
    save_comparison=True,
    save_mask_only=False
):
    """
    批量可视化预测结果
    
    Args:
        image_dir: 输入图片目录
        output_dir: 输出目录
        checkpoint_path: 模型checkpoint路径
        annotation_dir: 标注目录（可选，用于对比）
        image_size: 模型输入尺寸
        batch_size: 批次大小
        device: 设备
        overlay_color: 叠加颜色 (R, G, B)
        overlay_alpha: 叠加透明度
        save_individual: 是否保存单独的叠加图
        save_comparison: 是否保存对比图
        save_mask_only: 是否保存纯mask图
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_individual:
        overlay_dir = output_dir / 'overlay'
        overlay_dir.mkdir(exist_ok=True)
    
    if save_comparison:
        comparison_dir = output_dir / 'comparison'
        comparison_dir.mkdir(exist_ok=True)
    
    if save_mask_only:
        mask_dir = output_dir / 'masks'
        mask_dir.mkdir(exist_ok=True)
    
    # 加载数据集
    dataset = VisualizationDataset(image_dir, image_size)
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
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"开始可视化...")
    print(f"输出目录: {output_dir}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="生成可视化"):
            images = batch['image'].to(device)
            original_images = batch['original_image']
            filenames = batch['filename']
            
            # 预测
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # 处理每张图片
            for i in range(len(filenames)):
                original_img = original_images[i].numpy()
                pred_mask = predictions[i]
                filename = filenames[i]
                
                # 调整mask尺寸到原图大小
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((pred_mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize(
                    (original_img.shape[1], original_img.shape[0]),
                    PILImage.NEAREST
                )
                pred_mask_resized = (np.array(mask_pil) > 127).astype(np.uint8)
                
                # 生成叠加图
                overlay_img = overlay_mask_on_image(
                    original_img,
                    pred_mask_resized,
                    color=overlay_color,
                    alpha=overlay_alpha
                )
                
                # 保存单独的叠加图
                if save_individual:
                    overlay_path = overlay_dir / filename
                    PILImage.fromarray(overlay_img).save(overlay_path)
                
                # 保存对比图
                if save_comparison:
                    # 创建mask的彩色版本用于显示
                    mask_colored = np.zeros_like(original_img)
                    mask_colored[pred_mask_resized == 1] = overlay_color
                    
                    comparison_img = create_side_by_side(
                        original_img,
                        overlay_img,
                        mask_colored
                    )
                    comparison_path = comparison_dir / filename
                    PILImage.fromarray(comparison_img).save(comparison_path)
                
                # 保存纯mask
                if save_mask_only:
                    mask_path = mask_dir / filename
                    PILImage.fromarray((pred_mask_resized * 255).astype(np.uint8)).save(mask_path)
    
    print(f"\n✅ 可视化完成！")
    if save_individual:
        print(f"   叠加图保存在: {overlay_dir}")
    if save_comparison:
        print(f"   对比图保存在: {comparison_dir}")
    if save_mask_only:
        print(f"   Mask保存在: {mask_dir}")


def main():
    parser = argparse.ArgumentParser(description='血管分割结果可视化')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='输入图片目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--annotation_dir', type=str, default=None,
                        help='标注目录（可选，用于对比）')
    parser.add_argument('--image_size', type=int, default=512,
                        help='模型输入尺寸')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--color', type=str, default='255,0,0',
                        help='叠加颜色 R,G,B (默认红色)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='叠加透明度 0-1')
    parser.add_argument('--no_individual', action='store_true',
                        help='不保存单独的叠加图')
    parser.add_argument('--no_comparison', action='store_true',
                        help='不保存对比图')
    parser.add_argument('--save_mask', action='store_true',
                        help='保存纯mask图')
    parser.add_argument('--flag', type=str, default=None,
                        help='实验标识flag, 若提供则在输出目录下创建该子目录用于区分保存')
    
    args = parser.parse_args()
    
    # 解析颜色
    color = tuple(map(int, args.color.split(',')))
    
    # 处理flag子目录
    resolved_output_dir = args.output_dir
    if args.flag:
        resolved_output_dir = str(Path(args.output_dir) / args.flag)

    visualize_predictions(
        image_dir=args.image_dir,
        output_dir=resolved_output_dir,
        checkpoint_path=args.checkpoint,
        annotation_dir=args.annotation_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        overlay_color=color,
        overlay_alpha=args.alpha,
        save_individual=not args.no_individual,
        save_comparison=not args.no_comparison,
        save_mask_only=args.save_mask
    )


if __name__ == '__main__':
    main()
