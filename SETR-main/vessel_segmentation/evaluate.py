"""
评估脚本：计算测试集上的性能指标
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import json


def calculate_metrics(pred_mask, gt_mask):
    """
    计算单张图片的指标
    
    Args:
        pred_mask: 预测mask (0或1)
        gt_mask: 真实mask (0或1)
    
    Returns:
        dict: 包含各项指标的字典
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    # 计算交并比
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # IoU
    iou = intersection / union if union > 0 else 0.0
    
    # Dice系数
    dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0
    
    # 准确率
    accuracy = (pred_mask == gt_mask).sum() / gt_mask.size
    
    # 精确率和召回率
    tp = intersection
    fp = (pred_mask & ~gt_mask).sum()
    fn = (gt_mask & ~pred_mask).sum()
    tn = (~pred_mask & ~gt_mask).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate(pred_dir, gt_dir, output_file=None):
    """
    评估预测结果
    
    Args:
        pred_dir: 预测mask目录
        gt_dir: 真实mask目录
        output_file: 输出文件路径（JSON格式）
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    
    # 获取所有预测文件
    pred_files = sorted(list(pred_dir.glob('*.png')))
    
    if len(pred_files) == 0:
        print(f"❌ 在 {pred_dir} 中未找到预测mask")
        return
    
    print(f"找到 {len(pred_files)} 个预测mask")
    
    # 计算每张图片的指标
    all_metrics = []
    missing_gt = []
    
    for pred_file in tqdm(pred_files, desc="评估中"):
        # 查找对应的GT文件
        gt_file = gt_dir / pred_file.name
        
        if not gt_file.exists():
            missing_gt.append(pred_file.name)
            continue
        
        # 读取mask（预测mask是0/255，GT mask可能是0/1或0/255）
        pred_mask = np.array(Image.open(pred_file).convert('L')) > 127
        gt_array = np.array(Image.open(gt_file).convert('L'))
        # 自动检测GT的值范围来确定阈值
        gt_max = gt_array.max()
        gt_mask = gt_array > (gt_max / 2.0)
        
        # 调整尺寸（如果不匹配）
        if pred_mask.shape != gt_mask.shape:
            from PIL import Image as PILImage
            pred_pil = PILImage.fromarray((pred_mask * 255).astype(np.uint8))
            pred_pil = pred_pil.resize((gt_mask.shape[1], gt_mask.shape[0]), PILImage.NEAREST)
            pred_mask = np.array(pred_pil) > 127
        
        # 计算指标
        metrics = calculate_metrics(pred_mask, gt_mask)
        metrics['filename'] = pred_file.name
        all_metrics.append(metrics)
    
    if missing_gt:
        print(f"\n⚠️  警告: {len(missing_gt)} 个文件未找到对应的GT")
    
    if len(all_metrics) == 0:
        print("❌ 没有可评估的图片")
        return
    
    # 计算平均指标
    avg_metrics = {
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics])
    }
    
    # 计算标准差
    std_metrics = {
        'iou_std': np.std([m['iou'] for m in all_metrics]),
        'dice_std': np.std([m['dice'] for m in all_metrics]),
        'accuracy_std': np.std([m['accuracy'] for m in all_metrics]),
        'precision_std': np.std([m['precision'] for m in all_metrics]),
        'recall_std': np.std([m['recall'] for m in all_metrics]),
        'f1_std': np.std([m['f1'] for m in all_metrics])
    }
    
    # 打印结果
    print("\n" + "=" * 60)
    print("📊 评估结果")
    print("=" * 60)
    print(f"评估图片数: {len(all_metrics)}")
    print("-" * 60)
    print(f"IoU (Jaccard):  {avg_metrics['iou']:.4f} ± {std_metrics['iou_std']:.4f}")
    print(f"Dice系数:       {avg_metrics['dice']:.4f} ± {std_metrics['dice_std']:.4f}")
    print(f"准确率:         {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy_std']:.4f}")
    print(f"精确率:         {avg_metrics['precision']:.4f} ± {std_metrics['precision_std']:.4f}")
    print(f"召回率:         {avg_metrics['recall']:.4f} ± {std_metrics['recall_std']:.4f}")
    print(f"F1分数:         {avg_metrics['f1']:.4f} ± {std_metrics['f1_std']:.4f}")
    print("=" * 60)
    
    # 找出最好和最差的样本
    sorted_by_dice = sorted(all_metrics, key=lambda x: x['dice'], reverse=True)
    print("\n🏆 Dice系数最高的5个样本:")
    for i, m in enumerate(sorted_by_dice[:5], 1):
        print(f"  {i}. {m['filename']}: {m['dice']:.4f}")
    
    print("\n⚠️  Dice系数最低的5个样本:")
    for i, m in enumerate(sorted_by_dice[-5:], 1):
        print(f"  {i}. {m['filename']}: {m['dice']:.4f}")
    
    # 保存结果到JSON
    if output_file:
        output_file = Path(output_file)
        results = {
            'summary': {
                'num_images': len(all_metrics),
                'average': avg_metrics,
                'std': std_metrics
            },
            'per_image': all_metrics
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ 详细结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='评估血管分割结果')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='预测mask目录')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='真实mask目录')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='输出文件路径')
    parser.add_argument('--flag', type=str, default=None,
                        help='实验标识flag, 若提供则在输出目录下创建该子目录用于区分保存')
    
    args = parser.parse_args()
    
    # 处理flag: 若提供，将输出文件放到指定父目录下的flag子目录
    resolved_output = args.output
    if args.flag:
        out_path = Path(args.output)
        parent = out_path.parent if out_path.parent.name != '' else Path('.')
        resolved_output = str(parent / args.flag / out_path.name)
        Path(parent / args.flag).mkdir(parents=True, exist_ok=True)

    evaluate(args.pred_dir, args.gt_dir, resolved_output)


if __name__ == '__main__':
    main()
