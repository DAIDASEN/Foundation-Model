# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import argparse

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from vessel_segmentation.config import Config, ConfigViTB16Frozen
from vessel_segmentation.dataset import VesselDataset
from vessel_segmentation.model import get_model
from vessel_segmentation.utils import set_seed, save_checkpoint, DiceLoss, compute_metrics


def setup_distributed(rank, world_size):
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


def get_dataloader_distributed(config, rank, world_size):
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    
    # 加载数据集
    train_dataset = VesselDataset(
        image_dir=os.path.join(config.data_root, config.train_image_dir),
        mask_dir=os.path.join(config.data_root, config.train_mask_dir),
        input_size=config.input_size,
        is_train=True
    )
    
    # 划分训练集和验证集
    total_size = len(train_dataset)
    train_size = int(config.train_val_split * total_size)
    val_size = total_size - train_size
    
    indices = list(range(total_size))
    train_indices, val_indices = train_test_split(
        indices, train_size=train_size, random_state=config.seed
    )
    
    # 创建训练集
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    train_sampler = DistributedSampler(
        train_subset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_dataset = VesselDataset(
        image_dir=os.path.join(config.data_root, config.train_image_dir),
        mask_dir=os.path.join(config.data_root, config.train_mask_dir),
        input_size=config.input_size,
        is_train=False
    )
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    val_sampler = DistributedSampler(
        val_subset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, rank):
    model.train()
    total_loss = 0
    total_dice = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    else:
        pbar = dataloader
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=1)
        dice = compute_metrics(preds, masks)['dice']
        total_dice += dice
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    return avg_loss, avg_dice


def validate(model, dataloader, criterion, device, rank):
    """验证"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_acc = 0
    
    # 只在主进程显示进度条
    if rank == 0:
        pbar = tqdm(dataloader, desc='Validating')
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 计算指标
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            metrics = compute_metrics(preds, masks)
            total_dice += metrics['dice']
            total_iou += metrics['iou']
            total_acc += metrics['accuracy']
            
            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{metrics["dice"]:.4f}'
                })
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    
    return avg_loss, avg_dice, avg_iou, avg_acc


def train_distributed(rank, world_size, config, resume_path=None):
    setup_distributed(rank, world_size)
    
    set_seed(config.seed + rank)
    
    device = torch.device(f'cuda:{rank}')
    
    writer = None
    if rank == 0:
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        writer = SummaryWriter(config.log_dir)
        print(f"开始分布式训练，使用 {world_size} 个GPU")
        print(f"总batch size = {config.batch_size} x {world_size} = {config.batch_size * world_size}")
    
    model = get_model(config).to(device)
    
    # 在某些结构或分支未参与当前迭代的loss计算时，启用 unused 参数检测，避免 "Expected to have finished reduction" 错误
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")
    
    # 创建数据加载器
    train_loader, val_loader, train_sampler = get_dataloader_distributed(config, rank, world_size)
    
    if rank == 0:
        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"验证样本: {len(val_loader.dataset)}")
        print(f"每GPU批次数: {len(train_loader)}")
    
    # 定义损失函数
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    
    def criterion(outputs, targets):
        return ce_loss(outputs, targets) + dice_loss(outputs, targets)
    
    # 定义优化器
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer == 'adam':
        optimizer = optim.Adam(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = optim.SGD(
            trainable_params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    
    # 定义学习率调度器
    if config.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    
    # 从checkpoint恢复训练
    start_epoch = 0
    best_dice = 0.0
    
    if resume_path and os.path.exists(resume_path):
        if rank == 0:
            print(f"\n从checkpoint恢复训练: {resume_path}")
        
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(resume_path, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_dice = checkpoint.get('val_dice', 0.0)
        
        if rank == 0:
            print(f"  恢复自epoch {start_epoch}, 最佳Dice: {best_dice:.4f}")
            print(f"  将继续训练到epoch {config.num_epochs}\n")
    
    # 训练循环
    for epoch in range(start_epoch, config.num_epochs):
        # 设置sampler的epoch（保证每个epoch打乱顺序不同）
        train_sampler.set_epoch(epoch)
        
        # 训练
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, rank
        )
        
        # 验证
        val_loss, val_dice, val_iou, val_acc = validate(
            model, val_loader, criterion, device, rank
        )
        
        # 更新学习率
        scheduler.step()
        
        # 只在主进程记录日志和保存模型
        if rank == 0:
            # 记录到TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Dice/train', train_dice, epoch)
            writer.add_scalar('Dice/val', val_dice, epoch)
            writer.add_scalar('IoU/val', val_iou, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
            
            # 打印信息
            print(f'\nEpoch {epoch+1}/{config.num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, '
                  f'Val IoU: {val_iou:.4f}, Val Acc: {val_acc:.4f}')
            print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_dice > best_dice:
                best_dice = val_dice
                save_checkpoint(
                    model.module,  # 保存DDP模型的原始模型
                    optimizer,
                    epoch,
                    val_dice,
                    os.path.join(config.save_dir, 'best_model.pth'),
                    scheduler=scheduler,
                    val_iou=val_iou,
                    val_acc=val_acc
                )
                print(f'✓ 保存最佳模型 (Dice: {val_dice:.4f})')
            
            # 定期保存
            if (epoch + 1) % config.save_interval == 0:
                save_checkpoint(
                    model.module,
                    optimizer,
                    epoch,
                    val_dice,
                    os.path.join(config.save_dir, f'model_epoch_{epoch+1}.pth'),
                    scheduler=scheduler
                )
    
    # 保存最终模型
    if rank == 0:
        save_checkpoint(
            model.module,
            optimizer,
            config.num_epochs - 1,
            val_dice,
            os.path.join(config.save_dir, 'final_model.pth'),
            scheduler=scheduler
        )
        print('\n训练完成！')
        writer.close()
    
    # 清理
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--flag', type=str, default=None)
    parser.add_argument('--cfg', type=str, default='default', choices=['default', 'vitb16_frozen'],
                        help='选择配置：default=当前Config，vitb16_frozen=与VFM权重对齐并冻结Transformer')
    parser.add_argument('--master_port', type=int, default=None,
                        help='分布式MASTER_PORT，未指定则自动选择空闲端口')
    args = parser.parse_args()

    # 选择配置
    if args.cfg == 'vitb16_frozen':
        config = ConfigViTB16Frozen()
    else:
        config = Config()

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.epochs is not None:
        config.num_epochs = args.epochs

    if args.flag is not None:
        config.save_dir = os.path.join('vessel_segmentation/checkpoints', args.flag)
        config.log_dir = os.path.join('vessel_segmentation/logs', args.flag)
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    world_size = args.gpus

    if args.resume:
        print(f"Reused from: {args.resume}")

    # 设置分布式主机与端口：优先使用用户传入的 --master_port；否则自动选择空闲端口
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if args.master_port is not None:
        os.environ['MASTER_PORT'] = str(args.master_port)
    elif 'MASTER_PORT' not in os.environ:
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                port = s.getsockname()[1]
            os.environ['MASTER_PORT'] = str(port)
        except Exception:
            # 回退默认端口
            os.environ['MASTER_PORT'] = '12355'

    torch.multiprocessing.spawn(
        train_distributed,
        args=(world_size, config, args.resume),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
