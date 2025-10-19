# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image


class VesselDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, input_size=512, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.is_train = is_train
        
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        self.normalize = T.Normalize(mean=[0.33, 0.15, 0.06], std=[0.24, 0.12, 0.05])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        返回:
            image: Tensor [3, H, W]
            mask: Tensor [H, W] - 值为0或1
        """
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert('L')
        
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)
        
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        if self.is_train:
            image_np, mask_np = self._augment(image_np, mask_np)
        
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0  # [H,W,3] -> [3,H,W]
        mask_tensor = torch.from_numpy(mask_np).long()  # [H,W]
        
        mask_tensor = (mask_tensor > 0).long()
        
        image_tensor = self.normalize(image_tensor)
        
        return image_tensor, mask_tensor
    
    def _augment(self, image, mask):
        #数据增强
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        
        if random.random() > 0.5:
            k = random.randint(1, 3) 
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        
        if random.random() > 0.7:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        if random.random() > 0.7:
            factor = random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        return image, mask


def get_dataloader(config, is_train=True):
    if is_train:
        image_dir = os.path.join(config.data_root, config.train_image_dir)
        mask_dir = os.path.join(config.data_root, config.train_mask_dir)
        
        dataset = VesselDataset(image_dir, mask_dir, config.input_size, is_train=True)
        
        train_size = int(config.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        
        generator = torch.Generator().manual_seed(config.seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )
        
        val_dataset_no_aug = VesselDataset(image_dir, mask_dir, config.input_size, is_train=False)

        val_indices = val_dataset.indices
        val_dataset_final = torch.utils.data.Subset(val_dataset_no_aug, val_indices)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset_final,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    else:
        image_dir = os.path.join(config.data_root, config.test_image_dir)
        mask_dir = os.path.join(config.data_root, config.test_mask_dir)
        
        dataset = VesselDataset(image_dir, mask_dir, config.input_size, is_train=False)
        
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        return test_loader


def get_train_transform(image_size):
    return None


def get_val_transform(image_size):
    return None
