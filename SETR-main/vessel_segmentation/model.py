# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=512, patch_size=32, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class VesselSETR(nn.Module):    
    def __init__(self, img_size=512, patch_size=32, num_classes=2, 
                 embed_dim=768, num_layers=6, num_heads=12):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.mla_channels = 64
        # 选取用于 MLA 的层索引，并做去重/排序，避免小层数时重复
        self.mla_layers_idx = sorted(set([
            0,
            max(0, num_layers // 2),
            max(0, num_layers - 1)
        ]))
        # 与实际使用的层数一致地创建投影卷积
        self.mla_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, self.mla_channels, kernel_size=1)
            for _ in self.mla_layers_idx
        ])
        # 解码器：根据 patch_size 动态确定上采样步数
        # 起始特征分辨率为 (img_size / patch_size)，目标为 img_size
        # 需要的上采样次数 = log2(patch_size)
        up_steps = int(math.log2(self.patch_size))
        # 生成每一步的输出通道数，使得在 patch_size=32 时等价于原实现
        base_channels = [128, 128, 64, 64, 32]
        if up_steps <= len(base_channels):
            ch_plan = base_channels[:up_steps]
        else:
            ch_plan = base_channels + [32] * (up_steps - len(base_channels))
        dec_layers = []
        in_ch = self.mla_channels
        for out_ch in ch_plan:
            dec_layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ])
            in_ch = out_ch
        self.decoder = nn.Sequential(*dec_layers)
        # 以解码器最后一层的通道数作为 head 的输入通道
        self.head = nn.Conv2d(in_ch, num_classes, 1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        mla_feats = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.mla_layers_idx:
                h = w = self.img_size // self.patch_size
                feat2d = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
                mla_feats.append(feat2d)
        # 注：此处不再对 x 进行额外 LayerNorm，因为下游未使用 x 本身
        proj_feats = []
        for proj, feat in zip(self.mla_projs, mla_feats):
            proj_feats.append(proj(feat))
        fused = proj_feats[0]
        for pf in proj_feats[1:]:
            fused = fused + pf
        x = self.decoder(fused)
        x = self.head(x)
        return x

def get_model(config):
    model = VesselSETR(
        img_size=config.input_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        embed_dim=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads
    )
    if getattr(config, 'freeze_transformer', False):
        import os
        ckpt_path = getattr(config, 'pretrained_transformer_path', None)
        if ckpt_path and not os.path.isabs(ckpt_path) and not os.path.exists(ckpt_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            alt_path = os.path.join(project_root, ckpt_path)
            if os.path.exists(alt_path):
                ckpt_path = alt_path
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                sd = ckpt['model_state_dict']
            else:
                sd = ckpt if isinstance(ckpt, dict) else None
            if isinstance(sd, dict):
                def strip_prefix(key):
                    # 递归去掉已知的多重前缀，直到不再匹配
                    prefixes = ('module.', 'encoder.', 'backbone.', 'transformer.')
                    changed = True
                    while changed:
                        changed = False
                        for pref in prefixes:
                            if key.startswith(pref):
                                key = key[len(pref):]
                                changed = True
                    return key
                sd_norm = {}
                for k, v in sd.items():
                    k2 = strip_prefix(k)
                    k2 = k2.replace('mlp.fc1.', 'ffn.fc1.').replace('mlp.fc2.', 'ffn.fc2.')
                    sd_norm[k2] = v
                sd_merged = dict(sd_norm)
                for i in range(getattr(config, 'num_layers', 0)):
                    base = f'blocks.{i}.attn.'
                    q_w = sd_norm.get(base + 'q.weight', None)
                    k_w = sd_norm.get(base + 'k.weight', None)
                    v_w = sd_norm.get(base + 'v.weight', None)
                    q_b = sd_norm.get(base + 'q.bias', None)
                    k_b = sd_norm.get(base + 'k.bias', None)
                    v_b = sd_norm.get(base + 'v.bias', None)
                    if (q_w is not None) and (k_w is not None) and (v_w is not None):
                        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
                        sd_merged[base + 'qkv.weight'] = qkv_w
                        if (q_b is not None) and (k_b is not None) and (v_b is not None):
                            qkv_b = torch.cat([q_b, k_b, v_b], dim=0)
                            sd_merged[base + 'qkv.bias'] = qkv_b
                model_sd = model.state_dict()
                loadable = {}
                backbone_prefixes = ('patch_embed.', 'pos_embed', 'blocks.', 'norm.')
                for k, v in sd_merged.items():
                    if any(k.startswith(p) for p in backbone_prefixes):
                        if k in model_sd:
                            if k == 'pos_embed' and model_sd[k].shape != v.shape:
                                pe = v
                                if pe.dim() == 2:
                                    pe = pe.unsqueeze(0)
                                N_src = pe.shape[1]
                                C = pe.shape[2]
                                N_tgt = model_sd[k].shape[1]
                                if int(math.sqrt(N_src)) ** 2 != N_src and (N_src - 1) > 0:
                                    pe_token, pe_grid = pe[:, :1, :], pe[:, 1:, :]
                                else:
                                    pe_token, pe_grid = None, pe
                                gh_src = int(math.sqrt(pe_grid.shape[1]))
                                gw_src = gh_src
                                gh_tgt = int(math.sqrt(N_tgt))
                                gw_tgt = gh_tgt
                                pe_grid = pe_grid[0].permute(1, 0).reshape(C, gh_src, gw_src).unsqueeze(0)
                                pe_grid = F.interpolate(pe_grid, size=(gh_tgt, gw_tgt), mode='bilinear', align_corners=False)
                                pe_grid = pe_grid.squeeze(0).reshape(C, gh_tgt * gw_tgt).permute(1, 0).unsqueeze(0)
                                if pe_token is not None:
                                    pe_new = pe_grid
                                else:
                                    pe_new = pe_grid
                                if pe_new.shape == model_sd[k].shape:
                                    loadable[k] = pe_new
                            elif model_sd[k].shape == v.shape:
                                loadable[k] = v
                model.load_state_dict(loadable, strict=False)
                loaded_keys = set(loadable.keys())
        # 仅冻结成功加载的骨干参数，避免冻结随机初始化的模块
        for name, p in model.named_parameters():
            is_backbone = (
                name.startswith('patch_embed') or
                name.startswith('pos_embed') or
                name.startswith('blocks') or
                name.startswith('norm')
            )
            if is_backbone:
                # 与 state_dict 键名对齐：参数名即键名
                if name in locals().get('loaded_keys', set()):
                    p.requires_grad = False
                else:
                    # 未加载成功的骨干参数保持可训练，避免冻结随机权重
                    p.requires_grad = True
            else:
                p.requires_grad = True
    return model
