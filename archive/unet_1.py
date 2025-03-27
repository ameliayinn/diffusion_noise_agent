# unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from archive.reparam_moe_2 import ReparamGaussianMoE

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer("dummy_param", torch.tensor(0.0))
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        dtype = self.dummy_param.dtype
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class DownBlock(nn.Module):
    """下采样块（带时间嵌入和自适应归一化）"""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1, stride=2)
        self.time_emb_proj = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch * 2),
            nn.GELU()
        )
        self.act = nn.GELU()

    def forward(self, x, t_emb):
        h = self.conv(x)
        scale, shift = self.time_emb_proj(t_emb).chunk(2, dim=1)
        h = h * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        return self.act(h) + self.res_conv(x)

class UpBlock(nn.Module):
    """上采样块（带跳跃连接和时间嵌入）"""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
        )
        self.time_emb_proj = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch * 2),
            nn.GELU()
        )
        self.act = nn.GELU()

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        h = self.conv(x)
        scale, shift = self.time_emb_proj(t_emb).chunk(2, dim=1)
        h = h * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        return self.act(h)

class UNetBase(nn.Module):
    """UNet基础类（共享结构）"""
    def __init__(self, in_channels, out_channels, time_emb_dim=128, image_size=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(image_size),
            nn.Linear(image_size, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 下采样路径
        self.conv1 = nn.Conv2d(in_channels, image_size, 3, padding=1)
        self.down1 = DownBlock(image_size, image_size*2, time_emb_dim)
        self.down2 = DownBlock(image_size*2, image_size*4, time_emb_dim)
        self.down3 = DownBlock(image_size*4, image_size*8, time_emb_dim)
        
        # 中间层
        self.mid_conv1 = nn.Conv2d(image_size*8, image_size*8, 3, padding=1)
        self.mid_norm = nn.GroupNorm(8, image_size*8)
        self.mid_time_proj = nn.Linear(time_emb_dim, image_size*8*2)
        self.mid_act = nn.GELU()
        
        # 上采样路径
        self.up1 = UpBlock(image_size*8, image_size*4, time_emb_dim)
        self.up2 = UpBlock(image_size*4, image_size*2, time_emb_dim)
        self.up3 = UpBlock(image_size*2, image_size, time_emb_dim)
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(image_size, image_size, 3, padding=1),
            nn.GroupNorm(8, image_size),
            nn.GELU(),
            nn.Conv2d(image_size, out_channels, 1),
        )

    def forward(self, x, t):
        # 确保输入数据类型与模型参数一致
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        t = t.to(dtype)
        
        # 时间嵌入
        t_embed = self.time_embed(t)
        
        # 下采样
        x1 = F.gelu(self.conv1(x))
        x2 = self.down1(x1, t_embed)
        x3 = self.down2(x2, t_embed)
        x = self.down3(x3, t_embed)
        
        # 中间处理
        x = self.mid_conv1(x)
        scale, shift = self.mid_time_proj(t_embed).chunk(2, dim=1)
        x = x * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        x = self.mid_norm(x)
        x = self.mid_act(x)
        
        # 上采样
        x = self.up1(x, x3, t_embed)
        x = self.up2(x, x2, t_embed)
        x = self.up3(x, x1, t_embed)
        
        return self.final_conv(x)

class UNetWithMoE(nn.Module):
    """集成MoE的UNet"""
    def __init__(self, in_channels, out_channels, time_emb_dim=128, image_size=64, 
                 num_experts=4, moe_hidden_dim=64):
        super().__init__()
        self.unet = UNetBase(in_channels, out_channels, time_emb_dim, image_size)
        self.moe = ReparamGaussianMoE(
            input_dim=image_size**2,
            num_experts=num_experts,
            hidden_dim=moe_hidden_dim,
            tau=0.1,
            flatten=True
        )
        
    def forward(self, x, t):
        # UNet预测
        base_pred = self.unet(x, t)
        # MoE细化
        B, C, H, W = base_pred.shape
        flat_pred = base_pred.view(B, -1)
        moe_output = self.moe(flat_pred)
        return moe_output.view(B, C, H, W)

class UNet(nn.Module):
    """标准UNet（RGB图像）"""
    def __init__(self, time_emb_dim=128, image_size=64):
        super().__init__()
        self.model = UNetBase(3, 3, time_emb_dim, image_size)
    
    def forward(self, x, t):
        return self.model(x, t)

class UNetSimulation(nn.Module):
    """模拟数据UNet（单通道）"""
    def __init__(self, time_emb_dim=128, image_size=64, use_moe=False, 
                 num_experts=4, moe_hidden_dim=64):
        super().__init__()
        if use_moe:
            self.model = UNetWithMoE(1, 1, time_emb_dim, image_size, 
                                   num_experts, moe_hidden_dim)
        else:
            self.model = UNetBase(1, 1, time_emb_dim, image_size)
    
    def forward(self, x, t):
        return self.model(x, t)