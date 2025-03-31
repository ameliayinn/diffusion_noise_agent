# unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from reparam_moe import ReparamGaussianMoE

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 注册一个虚拟参数以获取正确的dtype
        self.register_buffer("dummy_param", torch.tensor(0.0))
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        
        # 通过虚拟参数获取dtype
        dtype = self.dummy_param.dtype
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -embeddings)
        
        time = time.to(dtype)  # 确保时间步类型一致
        embeddings = time[:, None] * embeddings[None, :]
        
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DownBlock(nn.Module):
    """修复后的下采样块，确保残差连接正确下采样"""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        # 主路径：3x3卷积下采样
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
        )
        # 残差路径：1x1卷积下采样
        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, padding=0)
        
        # 时间嵌入处理
        self.time_emb_proj = nn.Linear(time_emb_dim, out_ch * 2)
        self.act = nn.GELU()

    def forward(self, x, t_emb):
        h = self.conv(x)
        
        # 时间嵌入影响
        scale, shift = self.time_emb_proj(t_emb).chunk(2, dim=1)
        h = h * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        h = self.act(h)
        
        # 残差连接（已下采样）
        return h + self.res_conv(x)

class UpBlock(nn.Module):
    """上采样块（带跳跃连接和时间嵌入）"""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
        )
        self.time_emb_proj = nn.Linear(time_emb_dim, out_ch * 2)
        self.act = nn.GELU()

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        h = self.conv(x)
        # 处理时间嵌入
        scale, shift = self.time_emb_proj(t_emb).chunk(2, dim=1)
        h = h * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        h = self.act(h)
        return h

class UNet(nn.Module):
    def __init__(self, time_emb_dim=128, image_size=64):
        super().__init__()
        # 时间嵌入处理（image_size -> time_emb_dim）
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(image_size),
            nn.Linear(image_size, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 下采样路径
        self.conv1 = nn.Conv2d(3, image_size, 3, padding=1)
        self.down1 = DownBlock(image_size, image_size * 2, time_emb_dim=time_emb_dim)
        self.down2 = DownBlock(image_size * 2, image_size * 4, time_emb_dim=time_emb_dim)
        self.down3 = DownBlock(image_size * 4, image_size * 8, time_emb_dim=time_emb_dim)
        
        # 中间层（带时间嵌入）
        self.mid_conv1 = nn.Conv2d(image_size * 8, image_size * 8, 3, padding=1)
        self.mid_norm = nn.GroupNorm(8, image_size * 8)
        self.mid_time_proj = nn.Linear(time_emb_dim, image_size * 8 * 2)
        self.mid_act = nn.GELU()
        
        # 上采样路径
        self.up1 = UpBlock(image_size * 8, image_size * 4, time_emb_dim=time_emb_dim)
        self.up2 = UpBlock(image_size * 4, image_size * 2, time_emb_dim=time_emb_dim)
        self.up3 = UpBlock(image_size * 2, image_size, time_emb_dim=time_emb_dim)
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(image_size, image_size, 3, padding=1),
            nn.GroupNorm(8, image_size),
            nn.GELU(),
            nn.Conv2d(image_size, 3, 1),
        )

    def forward(self, x, t):
        x = x.to(next(self.parameters()).dtype)
        t = t.to(next(self.parameters()).dtype)
        t = t.to(next(self.parameters()).dtype)
        t_embed = self.time_embed(t)  
        
        # 下采样路径
        x1 = F.gelu(self.conv1(x))  # [B,64,64,64]
        x2 = self.down1(x1, t_embed)
        x3 = self.down2(x2, t_embed)
        x = self.down3(x3, t_embed)
        '''
        print(f"x1 shape: {x1.shape}")
        print(f"x2 shape: {x2.shape}")
        print(f"x3 shape: {x3.shape}")
        print(f"x shape before up1: {x.shape}")
        '''
        # 中间处理
        x = self.mid_conv1(x)
        # 应用时间嵌入
        scale, shift = self.mid_time_proj(t_embed).chunk(2, dim=1)
        x = x * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        x = self.mid_norm(x)
        x = self.mid_act(x)
        
        # 上采样路径
        x = self.up1(x, x3, t_embed)
        x = self.up2(x, x2, t_embed)
        x = self.up3(x, x1, t_embed)
        
        return self.final_conv(x)

class UNetSimulation(nn.Module):
    def __init__(self, time_emb_dim=128, image_size=64):
        super().__init__()
        # 时间嵌入处理（image_size -> time_emb_dim）
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(image_size),
            nn.Linear(image_size, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 下采样路径
        self.conv1 = nn.Conv2d(1, image_size, 3, padding=1)
        self.down1 = DownBlock(image_size, image_size * 2, time_emb_dim=time_emb_dim)
        self.down2 = DownBlock(image_size * 2, image_size * 4, time_emb_dim=time_emb_dim)
        self.down3 = DownBlock(image_size * 4, image_size * 8, time_emb_dim=time_emb_dim)
        
        # 中间层（带时间嵌入）
        self.mid_conv1 = nn.Conv2d(image_size * 8, image_size * 8, 3, padding=1)
        self.mid_norm = nn.GroupNorm(8, image_size * 8)
        self.mid_time_proj = nn.Linear(time_emb_dim, image_size * 8 * 2)
        self.mid_act = nn.GELU()
        
        # 上采样路径
        self.up1 = UpBlock(image_size * 8, image_size * 4, time_emb_dim=time_emb_dim)
        self.up2 = UpBlock(image_size * 4, image_size * 2, time_emb_dim=time_emb_dim)
        self.up3 = UpBlock(image_size * 2, image_size, time_emb_dim=time_emb_dim)
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(image_size, image_size, 3, padding=1),
            nn.GroupNorm(8, image_size),
            nn.GELU(),
            nn.Conv2d(image_size, 1, 1),
        )

    def forward(self, x, t):
        x = x.to(next(self.parameters()).dtype)
        t = t.to(next(self.parameters()).dtype)
        t = t.to(next(self.parameters()).dtype)
        t_embed = self.time_embed(t)  
        
        # 下采样路径
        x1 = F.gelu(self.conv1(x))  # [B,64,64,64]
        x2 = self.down1(x1, t_embed)
        x3 = self.down2(x2, t_embed)
        x = self.down3(x3, t_embed)
        
        '''
        print(f"x1 shape: {x1.shape}")
        print(f"x2 shape: {x2.shape}")
        print(f"x3 shape: {x3.shape}")
        print(f"x shape before up1: {x.shape}")
        
        '''
        
        # 中间处理
        x = self.mid_conv1(x)
        # 应用时间嵌入
        scale, shift = self.mid_time_proj(t_embed).chunk(2, dim=1)
        x = x * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        x = self.mid_norm(x)
        x = self.mid_act(x)
        
        # 上采样路径
        x = self.up1(x, x3, t_embed)
        x = self.up2(x, x2, t_embed)
        x = self.up3(x, x1, t_embed)
        
        return self.final_conv(x)

class UNetSimulationWithMoE(nn.Module):
    def __init__(self, time_emb_dim, image_size, num_experts=4, moe_hidden_dim=64, moe_tau=0.1):
        super().__init__()
        # 时间嵌入处理（image_size -> time_emb_dim）
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(image_size),
            nn.Linear(image_size, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 下采样路径
        self.conv1 = nn.Conv2d(1, image_size, 3, padding=1)
        self.down1 = DownBlock(image_size, image_size * 2, time_emb_dim=time_emb_dim)
        self.down2 = DownBlock(image_size * 2, image_size * 4, time_emb_dim=time_emb_dim)
        self.down3 = DownBlock(image_size * 4, image_size * 8, time_emb_dim=time_emb_dim)
        
        # 中间层（带时间嵌入）
        self.mid_conv1 = nn.Conv2d(image_size * 8, image_size * 8, 3, padding=1)
        self.mid_norm = nn.GroupNorm(8, image_size * 8)
        self.mid_time_proj = nn.Linear(time_emb_dim, image_size * 8 * 2)
        self.mid_act = nn.GELU()
        
        # 上采样路径
        self.up1 = UpBlock(image_size * 8, image_size * 4, time_emb_dim=time_emb_dim)
        self.up2 = UpBlock(image_size * 4, image_size * 2, time_emb_dim=time_emb_dim)
        self.up3 = UpBlock(image_size * 2, image_size, time_emb_dim=time_emb_dim)
        
        # 替换最终输出层为MoE
        self.final_proj = nn.Conv2d(image_size, 1, 3, padding=1)
        self.moe = ReparamGaussianMoE(
            input_dim=1,  # MoE输入维度
            num_experts=num_experts,
            hidden_dim=moe_hidden_dim,
            tau=moe_tau,
            flatten=False  # 输入已是2D [B, C, H, W]
        )

    def forward(self, x, t):
        assert x.dim() == 4, f"输入必须是4D张量，但得到 {x.shape}"
        x = x.to(next(self.parameters()).dtype)
        t = t.to(next(self.parameters()).dtype)
        t = t.to(next(self.parameters()).dtype)
        t_embed = self.time_embed(t)  
        
        # 下采样路径
        x1 = F.gelu(self.conv1(x))  # [B,64,64,64]
        x2 = self.down1(x1, t_embed)
        x3 = self.down2(x2, t_embed)
        x = self.down3(x3, t_embed)
        
        # 中间处理
        x = self.mid_conv1(x)
        # 应用时间嵌入
        scale, shift = self.mid_time_proj(t_embed).chunk(2, dim=1)
        x = x * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        x = self.mid_norm(x)
        x = self.mid_act(x)
        
        # 上采样路径
        x = self.up1(x, x3, t_embed)
        x = self.up2(x, x2, t_embed)
        x = self.up3(x, x1, t_embed)
        
        # 确保输入final_proj的形状是[B,64,H,W]
        # print(f"输入final_proj前的形状: {x.shape}")  # 应为[B,64,H,W]
        x = self.final_proj(x)  # [B, C, H, W]
        # print(f"输入final_proj后的形状: {x.shape}")  # 应为[B,64,H,W]
        
        # 将空间维度展平为特征维度
        B, C, H, W = x.shape
        
        
        # x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)  # [B, H*W, C]
        # x_flat = x_flat.reshape(-1, C)  # [B*H*W, C] （强制展平以满足MoE输入要求）
        
        # 转换为MoE需要的2D输入 [B, C]
        x_flat = x.mean(dim=[2, 3])  # 全局平均池化 → [B, 8]
        # print(f"x_flat的形状: {x_flat.shape}")
        
        # 通过MoE生成输出
        moe_out = self.moe(x_flat)  # [B, H*W, C]
        # print(f"moe_out的形状: {moe_out.shape}")
        
        # 恢复空间维度
        # output = moe_out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        output = moe_out[:, :, None, None].expand(-1, -1, H, W)  # [B,8,H,W]
        # print(f"output的形状: {output.shape}")
        return output