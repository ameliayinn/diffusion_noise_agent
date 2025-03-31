# diffusion.py
import torch
import numpy as np
import torch.nn as nn
from reparam_moe import ReparamGaussianMoE

def linear_beta_schedule(timesteps):
    """生成线性beta调度表
    Args:
        timesteps (int): 总时间步数
    Returns:
        betas (Tensor): [timesteps]
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def forward_diffusion(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """前向扩散过程（闭式解）
    Args:
        x0 (Tensor): 原始图像 [B,C,H,W]
        t (Tensor): 时间步 [B]
    Returns:
        noisy_images (Tensor): 加噪图像 [B,C,H,W]
        noise (Tensor): 添加的噪声 [B,C,H,W]
    """
    noise = torch.randn_like(x0)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise, noise

def forward_diffusion_with_different_noise(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, mu1=-0.3, mu2=0.5, sigma1=0.95, sigma2=0.95, p=0.5):
    """前向扩散过程（闭式解）
    Args:
        x0 (Tensor): 原始图像 [B,C,H,W]
        t (Tensor): 时间步 [B]
        sqrt_alphas_cumprod (Tensor): 累积乘积的平方根 [T]
        sqrt_one_minus_alphas_cumprod (Tensor): 1减去累积乘积的平方根 [T]
        mu1 (float): 第一个正态分布的均值
        mu2 (float): 第二个正态分布的均值
        sigma1 (float): 第一个正态分布的标准差
        sigma2 (float): 第二个正态分布的标准差
        p (float): 从第一个分布采样的概率
    Returns:
        noisy_images (Tensor): 加噪图像 [B,C,H,W]
        noise (Tensor): 添加的噪声 [B,C,H,W]
    """
    # 生成一个随机数来决定从哪个分布采样
    mask = torch.rand(x0.size(0), 1, 1, 1, device=x0.device) < p
    
    # 从两个正态分布中采样噪声
    noise1 = torch.randn_like(x0) * sigma1 + mu1
    noise2 = torch.randn_like(x0) * sigma2 + mu2
    
    # 根据 mask 选择噪声
    noise = torch.where(mask, noise1, noise2)
    
    # 计算加噪图像
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    
    noisy_images = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
    
    return noisy_images, noise

def forward_diffusion_with_different_noise_1(x0_1, x0_2, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    对两份数据分别进行加噪
    Args:
        x0_1: 第一份数据 [B, C, H, W]
        x0_2: 第二份数据 [B, C, H, W]
        t: 时间步 [B]
        sqrt_alphas_cumprod: 累积 alpha 的平方根 [T]
        sqrt_one_minus_alphas_cumprod: 1 - 累积 alpha 的平方根 [T]
    Returns:
        noisy_x1: 加噪后的第一份数据
        noisy_x2: 加噪后的第二份数据
        noise1: 第一份数据的噪声
        noise2: 第二份数据的噪声
    """
    noise1 = torch.randn_like(x0_1)
    noise2 = torch.randn_like(x0_2)
    
    # 计算加噪后的数据
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    
    noisy_x1 = sqrt_alpha_cumprod_t * x0_1 + sqrt_one_minus_alpha_cumprod_t * noise1
    noisy_x2 = sqrt_alpha_cumprod_t * x0_2 + sqrt_one_minus_alpha_cumprod_t * noise2
    
    return noisy_x1, noisy_x2, noise1, noise2

def forward_diffusion_with_moe(model, x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """使用模型预测的混合高斯噪声"""
    assert x0.dim() == 4, "输入必须是4D张量[B,C,H,W]"
    # 模型预测噪声分布参数
    # with torch.no_grad():
    pred_params = model(x0, t)  # 输出已包含混合高斯的采样结果
    print(f"x0形状: {x0.shape}, 模型输出形状: {pred_params.shape}")
    assert pred_params.shape == x0.shape, f"模型输出形状{pred_params.shape}与输入{x0.shape}不匹配"
    
    # 直接使用模型输出作为噪声
    noise = pred_params  # [B, C, H, W]
    
    # 计算加噪图像
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    noisy_images = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
    
    return noisy_images, noise