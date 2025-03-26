# diffusion.py
import torch
import numpy as np
from reparam_moe import ReparamGaussianMoE  # 导入MoE模块

class DiffusionProcess:
    def __init__(self, timesteps, image_size, use_moe=False, moe_config=None):
        """
        Args:
            timesteps (int): 总时间步数
            image_size (int): 图像尺寸
            use_moe (bool): 是否使用混合专家
            moe_config (dict): MoE配置参数
        """
        self.timesteps = timesteps
        self.image_size = image_size
        
        # 初始化beta调度表
        self.betas = self.linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # 初始化MoE
        self.use_moe = use_moe
        if use_moe:
            moe_config = moe_config or {}
            self.moe = ReparamGaussianMoE(
                input_dim=image_size**2,
                num_experts=moe_config.get('num_experts', 4),
                hidden_dim=moe_config.get('hidden_dim', 64),
                tau=moe_config.get('tau', 0.1),
                flatten=True
            )

    @staticmethod
    def linear_beta_schedule(timesteps):
        """生成线性beta调度表"""
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    def forward_diffusion(self, x0, t, p=0.5, noise_type='mixed'):
        """
        改进的前向扩散过程，支持多种噪声类型和MoE
        
        Args:
            x0 (Tensor): 原始图像 [B,C,H,W]
            t (Tensor): 时间步 [B]
            p (float): 混合噪声的比例
            noise_type (str): 噪声类型 ('mixed', 'gaussian', 'moe')
            
        Returns:
            noisy_images (Tensor): 加噪图像 [B,C,H,W]
            noise (Tensor): 添加的噪声 [B,C,H,W]
        """
        B, C, H, W = x0.shape
        
        # 生成基础噪声
        if noise_type == 'mixed':
            noise = self._generate_mixed_noise(x0, p)
        elif noise_type == 'moe' and self.use_moe:
            noise = self._generate_moe_noise(x0)
        else:  # 默认高斯噪声
            noise = torch.randn_like(x0)
        
        # 计算加噪图像
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        noisy_images = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return noisy_images, noise

    def _generate_mixed_noise(self, x0, p):
        """生成混合噪声"""
        mask = torch.rand(x0.size(0), 1, 1, 1, device=x0.device) < p
        noise1 = torch.randn_like(x0) * 0.95 - 0.3  # N(-0.3, 0.95)
        noise2 = torch.randn_like(x0) * 0.95 + 0.5  # N(0.5, 0.95)
        return torch.where(mask, noise1, noise2)

    def _generate_moe_noise(self, x0):
        """使用MoE生成噪声"""
        B, C, H, W = x0.shape
        x_flat = x0.view(B, -1)  # 展平输入
        moe_noise = self.moe(x_flat)
        return moe_noise.view(B, C, H, W)  # 恢复形状

    def q_sample(self, x0, t, noise=None):
        """更方便的扩散采样接口"""
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self.sqrt_alphas_cumprod[t] * x0 + 
            self.sqrt_one_minus_alphas_cumprod[t] * noise
        )

# 保持原有函数兼容性
def linear_beta_schedule(timesteps):
    return DiffusionProcess.linear_beta_schedule(timesteps)

def forward_diffusion_with_different_noise(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, p=0.5):
    """兼容旧代码的包装函数"""
    dp = DiffusionProcess(len(sqrt_alphas_cumprod), x0.size(2))
    dp.sqrt_alphas_cumprod = sqrt_alphas_cumprod
    dp.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
    return dp.forward_diffusion(x0, t, p=p, noise_type='mixed')