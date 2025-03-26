import torch

def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """生成线性 beta 调度表"""
    return torch.linspace(0.0001, 0.02, timesteps, dtype=torch.float32)

def forward_diffusion_with_different_noise(
    x0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    mu1: float = -0.3,
    mu2: float = 0.5,
    sigma1: float = 0.95,
    sigma2: float = 0.95,
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    前向扩散过程，支持不同噪声分布
    
    Args:
        x0: 原始图像 [B, C, H, W] 或其他形状
        t: 时间步 [B]
        sqrt_alphas_cumprod: 预计算 alpha 累积乘积的平方根 [T]
        sqrt_one_minus_alphas_cumprod: 1 减去 alpha 累积乘积的平方根 [T]
        mu1, mu2: 两个噪声分布的均值
        sigma1, sigma2: 两个噪声分布的标准差
        p: 采样第一个分布的概率
    
    Returns:
        noisy_images: 加噪后的图像
        noise: 添加的噪声
    """
    device = x0.device
    batch_size = x0.shape[0]
    
    # 随机选择噪声来源
    mask = torch.rand(batch_size, device=device) < p
    noise1 = torch.randn_like(x0, dtype=torch.float32) * sigma1 + mu1
    noise2 = torch.randn_like(x0, dtype=torch.float32) * sigma2 + mu2
    noise = torch.where(mask.view(-1, *([1] * (x0.ndim - 1))), noise1, noise2)
    
    # 计算加噪图像
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, *([1] * (x0.ndim - 1)))
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x0.ndim - 1)))
    
    noisy_images = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
    return noisy_images, noise
