import torch

# 定义两个高斯分布的参数
mu1, sigma1 = 0.0, 1.0
mu2, sigma2 = 3.0, 2.0
w1, w2 = 0.5, 0.5  # 50% 概率选择第一个正态分布

def sample_mixed_gaussian(batch_size, shape, device):
    """ 采样来自混合高斯分布的噪声 """
    b = torch.bernoulli(torch.full((batch_size, 1, 1, 1), w1, device=device))  # 选择哪个高斯分布
    noise1 = torch.randn(shape, device=device) * sigma1 + mu1
    noise2 = torch.randn(shape, device=device) * sigma2 + mu2
    return b * noise1 + (1 - b) * noise2

# 训练时采样噪声
batch_size = 16
image_shape = (batch_size, 3, 32, 32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epsilon = sample_mixed_gaussian(batch_size, image_shape, device)

with torch.no_grad():
    x_T = sample_mixed_gaussian(num_samples, (num_samples, 3, 32, 32), device)  # 从混合高斯初始化
    for t in reversed(range(T)):  # 逐步去噪
        z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        pred_noise = model(x_T, torch.full((num_samples,), t, device=device))
        x_T = (1 / torch.sqrt(alpha[t])) * (x_T - (beta[t] / torch.sqrt(1 - alpha_cumprod[t])) * pred_noise) + torch.sqrt(beta_t) * z
