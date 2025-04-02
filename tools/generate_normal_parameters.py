import torch
import math

# 参数设置
mu1 = -0.3  # 第一个分布的均值
p = 0.9     # 从第一个分布采样的概率
# mu2 = - (p / (1 - p)) * mu1  # 第二个分布的均值
mu2 = 0.5

# 检查约束条件
term = p * mu1**2 + (1 - p) * mu2**2
if term > 1:
    raise ValueError("参数不满足约束条件：p * mu1^2 + (1-p) * mu2^2 必须 <= 1")

# 计算 sigma
sigma = math.sqrt(1 - term)

print(f"mu1: {mu1}, mu2: {mu2}, sigma: {sigma}")

'''
noise_samples = []
for _ in range(10000):
    mask = torch.rand(1) < p
    if mask:
        noise_samples.append(torch.randn(1) * sigma + mu1)
    else:
        noise_samples.append(torch.randn(1) * sigma + mu2)
noise_samples = torch.cat(noise_samples)

print("Mean:", noise_samples.mean().item())  # 应接近 0
print("Std:", noise_samples.std().item())   # 应接近 1
'''