import torch
import torch.nn as nn
from unet import UNetSimulationWithMoE  # 假设您的UNet类在这个模块中

# 测试输入
B, C, H, W = 8, 1, 64, 64
x = torch.randn(B, C, H, W)
t = torch.randn(B)

# 模型初始化
model = UNetSimulationWithMoE(time_emb_dim=64, image_size=64, num_experts=4)

# 方案1测试
# output1 = model(x, t)
# print("方案1输出形状:", output1.shape)  # 应该是 [B, 1, H, W]

# 方案2测试（需调整 MoE 支持 3D 输入）
# output2 = model(x, t)
# print("方案2输出形状:", output2.shape)  # 应该是 [B, 1, H, W]

# 方案3测试（需 MoE 输出多通道）
output3 = model(x, t)
print("方案3输出形状:", output3.shape)  # 应该是 [B, image_size, H, W]