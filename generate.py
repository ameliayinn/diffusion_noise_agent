# generate.py
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np
from diffusion import DiffusionProcess
from tqdm import tqdm
from unet import UNetSimulation
import torch.nn as nn

@torch.no_grad()
def generate_samples(config, model_path, num_images=16, use_moe=False):
    """样本生成函数（整合MoE版本）
    
    Args:
        config (Config): 配置参数
        model_path (str): 模型路径
        num_images (int): 生成数量
        use_moe (bool): 是否使用混合专家
    """
    # 初始化扩散过程
    dp = DiffusionProcess(config.timesteps, config.image_size, use_moe=use_moe)
    
    # 加载模型
    model = UNetSimulation(time_emb_dim=config.time_emb_dim, 
                          image_size=config.image_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(config.device).eval()
    
    # 组合模型
    if use_moe:
        model = nn.Sequential(model, dp.moe.to(config.device))
    
    # 生成初始噪声（可选择噪声类型）
    noise_type = 'moe' if use_moe else 'gaussian'
    
    # 首先创建一个适当形状的初始噪声张量
    x0 = torch.randn(num_images, 3, 64, 64)  # 调整形状参数

    # 然后调用函数
    x = dp._generate_moe_noise(x0)
    # x = dp._generate_moe_noise(num_images, noise_type, device=config.device)
    
    # 反向扩散过程
    for t in tqdm(reversed(range(0, config.timesteps)), desc="Generating"):
        t_batch = torch.full((num_images,), t, device=config.device)
        
        # 通过模型预测噪声
        pred_noise = model(x, t_batch)
        
        # 更新样本
        x = dp.reverse_step(x, pred_noise, t)
    
    # 后处理并保存
    _save_results(x, "generated_samples.png")

@torch.no_grad()
def generate_during_training(model_engine, save_dir, config, epoch=None, 
                           num_images=16, p=0.5, use_moe=False):
    """训练时生成样本（整合MoE版本）"""
    model_engine.eval()
    dp = DiffusionProcess(config.timesteps, config.image_size, use_moe=use_moe)
    
    # 生成初始噪声
    noise_type = 'moe' if use_moe else 'mixed'
    # 首先创建一个适当形状的初始噪声张量
    x0 = torch.randn(num_images, 3, 64, 64, device=model_engine.device)  # 调整形状参数

    # 然后调用函数
    x = dp._generate_moe_noise(x0)
    # x = dp._generate_moe_noise(num_images, noise_type, device=model_engine.device, p=p)
    
    # 反向扩散
    for t in reversed(range(0, config.timesteps)):
        t_batch = torch.full((num_images,), t, device=model_engine.device)
        
        # 通过模型预测噪声
        if use_moe:
            base_pred = model_engine.module[0](x, t_batch)  # UNet部分
            pred_noise = model_engine.module[1](base_pred)  # MoE部分
        else:
            pred_noise = model_engine(x, t_batch)
        
        # 更新样本
        x = dp.reverse_step(x, pred_noise, t)
    
    # 保存结果
    _save_simulation_results(x, save_dir, epoch)

def _save_results(x, filename):
    """保存生成结果（通用）"""
    x = (x.clamp(-1, 1) + 1) * 0.5  # [0,1]范围
    grid = torchvision.utils.make_grid(x.cpu(), nrow=4)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

def _save_simulation_results(x, save_dir, epoch=None):
    """保存模拟数据结果"""
    os.makedirs(save_dir, exist_ok=True)
    x = x.to(torch.float32).cpu()
    
    # 保存原始数据
    res_list = x.flatten().tolist()
    with open(os.path.join(save_dir, "samples.txt"), "w") as f:
        f.write(str(res_list))
    
    # 保存统计信息
    min_val, max_val = min(res_list), max(res_list)
    with open(os.path.join(save_dir, "range.txt"), "w") as f:
        f.write(f"min: {min_val}, max: {max_val}, len: {len(res_list)}")
    
    # 绘制分布图
    plt.hist(res_list, bins=50, edgecolor='black')
    plt.title(f"Epoch {epoch}" if epoch else "Frequency Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, "distribution.png"))
    plt.close()

# 兼容旧版本函数
def generate_during_training_simulation(model_engine, *args, **kwargs):
    return generate_during_training(model_engine, *args, **kwargs)

def generate_during_training_simulation_dif(model_engine, save_dir, config, 
                                          epoch, num_images=16, p=0.5):
    """兼容不同噪声版本的生成函数"""
    return generate_during_training(
        model_engine, save_dir, config, epoch, num_images, p, use_moe=config.use_moe
    )