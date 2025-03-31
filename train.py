# train.py
import os
import torch
import deepspeed
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusion import linear_beta_schedule, forward_diffusion_with_moe
from unet import UNetSimulationWithMoE, UNetSimulation
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import json
from collections import Counter

def train_deepspeed(config):
    """函数引用"""
    def get_function(type, use_different_noise):
        if type == 'data':
            from utils.dataloader import load_data
            from generate import generate_during_training
        else:
            if use_different_noise:
                from generate import generate_during_training_simulation_dif as generate_during_training
            else:
                from generate import generate_during_training_simulation as generate_during_training
            if type == 'normal':
                from utils.dataloader import load_data_normal as load_data
            elif type == 'poisson':
                from utils.dataloader import load_data_poisson as load_data
        if use_different_noise:
            from diffusion import forward_diffusion_with_different_noise as forward_diffusion
        else:
            from diffusion import forward_diffusion
        return load_data, generate_during_training, forward_diffusion
    
    """DeepSpeed训练主函数"""
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    # 创建带时间戳的 checkpoints, sample 和 logs 文件夹
    config.checkpoints_dir = os.path.join(config.checkpoints_dir, f"checkpoints_{timestamp}")
    config.samples_dir = os.path.join(config.samples_dir, f"samples_{timestamp}")
    config.logs_dir = os.path.join(config.logs_dir, f"logs_{timestamp}")
    
    # 初始化模型
    model = UNetSimulationWithMoE(time_emb_dim=config.time_emb_dim, image_size=config.image_size, num_experts=config.num_experts, moe_hidden_dim=config.moe_hidden_dim, moe_tau=config.moe_tau)
    # model = UNetSimulation(time_emb_dim=config.time_emb_dim, image_size=config.image_size)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # DeepSpeed配置 (移除scheduler部分)
    ds_config = {
        "train_batch_size": config.batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.lr,
                "weight_decay": 0.01
            }
        },
        "fp16": {
            "enabled": config.fp16,
            "loss_scale": 0,
            "loss_scale_window": 1000
        },
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True
        },
        "steps_per_print": 50,
        "gradient_clipping": 1.0
    }
    
    # get fuction
    simulation_distribution, use_different_noise = config.simulation_distribution, config.use_different_noise
    load_data, generate_during_training, forward_diffusion = get_function(simulation_distribution, use_different_noise)
    
    # 初始化引擎
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    train_dataset = load_data(config, local_rank)
    # train_dataset = load_data(config, local_rank, seed=42) # 使用固定种子
    # train_dataset_random = load_data(config, local_rank=0, seed=None)  # 不使用固定种子
    
    # 初始化DeepSpeed引擎
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        config_params=ds_config,
        training_data=train_dataset,
        dist_init_required=True
    )
    
    # 手动创建PyTorch调度器 (关键修改)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    base_optimizer = optimizer.optimizer  # 访问底层PyTorch优化器
    scheduler = MultiStepLR(
        base_optimizer, 
        milestones=[500, 1000, 1500],  # 在epoch=500、1000、1500时衰减
        gamma=0.1  # 每次衰减为之前的0.1倍
    )
    
    # 准备扩散参数
    betas = linear_beta_schedule(config.timesteps).to(model_engine.device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    # 提示开始
    print(f"****START TRAINING****\nimage_size: {config.image_size}, batch_size: {config.batch_size}, timesteps: {config.timesteps}, time_emb_dim: {config.time_emb_dim}")
    
    # 创建带有时间戳的路径
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)
    
    # 构造 CSV 文件路径
    csv_filename = f"is_{config.image_size}_bs_{config.batch_size}_tstep_{config.timesteps}_tdim_{config.time_emb_dim}.csv"
    csv_filepath = os.path.join(config.logs_dir, csv_filename)
    
    # 创建 CSV 文件并写入表头
    if model_engine.local_rank == 0:  # 只在主进程创建
        df = pd.DataFrame(columns=["epoch", "loss", "image_size", "batch_size", "timesteps", "time_emb_dim", "learning_rate"])
        df.to_csv(csv_filepath, index=False)
    
    with open(f'{csv_filepath[:-4]}.json', "w") as f:
        json.dump(vars(config), f, indent=4)  # 将 args 转换为字典并保存为 JSON
    
    # 训练循环
    for epoch in range(config.num_epochs):
        model_engine.train()
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'sampler'):
            if isinstance(train_loader.batch_sampler.sampler, DistributedSampler):
                train_loader.batch_sampler.sampler.set_epoch(epoch)
        
        for batch in tqdm(train_loader):
            images = batch["image"].to(model_engine.device)
            # images = images.unsqueeze(1)  # 增加通道维度，形状变为 [batch_size, 1, 10, 10]
            images = images.to(torch.float16)
            # print(type(images))  # 应该是 <class 'torch.Tensor'>
            # print('----****images.shape****----', images.shape)  # 应该是 [B, 1, H, W]
            t = torch.randint(0, config.timesteps, (images.size(0),)).to(model_engine.device)
            p = config.num1 / (config.num1 + config.num2)
            # p = 0.5
            # print('----****p****----', p)
            
            # 前向扩散
            # 使用MoE生成噪声
            noisy_images, noise = forward_diffusion_with_moe(
                model_engine.module,  # 注意DeepSpeed的模型访问方式
                images, t,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod
            )
            
            # 预测噪声
            pred_noise = model_engine(noisy_images, t)
            
            # 计算损失
            loss = F.mse_loss(pred_noise, noise)
            
            # 反向传播
            model_engine.backward(loss)
            model_engine.step()
        
        # 手动更新学习率 (关键修改)
        scheduler.step()
        
        # 保存检查点
        if model_engine.local_rank == 0:
            # print(f"Current lr: {scheduler.get_last_lr()[0]:.8f}")  # 验证学习率变化
            
            # 记录 epoch 结果到 CSV
            new_row = {
                "epoch": epoch + 1,
                "loss": loss.item(),
                "image_size": config.image_size,
                "batch_size": config.batch_size,
                "timesteps": config.timesteps,
                "time_emb_dim": config.time_emb_dim,
                "learning_rate": scheduler.get_last_lr()[0]
            }
            df = pd.read_csv(csv_filepath)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True) # 使用 pd.concat 追加数据
            df.to_csv(csv_filepath, index=False)
            
            # 保存模型检查点
            model_path = os.path.join(
                config.checkpoints_dir,
                f"model_is_{config.image_size}_bs_{config.batch_size}_tstep_{config.timesteps}_tdim_{config.time_emb_dim}_epoch_{epoch+1}.pt"
            )
            torch.save(model_engine.module.state_dict(), model_path)
            
            if (epoch + 1) % 20 == 0:
                os.makedirs(config.samples_dir, exist_ok=True)
                
                # 生成样本
                sample_dir = os.path.join(
                    config.samples_dir,
                    f"is_{config.image_size}_bs_{config.batch_size}_tstep_{config.timesteps}_tdim_{config.time_emb_dim}_epoch_{epoch+1}"
                )
                os.makedirs(sample_dir, exist_ok=True)
                
                generate_during_training(model_engine, sample_dir, config, epoch, num_images=config.num_images//config.image_size)
                # generate_during_training(model_engine, sample_dir, config, num_images=config.num_images)
            
            # print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Samples saved to {sample_dir}")