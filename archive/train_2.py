import os
import torch
import deepspeed
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from archive.diffusion_2 import linear_beta_schedule
from archive.unet_1 import UNetSimulation
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import json
from collections import Counter
from reparam_gaussian_moe import ReparamGaussianMoE  # 引入新的 MoE 模型

def train_deepspeed(config):
    """DeepSpeed训练主函数"""
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    config.checkpoints_dir = os.path.join(config.checkpoints_dir, f"checkpoints_{timestamp}")
    config.samples_dir = os.path.join(config.samples_dir, f"samples_{timestamp}")
    config.logs_dir = os.path.join(config.logs_dir, f"logs_{timestamp}")
    
    # 初始化 MoE 模型
    model = ReparamGaussianMoE(input_dim=config.input_dim, num_experts=config.num_experts)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # DeepSpeed配置
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
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    train_dataset = load_data(config, local_rank)
    
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        config_params=ds_config,
        training_data=train_dataset,
        dist_init_required=True
    )
    
    scheduler = MultiStepLR(optimizer.optimizer, milestones=[500, 1000, 1500], gamma=0.1)
    
    print(f"****START TRAINING****\ninput_dim: {config.input_dim}, batch_size: {config.batch_size}")
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)
    
    csv_filename = f"train_log.csv"
    csv_filepath = os.path.join(config.logs_dir, csv_filename)
    
    if model_engine.local_rank == 0:
        df = pd.DataFrame(columns=["epoch", "loss", "learning_rate"])
        df.to_csv(csv_filepath, index=False)
    
    with open(f'{csv_filepath[:-4]}.json', "w") as f:
        json.dump(vars(config), f, indent=4)
    
    for epoch in range(config.num_epochs):
        model_engine.train()
        if isinstance(train_loader.batch_sampler.sampler, DistributedSampler):
            train_loader.batch_sampler.sampler.set_epoch(epoch)
        
        for batch in tqdm(train_loader):
            x = batch["input"].to(model_engine.device)
            x = x.to(torch.float16)
            output = model_engine(x)
            loss = F.mse_loss(output, x)  # 使用 MSE 作为损失函数
            
            model_engine.backward(loss)
            model_engine.step()
        
        scheduler.step()
        
        if model_engine.local_rank == 0:
            new_row = {"epoch": epoch + 1, "loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]}
            df = pd.read_csv(csv_filepath)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(csv_filepath, index=False)
            
            model_path = os.path.join(config.checkpoints_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model_engine.module.state_dict(), model_path)
    
    print("Training Complete!")
