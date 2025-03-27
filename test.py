# -------------------- 环境配置 --------------------
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import deepspeed
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torch.optim.lr_scheduler import MultiStepLR
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用国内镜像加速
os.environ["HF_DATASETS_TIMEOUT"] = "600"            # 超时时间设为10分钟

# -------------------- 配置参数类 --------------------
class Config:
    """全局配置参数
    Attributes:
        image_size (int): 输入图像尺寸 [64x64]
        batch_size (int): 全局批次大小（所有GPU的总和） [256]
        num_epochs (int): 训练总轮次 [1000]
        timesteps (int): 扩散过程时间步数 [1000]
        lr (float): 学习率 [2e-4]
        dataset_name (str): 数据集名称 ["cats_vs_dogs"]
        checkpoint_dir (str): 检查点保存路径 ["./checkpoints"]
        fp16 (bool): 是否启用混合精度训练 [True]
    """
    image_size = 64
    batch_size = 512
    num_epochs = 5000
    timesteps = 1000
    samples_dir = "./samples"  
    lr = 2e-4
    dataset_name = "cats_vs_dogs"
    checkpoint_dir = "./checkpoints"
    fp16 = True

# -------------------- 时间嵌入模块 --------------------
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

# -------------------- UNet网络定义 --------------------
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
    def __init__(self):
        super().__init__()
        # 时间嵌入处理（64 -> 128）
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(64),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        
        # 下采样路径
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.down1 = DownBlock(64, 128, time_emb_dim=128)
        self.down2 = DownBlock(128, 256, time_emb_dim=128)
        self.down3 = DownBlock(256, 512, time_emb_dim=128)
        
        # 中间层（带时间嵌入）
        self.mid_conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.mid_norm = nn.GroupNorm(8, 512)
        self.mid_time_proj = nn.Linear(128, 512 * 2)
        self.mid_act = nn.GELU()
        
        # 上采样路径
        self.up1 = UpBlock(512, 256, time_emb_dim=128)
        self.up2 = UpBlock(256, 128, time_emb_dim=128)
        self.up3 = UpBlock(128, 64, time_emb_dim=128)
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 3, 1),
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

# -------------------- 扩散工具函数 --------------------
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

# -------------------- 数据加载 --------------------
def load_data(config, local_rank):
    """加载并预处理数据集
    Args:
        config (Config): 配置参数
        local_rank (int): 当前GPU编号
    Returns:
        DataLoader: 训练数据加载器
    """
    # 图像预处理流水线
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 正确归一化到[-1,1]
        transforms.Lambda(lambda x: x.half())  # 转换为FP16
    ])
    dataset = load_dataset(config.dataset_name, split='train')
    dataset = dataset.train_test_split(test_size=0.1)
    
    def transform_fn(examples):
        examples["image"] = [transform(img.convert("RGB")) for img in examples["image"]]
        return examples
    
    dataset = dataset.map(transform_fn, batched=True)
    dataset.set_format(type='torch', columns=['image'])
    
    # 确保只在分布式环境中使用DistributedSampler
    if torch.distributed.is_initialized():
        train_sampler = DistributedSampler(
            dataset["train"],
            shuffle=True,
            rank=local_rank,
            num_replicas=torch.distributed.get_world_size()
        )
    else:
        train_sampler = None
    
    return dataset["train"]

@torch.no_grad()
def generate_during_training(model_engine, save_dir, config, num_images=16):
    model_engine.eval()
    device = model_engine.device
    
    betas = linear_beta_schedule(config.timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_over_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    x = torch.randn(num_images, 3, config.image_size, config.image_size, device=device, dtype=torch.half)
    x = x.to(next(model_engine.parameters()).dtype)
    
    for t in reversed(range(0, config.timesteps)):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        pred_noise = model_engine(x, t_batch)
        
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        noise = torch.randn_like(x) if t > 0 else 0
        
        x = sqrt_one_over_alphas[t] * (x - beta_t * pred_noise / sqrt_one_minus_alphas_cumprod[t]) + torch.sqrt(beta_t) * noise
    
    # 后处理并转换数据类型
    x = (x.clamp(-1, 1) + 1) * 0.5
    x = x.to(torch.float32)  # 确保转换为float32
    grid = torchvision.utils.make_grid(x.cpu(), nrow=4)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).numpy())  # 数据现在是float32
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, "samples.png"))
    plt.close()

# -------------------- 训练循环 --------------------
# -------------------- 训练循环 --------------------
def train_deepspeed(config):
    """DeepSpeed训练主函数"""
    # 初始化模型
    model = UNet()
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
            "enabled": True,
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
    
    # 初始化引擎
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    train_dataset = load_data(config, local_rank)
    
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
    
    # 训练循环
    from tqdm import tqdm
    for epoch in range(config.num_epochs):
        model_engine.train()
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'sampler'):
            if isinstance(train_loader.batch_sampler.sampler, DistributedSampler):
                train_loader.batch_sampler.sampler.set_epoch(epoch)
        
        for batch in tqdm(train_loader):
            images = batch["image"].to(model_engine.device) # [B,3,64,64]
            images = images.to(torch.float16)
            t = torch.randint(0, config.timesteps, (images.size(0),)).to(model_engine.device)
            
            # 前向扩散
            noisy_images, noise = forward_diffusion(
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
            print(f"Current lr: {scheduler.get_last_lr()[0]:.8f}")  # 验证学习率变化
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            model_path = os.path.join(config.checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model_engine.module.state_dict(), model_path)
            
            sample_dir = os.path.join(config.samples_dir, f"epoch_{epoch+1}")
            os.makedirs(sample_dir, exist_ok=True)
            generate_during_training(model_engine, sample_dir, config, num_images=16)
            
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Samples saved to {sample_dir}")
# -------------------- 生成函数 --------------------
@torch.no_grad()
def generate_samples(config, model_path, num_images=16):
    """样本生成函数
    调用路径：
    手动调用 -> UNet.forward
                └── 反向扩散过程
    Args:
        config (Config): 配置参数
        model_path (str): 模型路径
        num_images (int): 生成数量
    """
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))  # 加载到CPU再转移
    model.to(config.device).eval()
    
    # 扩散参数移动到设备
    betas = linear_beta_schedule(config.timesteps).to(config.device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_over_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    betas_cumprod = 1. - alphas_cumprod
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
    
    # 生成噪声
    x = torch.randn(num_images, 3, config.image_size, config.image_size).to(config.device)  # [B,3,64,64]
    
    # 反向扩散过程
    for t in reversed(range(0, config.timesteps)):
        t_batch = torch.full((num_images,), t, device=config.device)  # [B]
        pred_noise = model(x, t_batch)  # [B,3,64,64]
        
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0
        
        # 更新公式
        x = sqrt_one_over_alphas[t] * (x - beta_t * pred_noise / sqrt_one_minus_alphas_cumprod[t]) + torch.sqrt(beta_t) * noise
    
    # 后处理
    x = (x.clamp(-1, 1) + 1) * 0.5  # [0,1]范围
    grid = torchvision.utils.make_grid(x.cpu(), nrow=4)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig("generated_samples.png")
    plt.close()

# -------------------- 主函数 --------------------
if __name__ == "__main__":
    config = Config()
    train_deepspeed(config)  # 训练入口
    # generate_samples(config, "checkpoints/epoch_1000/pytorch_model.bin")  # 生成入口