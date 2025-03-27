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
def get_config():
    parser = argparse.ArgumentParser(description="Diffusion Model Training with DeepSpeed")
    
    # 训练参数
    parser.add_argument("--image_size", type=int, default=64, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=256, help="Global batch size")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--dataset_name", type=str, default="cats_vs_dogs", help="Dataset name")
    
    # 路径参数
    parser.add_argument("--samples_dir", type=str, default="./samples", help="Directory to save samples")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--logs_dir", type=str, default="./logs", help="Directory to save logs")
    
    # 精度设置
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training (default: True)")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="Disable mixed precision training")
    
    # 运行模式
    parser.add_argument("--mode", choices=["train", "generate"], default="train", help="Run mode")
    parser.add_argument("--model_path", type=str, help="Model path for generation mode")
    parser.add_argument("--num_images", type=int, default=16, help="Number of images to generate")
    
    # 时间嵌入维度
    parser.add_argument("--time_emb_dim", type=int, default=128, help="Dimension of time embeddings")
    
    # MoE 相关参数
    parser.add_argument("--use_moe", action="store_true", default=False,
                      help="Enable Mixture of Experts module (default: False)")
    parser.add_argument("--num_experts", type=int, default=4,
                      help="Number of experts in MoE (default: 4)")
    parser.add_argument("--moe_hidden_dim", type=int, default=64,
                      help="Hidden dimension for MoE experts (default: 64)")
    parser.add_argument("--moe_tau", type=float, default=0.1,
                      help="Temperature for Gumbel-Softmax in MoE (default: 0.1)")
    
    # simulation data
    parser.add_argument("--simulation_distribution", type=str, default="data", help="Distribution of the simulation data")
    parser.add_argument("--num1", type=int, default=2, help="Number of data of the first normal distribution")
    parser.add_argument("--num2", type=int, default=2, help="Number of data of the second normal distribution")
    
    # normal distribution
    parser.add_argument("--mu1", type=int, default=2, help="Mu value of the first normal distribution")
    parser.add_argument("--sigma1", type=int, default=2, help="Sigma value of the first normal distribution")
    parser.add_argument("--mu2", type=int, default=2, help="Mu value of the second normal distribution")
    parser.add_argument("--sigma2", type=int, default=2, help="Sigma value of the second normal distribution")
    
    # poisson distribution
    parser.add_argument("--lambda1", type=int, default=2, help="Lambda value of the first poisson distribution")
    parser.add_argument("--lambda2", type=int, default=2, help="Lambda value of the second poisson distribution")
    
    # deepspeed 自动添加的参数
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed from deepspeed")
    
    # use different noise
    parser.add_argument("--use_different_noise", action="store_true", default=False, help="Use different noise for diffusion or not")
    
    # parser.set_defaults(fp16=True)
    return parser.parse_args()

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
    def __init__(self, time_emb_dim=128, image_size=64):
        super().__init__()
        # 时间嵌入处理（image_size -> time_emb_dim）
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(image_size),
            nn.Linear(image_size, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 下采样路径
        self.conv1 = nn.Conv2d(3, image_size, 3, padding=1)
        self.down1 = DownBlock(image_size, image_size * 2, time_emb_dim=time_emb_dim)
        self.down2 = DownBlock(image_size * 2, image_size * 4, time_emb_dim=time_emb_dim)
        self.down3 = DownBlock(image_size * 4, image_size * 8, time_emb_dim=time_emb_dim)
        
        # 中间层（带时间嵌入）
        self.mid_conv1 = nn.Conv2d(image_size * 8, image_size * 8, 3, padding=1)
        self.mid_norm = nn.GroupNorm(8, image_size * 8)
        self.mid_time_proj = nn.Linear(time_emb_dim, image_size * 8 * 2)
        self.mid_act = nn.GELU()
        
        # 上采样路径
        self.up1 = UpBlock(image_size * 8, image_size * 4, time_emb_dim=time_emb_dim)
        self.up2 = UpBlock(image_size * 4, image_size * 2, time_emb_dim=time_emb_dim)
        self.up3 = UpBlock(image_size * 2, image_size, time_emb_dim=time_emb_dim)
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(image_size, image_size, 3, padding=1),
            nn.GroupNorm(8, image_size),
            nn.GELU(),
            nn.Conv2d(image_size, 3, 1),
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
        '''
        print(f"x1 shape: {x1.shape}")
        print(f"x2 shape: {x2.shape}")
        print(f"x3 shape: {x3.shape}")
        print(f"x shape before up1: {x.shape}")
        '''
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

class UNetSimulation(nn.Module):
    def __init__(self, time_emb_dim=128, image_size=64):
        super().__init__()
        # 时间嵌入处理（image_size -> time_emb_dim）
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(image_size),
            nn.Linear(image_size, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 下采样路径
        self.conv1 = nn.Conv2d(1, image_size, 3, padding=1)
        self.down1 = DownBlock(image_size, image_size * 2, time_emb_dim=time_emb_dim)
        self.down2 = DownBlock(image_size * 2, image_size * 4, time_emb_dim=time_emb_dim)
        self.down3 = DownBlock(image_size * 4, image_size * 8, time_emb_dim=time_emb_dim)
        
        # 中间层（带时间嵌入）
        self.mid_conv1 = nn.Conv2d(image_size * 8, image_size * 8, 3, padding=1)
        self.mid_norm = nn.GroupNorm(8, image_size * 8)
        self.mid_time_proj = nn.Linear(time_emb_dim, image_size * 8 * 2)
        self.mid_act = nn.GELU()
        
        # 上采样路径
        self.up1 = UpBlock(image_size * 8, image_size * 4, time_emb_dim=time_emb_dim)
        self.up2 = UpBlock(image_size * 4, image_size * 2, time_emb_dim=time_emb_dim)
        self.up3 = UpBlock(image_size * 2, image_size, time_emb_dim=time_emb_dim)
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(image_size, image_size, 3, padding=1),
            nn.GroupNorm(8, image_size),
            nn.GELU(),
            nn.Conv2d(image_size, 1, 1),
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
        
        '''
        print(f"x1 shape: {x1.shape}")
        print(f"x2 shape: {x2.shape}")
        print(f"x3 shape: {x3.shape}")
        print(f"x shape before up1: {x.shape}")
        
        '''
        
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

# -------------------- 数据加载 --------------------
def load_data_normal(config, local_rank, seed=42):
    """加载并生成模拟数据集
    Args:
        config (Config): 配置参数
        local_rank (int): 当前GPU编号
        seed (int): 随机种子，用于控制数据生成的可重复性
    Returns:
        Dataset: Hugging Face Dataset 对象（train 部分）
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)

    # 定义高维正态分布的参数
    image_size = config.image_size
    # dim = 3 * image_size * image_size  # 确保维度可以被 reshape 为图像格式
    dim = image_size * image_size
    num_samples_1 = config.num1 // image_size # 数据集的样本数量
    num_samples_2 = config.num2 // image_size

    # 第一个数据集的均值和协方差矩阵
    mu1 = torch.ones(dim) * config.mu1  # 均值
    sigma1 = torch.eye(dim) * config.sigma2  # 协方差矩阵

    # 第二个数据集的均值和协方差矩阵
    mu2 = torch.ones(dim) * config.mu2  # 均值
    sigma2 = torch.eye(dim) * config.sigma2  # 协方差矩阵

    # 从高维正态分布中生成数据
    data1 = torch.distributions.MultivariateNormal(mu1, sigma1).sample((num_samples_1,))
    data2 = torch.distributions.MultivariateNormal(mu2, sigma2).sample((num_samples_2,))

    # 将 data1 和 data2 reshape 成 10x10 的矩阵
    # data1_reshaped = data1.view(num_samples_1, image_size, image_size)
    # data2_reshaped = data2.view(num_samples_2, image_size, image_size)

    # 如果你想保持原始数据的连续性，可以使用 reshape 方法
    data1_reshaped = data1.reshape(num_samples_1, image_size, image_size)
    data2_reshaped = data2.reshape(num_samples_2, image_size, image_size)

    # 将两个数据集合并
    data = torch.cat([data1_reshaped, data2_reshaped], dim=0)
    labels = torch.cat([torch.zeros(num_samples_1), torch.ones(num_samples_2)])

    # 将数据 reshape 为通道数为1的格式
    data = data.view(-1, 1, image_size, image_size)

    # 创建 Hugging Face Dataset
    dataset = Dataset.from_dict({
        "image": data,  # 将数据转换为列表形式
        "label": labels.tolist()  # 将标签转换为列表形式
    })

    # 划分训练集和测试集
    dataset = dataset.train_test_split(test_size=0.1)
    dataset.set_format(type='torch', columns=['image'])
    # print("After set_format:", dataset["train"][0]["image"].shape)
    
    # 返回训练集部分
    return dataset["train"]

@torch.no_grad()
def generate_during_training_simulation_dif(model_engine, save_dir, config, epoch, num_images=16, mu1=-0.3, mu2=0.5, sigma1=0.95, sigma2=0.95, p=0.5):
    """在训练过程中生成样本并保存
    Args:
        model_engine: DeepSpeed 模型引擎
        save_dir (str): 保存样本的目录
        config: 配置对象
        num_images (int): 生成的样本数量
    """
    model_engine.eval()
    device = model_engine.device
    
    # 准备扩散参数
    betas = linear_beta_schedule(config.timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_over_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    """ 采样来自混合高斯分布的噪声 """
    b = torch.bernoulli(torch.full((num_images, 1, 1, 1), p, device=device))  # 选择哪个高斯分布
    noise1 = torch.randn(num_images, 1, config.image_size, config.image_size, device=device, dtype=torch.half) * sigma1 + mu1
    noise2 = torch.randn(num_images, 1, config.image_size, config.image_size, device=device, dtype=torch.half) * sigma2 + mu2
    x = b * noise1 + (1 - b) * noise2
    
    x = x.to(next(model_engine.parameters()).dtype)
    
    # 反向扩散过程
    for t in reversed(range(0, config.timesteps)):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        pred_noise = model_engine(x, t_batch)
        
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        # noise = torch.randn_like(x) if t > 0 else 0
        noise = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        
        # 更新公式
        x = sqrt_one_over_alphas[t] * (x - beta_t * pred_noise / sqrt_one_minus_alphas_cumprod[t]) + torch.sqrt(beta_t) * noise
        
        del pred_noise
        torch.cuda.empty_cache()

    x = x.to(torch.float32)  # 最终转换为 float32 以便保存和处理
    # print("----****x2 in epoch ", epoch, "****----", x)
    
    with open(os.path.join(save_dir, "samples.txt"), "w") as f:
        res_list = []
        for i in range(num_images):
            # sample = x[i].cpu().numpy()
            sample = x[i].cpu().detach().numpy()
            for row in sample:
                flattened_row = row.flatten().tolist()
                for item in flattened_row:
                    res_list.append(item)
        f.write(str(res_list))
    
    # 确定横坐标范围
    min_value = min(res_list)
    max_value = max(res_list)
    l = len(res_list)
    with open(os.path.join(save_dir, "range.txt"), "w") as f:
        content = 'min: ' + str(min_value) + ', max: ' + str(max_value) + ', len: ' + str(l)
        f.write(content)
    bins = np.arange(min_value, max_value + 1, 1)  # 左闭右开区间
    
    # 计算频次
    hist, bin_edges = np.histogram(res_list, bins=bins)
    
    # 绘制柱状图
    plt.bar(bin_edges[:-1], hist, width=0.8, align='edge', edgecolor='black')

    # 设置图表标题和坐标轴标签
    plt.title("Frequency Distribution")
    plt.xlabel("Value Range")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, "samples.png"))

# -------------------- 训练循环 --------------------
# -------------------- 训练循环 --------------------
def train_deepspeed(config):
    
    """DeepSpeed训练主函数"""
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    # 创建带时间戳的 checkpoints, sample 和 logs 文件夹
    config.checkpoints_dir = os.path.join(config.checkpoints_dir, f"checkpoints_{timestamp}")
    config.samples_dir = os.path.join(config.samples_dir, f"samples_{timestamp}")
    config.logs_dir = os.path.join(config.logs_dir, f"logs_{timestamp}")
    
    # 初始化模型
    model = UNetSimulation(time_emb_dim=config.time_emb_dim, image_size=config.image_size)
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
    
    # 初始化引擎
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    train_dataset = load_data_normal(config, local_rank)
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
            # print(images.shape)  # 应该是 [B, 1, H, W]
            t = torch.randint(0, config.timesteps, (images.size(0),)).to(model_engine.device)
            p = config.num1 / (config.num1 + config.num2)
            # p = 0.5
            # print('----****p****----', p)
            
            # 前向扩散
            noisy_images, noise = forward_diffusion_with_different_noise(
                images, t,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
                p=p
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
                
                generate_during_training_simulation_dif(model_engine, sample_dir, config, epoch, num_images=config.num_images//config.image_size, p=p)
                # generate_during_training(model_engine, sample_dir, config, num_images=config.num_images)
            
            # print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Samples saved to {sample_dir}")

# -------------------- 主函数 --------------------
if __name__ == "__main__":
    cfg = get_config() 

    if cfg.mode == "train":
        train_deepspeed(cfg)