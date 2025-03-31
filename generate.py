# generate.py
import torch
import torchvision
import matplotlib.pyplot as plt
from unet import UNet
from diffusion import linear_beta_schedule
import os
import numpy as np

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

@torch.no_grad()
def generate_during_training(model_engine, save_dir, config, num_images=16):
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
    
    # 生成初始噪声
    x = torch.randn(num_images, 3, config.image_size, config.image_size, device=device, dtype=torch.half)
    x = x.to(next(model_engine.parameters()).dtype)
    
    # 反向扩散过程
    for t in reversed(range(0, config.timesteps)):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        pred_noise = model_engine(x, t_batch)
        
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        noise = torch.randn_like(x) if t > 0 else 0
        
        # 更新公式
        x = sqrt_one_over_alphas[t] * (x - beta_t * pred_noise / sqrt_one_minus_alphas_cumprod[t]) + torch.sqrt(beta_t) * noise
    
    # 后处理并转换数据类型
    x = (x.clamp(-1, 1) + 1) * 0.5  # 将图像范围从 [-1, 1] 转换到 [0, 1]
    x = x.to(torch.float32)  # 确保转换为 float32
    grid = torchvision.utils.make_grid(x.cpu(), nrow=4)  # 将图像拼接成网格
    
    # 保存图像
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).numpy())  # 数据现在是 float32
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, "samples.png"))
    plt.close()

@torch.no_grad()
def generate_during_training_simulation(model_engine, save_dir, config, num_images=16):
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
    
    # 生成初始噪声
    x = torch.randn(num_images, 1, config.image_size, config.image_size, device=device, dtype=torch.half)
    x = x.to(next(model_engine.parameters()).dtype)
    
    # 反向扩散过程
    for t in reversed(range(0, config.timesteps)):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        pred_noise = model_engine(x, t_batch)
        
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        noise = torch.randn_like(x) if t > 0 else 0
        
        # 更新公式
        x = sqrt_one_over_alphas[t] * (x - beta_t * pred_noise / sqrt_one_minus_alphas_cumprod[t]) + torch.sqrt(beta_t) * noise
    
    # 后处理并转换数据类型
    # x = (x.clamp(-1, 1) + 1) * 0.5  # 将图像范围从 [-1, 1] 转换到 [0, 1]
    x = x.to(torch.float32)  # 确保转换为 float32
    
    with open(os.path.join(save_dir, "samples.txt"), "w") as f:
        res_list = []
        for i in range(num_images):
            sample = x[i].cpu().numpy()
            mean_list = []
            for row in sample:
                # print("*****" ,type(row), row.shape) # <class 'numpy.ndarray'> (8, 8)
                # f.write(" ".join(f"{float(val):.7f}" for val in row.reshape(-1))) # flaten row to one dimension
                '''
                mean_row = np.mean(row)
                mean_list.append(mean_row)
                '''
                
                flattened_row = row.flatten().tolist()
                for item in flattened_row:
                    res_list.append(item)
                
            '''
            mean_res = np.mean(mean_list)
            res_list.append(mean_res)
            '''
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

    # 或者使用 Counter 计算频次
    # counter = Counter(res_list)
    # hist = [counter.get(i, 0) for i in range(min_value, max_value)]
    # bin_edges = np.arange(min_value, max_value + 1)
    
    # 绘制柱状图
    plt.bar(bin_edges[:-1], hist, width=0.8, align='edge', edgecolor='black')

    # 设置图表标题和坐标轴标签
    plt.title("Frequency Distribution")
    plt.xlabel("Value Range")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, "samples.png"))

def generate_during_training_simulation_dif(model_engine, save_dir, config, epoch, num_images=16):
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
    # 方案1：完全随机初始化 + 依赖模型去噪
    # x = torch.randn(num_images, 1, config.image_size, config.image_size, device=device)
    
    # 方案2（可选）：若需保留部分手动控制，可通过config传递参数
    x = model_engine.module.sample_initial_noise(num_images, config) 
    
    x = x.to(next(model_engine.parameters()).dtype)
    
    # 创建参数记录文件（如果不存在）
    params_csv_path = os.path.join(save_dir, "distribution_params.csv")
    if not os.path.exists(params_csv_path):
        with open(params_csv_path, 'w') as f:
            f.write("epoch,timestep,expert_idx,mu,sigma\n")  # 表头
    
    # 反向扩散过程
    for t in reversed(range(0, config.timesteps)):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        pred_noise = model_engine(x, t_batch)
        
        # --- 新增：提取并保存分布参数 ---
        if t % 100 == 0:  # 每100个timestep保存一次
            # 获取当前MoE层的参数
            moe = model_engine.module.moe
            dummy_input = torch.zeros(1, moe.input_dim, device=device)
            
            # 提取所有专家的mu和logvar
            mus = torch.stack([expert(dummy_input) for expert in moe.expert_mu]).squeeze(1)  # [num_experts, input_dim]
            logvars = torch.stack([expert(dummy_input) for expert in moe.expert_logvar]).squeeze(1)
            sigmas = torch.exp(0.5 * logvars)
            
            # 写入CSV
            with open(params_csv_path, 'a') as f:
                for expert_idx in range(moe.num_experts):
                    for dim in range(moe.input_dim):
                        f.write(
                            f"{epoch},{t},{expert_idx},"
                            f"{mus[expert_idx, dim].item():.6f},"
                            f"{sigmas[expert_idx, dim].item():.6f}\n"
                        )
        # ------------------------------
        
        # 更新x
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
    
    '''
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
    '''
    
    with open(os.path.join(save_dir, "samples.txt"), "w") as f:
        res_list = x.cpu().detach().numpy().flatten().tolist()
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

@torch.no_grad()
def generate_during_training_simulation_dif_1(model_engine, save_dir, config, epoch, num_images=16, mu1=-0.3, mu2=0.5, sigma1=0.95, sigma2=0.95, p=0.5):
    """在训练过程中生成样本并保存
    Args:
        model_engine: DeepSpeed 模型引擎
        save_dir (str): 保存样本的目录
        config: 配置对象
        num_images (int): 生成的样本数量
        mu1 (float): 第一个正态分布的均值
        mu2 (float): 第二个正态分布的均值
        sigma1 (float): 第一个正态分布的标准差
        sigma2 (float): 第二个正态分布的标准差
        p (float): 从第一个分布采样的概率
    """
    model_engine.eval()
    device = model_engine.device
    
    # 准备扩散参数
    betas = linear_beta_schedule(config.timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_over_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    # p_dgp = config.num1 / (config.num1 + config.num2)
    # 生成一个随机数来决定从哪个分布采样
    mask = torch.rand(num_images, 1, 1, 1, device=device) < p
    
    # 从两个正态分布中采样噪声
    # x1 = torch.randn(num_images, 1, config.image_size, config.image_size, device=device) * config.sigma1 + config.mu1
    # x2 = torch.randn(num_images, 1, config.image_size, config.image_size, device=device) * config.sigma2 + config.mu2
    x1 = torch.randn(num_images, 1, config.image_size, config.image_size, device=device) * sigma1 + mu1
    x2 = torch.randn(num_images, 1, config.image_size, config.image_size, device=device) * sigma2 + mu2
    
    # 根据 mask 选择噪声
    x = torch.where(mask, x1, x2)
    # print("----****x1 in epoch ", epoch, "****----", x)
    
    '''
    # 生成初始噪声
    x = torch.randn(num_images, 1, config.image_size, config.image_size, device=device, dtype=torch.half)
    '''
    # 将噪声转换为模型参数的数据类型
    x = x.to(next(model_engine.parameters()).dtype)
    
    # 反向扩散过程
    for t in reversed(range(0, config.timesteps)):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        pred_noise = model_engine(x, t_batch)
        print("----****pred_noise in epoch ", epoch, "****----", pred_noise)
        
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        # 生成噪声（与训练时一致）
        if t > 0:
            # 生成一个随机数来决定从哪个分布采样
            mask = torch.rand(num_images, 1, 1, 1, device=device) < p
            
            # 从两个正态分布中采样噪声
            noise1 = torch.randn_like(x) * sigma1 + mu1
            noise2 = torch.randn_like(x) * sigma2 + mu2
            
            # 根据 mask 选择噪声
            noise = torch.where(mask, noise1, noise2)
        else:
            noise = 0  # 在最后一步（t=0）不加噪声
        
        # 更新公式
        # x = sqrt_one_over_alphas[t] * (x - beta_t * pred_noise / sqrt_one_minus_alphas_cumprod[t]) + torch.sqrt(beta_t) * noise
        x = sqrt_one_over_alphas[t] * (x - beta_t * pred_noise / sqrt_one_minus_alphas_cumprod[t])
        if t > 0:
            x = x + torch.sqrt(beta_t) * noise  # 只有 t > 0 时加噪
    print("----****x2 in epoch ", epoch, "****----", x)
    
    '''
    # 反向扩散过程
    for t in reversed(range(0, config.timesteps)):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        pred_noise = model_engine(x, t_batch)
        
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        noise = torch.randn_like(x) if t > 0 else 0
        
        # 更新公式
        x = sqrt_one_over_alphas[t] * (x - beta_t * pred_noise / sqrt_one_minus_alphas_cumprod[t]) + torch.sqrt(beta_t) * noise
    '''
    
    # 后处理并转换数据类型
    x = x.to(torch.float32)  # 确保转换为 float32
    
    with open(os.path.join(save_dir, "samples.txt"), "w") as f:
        res_list = []
        for i in range(num_images):
            sample = x[i].cpu().numpy()
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