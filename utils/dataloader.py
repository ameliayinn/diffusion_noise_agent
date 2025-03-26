# dataloader.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset, Dataset
from torch.utils.data.distributed import DistributedSampler

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

def load_data_poisson(config, local_rank, seed=42):
    """加载并生成泊松分布模拟数据集
    Args:
        config (Config): 配置参数
        local_rank (int): 当前GPU编号
        seed (int): 随机种子，用于控制数据生成的可重复性
    Returns:
        Dataset: Hugging Face Dataset 对象（train 部分）
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)

    # 定义泊松分布的参数
    image_size = config.image_size
    dim = image_size * image_size
    num_samples_1 = config.num1 // image_size  # 数据集的样本数量
    num_samples_2 = config.num2 // image_size

    # 第一个数据集的泊松分布参数
    lambda1 = config.lambda1  # 泊松分布的参数 lambda

    # 第二个数据集的泊松分布参数
    lambda2 = config.lambda2

    # 从泊松分布中生成数据
    data1 = torch.poisson(torch.ones((num_samples_1, dim)) * lambda1)
    data2 = torch.poisson(torch.ones((num_samples_2, dim)) * lambda2)

    # 将 data1 和 data2 reshape 成 image_size x image_size 的矩阵
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
    
    # 返回训练集部分
    return dataset["train"]