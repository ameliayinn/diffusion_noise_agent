import torch
from datasets import Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib.pyplot as plt

seed = 42

# 设置随机种子以确保可重复性
torch.manual_seed(seed)

# 定义高维正态分布的参数
image_size = 8
# dim = 3 * image_size * image_size  # 确保维度可以被 reshape 为图像格式
# dim = image_size * image_size
dim = image_size
num_samples_1 = 9900 # 数据集的样本数量
num_samples_2 = 100
p = num_samples_1 / (num_samples_1 + num_samples_2)

mu_1 = 4
sigma_1 = 1
mu_2 = 10
sigma_2 = 4

# 第一个数据集的均值和协方差矩阵
mu1 = torch.ones(dim) * mu_1  # 均值
sigma1 = torch.eye(dim) * sigma_1  # 协方差矩阵

# 第二个数据集的均值和协方差矩阵
mu2 = torch.ones(dim) * mu_2  # 均值
sigma2 = torch.eye(dim) * sigma_2  # 协方差矩阵

# 从高维正态分布中生成数据
data1 = torch.distributions.MultivariateNormal(mu1, sigma1).sample((num_samples_1,))
data2 = torch.distributions.MultivariateNormal(mu2, sigma2).sample((num_samples_2,))

# 将两个数据集合并
data = torch.cat([data1, data2], dim=0)
labels = torch.cat([torch.zeros(num_samples_1), torch.ones(num_samples_2)])

# 将所有生成的数值拉平并保存到一个一维列表中
flattened_data = data.flatten().tolist()
'''
res_list = []
for item in data:
    mean = np.mean(item.numpy())
    res_list.append(mean)
'''

# 将一维列表写入到 txt 文件中
with open(f"tools/normal_data_{mu_1}_{sigma_1}_{mu_2}_{sigma_2}_0{int(p*100)}.txt", "w") as f:
    f.write(str(flattened_data))
    # f.write(str(res_list))

# 确定横坐标范围
min_value = min(flattened_data)
max_value = max(flattened_data)
l = len(flattened_data)
# bins = np.arange(min_value, max_value + 1, 1)  # 左闭右开区间
bins = 100
alpha = 0.4

# 计算频次
hist, bin_edges = np.histogram(flattened_data, bins=bins)

# 绘制柱状图
plt.hist(flattened_data, bins=bins, alpha=alpha, color='green', label=f'output (total {len(data1)})', density=True)

# 设置图表标题和坐标轴标签
plt.title("Frequency Distribution")
plt.xlabel("Value Range")
plt.ylabel("Frequency")
plt.savefig(f"tools/normal_data_{mu_1}_{sigma_1}_{mu_2}_{sigma_2}_0{int(p*100)}.png")