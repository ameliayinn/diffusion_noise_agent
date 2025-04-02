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
num_samples_1 = 8000 # 数据集的样本数量
num_samples_2 = 2000
p = num_samples_1 / (num_samples_1 + num_samples_2)

# 第一个数据集的泊松分布参数
lambda1 = 1  # 泊松分布的参数 lambda

# 第二个数据集的泊松分布参数
lambda2 = 5

# 从泊松分布中生成数据
data1 = torch.poisson(torch.ones((num_samples_1, dim)) * lambda1)
data2 = torch.poisson(torch.ones((num_samples_2, dim)) * lambda2)

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
with open(f"tools/poisson_data_{lambda1}_{lambda2}_0{int(p*100)}.txt", "w") as f:
    f.write(str(flattened_data))
    # f.write(str(res_list))

# 确定横坐标范围
min_value = min(flattened_data)
max_value = max(flattened_data)
l = len(flattened_data)
bins = np.arange(min_value, max_value + 1, 1)  # 左闭右开区间
alpha = 0.4

# 计算频次
hist, bin_edges = np.histogram(flattened_data, bins=bins)

# 绘制柱状图
plt.bar(bin_edges[:-1], hist, width=0.8, align='edge', edgecolor='black')

'''
# 绘制柱状图
plt.hist(flattened_data, bins=bins, alpha=alpha, color='green', label=f'output (total {len(flattened_data)})', density=True)
'''

# 设置图表标题和坐标轴标签
plt.title("Frequency Distribution")
plt.xlabel("Value Range")
plt.ylabel("Frequency")
plt.savefig(f"tools/poisson_data_{lambda1}_{lambda2}_0{int(p*100)}.png")