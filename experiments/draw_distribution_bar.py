import matplotlib.pyplot as plt
import numpy as np
import ast

# 读取两个txt文件中的数据
def read_data(file_path):
    with open(file_path, 'r') as file:
        # 读取文件内容并解析为列表
        data = []
        for line in file:
            # 将字符串形式的列表转换为实际的列表
            line_data = ast.literal_eval(line.strip())
            data.extend(line_data)  # 将列表中的元素添加到数据中
    return data

# 文件路径
file1 = 'experiments/dif_noise_agent_1/2_2_5_4_099_dif.txt'
file2 = 'tools/normal_data_2_2_5_4_099.txt'
# file1 = 'experiments/dif_noise_4/1_10_080_dif.txt'
# file2 = 'tools/poisson_data_1_10_080.txt'

# 读取数据
data1 = read_data(file1)
data2 = read_data(file2)
print(len(data1), len(data2))

# 设置柱状图的参数
bins = 500  # 柱状图的柱子数量
alpha = 0.4  # 透明度

# 绘制柱状图
plt.hist(data1, bins=bins, alpha=alpha, color='green', label=f'output (total {len(data1)})', density=True)
plt.hist(data2, bins=bins, alpha=alpha, color='red', label=f'input (total {len(data2)})', density=True)

# 限定横坐标范围
# plt.xlim(-50, 100)  # 设置 x 轴范围为 -10 到 20

# 添加标题和标签
plt.title(f"{file1[30:-4]}")
# plt.title("4_1_10_4_090")
plt.xlabel('Value')
plt.ylabel('Density')

# 添加图例
plt.legend()
plt.savefig(f"{file1[:-4]}_density.png")
print(f"图片已保存")