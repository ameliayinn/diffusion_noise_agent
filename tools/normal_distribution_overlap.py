import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import json

# 设置参数
mu1, sigma1 = -0.3, 0.95
mu2, sigma2 = 0.5, 0.95

# 计算重叠面积
x = np.linspace(-5, 5, 1000)
y1 = norm.pdf(x, mu1, sigma1)
y2 = norm.pdf(x, mu2, sigma2)

overlap = np.minimum(y1, y2)
overlap_area = np.trapz(overlap, x)

print(f"重叠面积: {overlap_area:.2%}")

# 将数据转换为字典格式
data = {
    "mu1": mu1,  # 将张量转换为列表
    "sigma1": sigma1,
    "mu2": mu2,
    "sigma2": sigma2,
    "overlap_area": overlap_area
}

# 将数据写入 JSONL 文件
with open("tools/normal_distribution_overlap.jsonl", "a") as f:  # 使用 "a" 模式追加到文件
    json.dump(data, f)  # 将字典转换为 JSON 格式
    f.write("\n")  # 写入换行符，确保下一行数据在新的一行

# 绘图
plt.plot(x, y1, label=f'N({mu1}, {sigma1}^2)')
plt.plot(x, y2, label=f'N({mu2}, {sigma2}^2)')
plt.fill_between(x, overlap, color='gray', alpha=0.5, label=f'重叠面积: {overlap_area:.2%}')
plt.legend()
plt.show()