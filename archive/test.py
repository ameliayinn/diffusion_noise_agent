import torch
import torch.nn as nn
import torch.nn.functional as F


class ReparamGaussianMoE(nn.Module):
    def __init__(self, input_dim, num_experts=4, hidden_dim=64, tau=0.1, flatten=True):
        """
        Args:
            input_dim: 输入特征维度
            num_experts: 专家数量
            hidden_dim: 专家网络的隐藏层维度
            tau: Gumbel-Softmax温度参数
            flatten: 是否展平输入（True支持矩阵输入，False支持一维向量输入）
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.tau = tau
        self.flatten = flatten  # 控制输入是否为矩阵

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

        # 每个专家预测高斯分布的参数
        self.expert_mu = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])

        self.expert_logvar = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        输入x:
            - 若flatten=True: 任意形状，但最后一维必须是input_dim（如[batch, seq_len, input_dim]）
            - 若flatten=False: 必须是2D张量 [batch, input_dim]
        输出: 与输入形状相同（除了最后一维可能变化，但此处保持input_dim）
        """
        # 检查输入形状
        assert x.shape[-1] == self.input_dim, f"Input last dim must be {self.input_dim}, got {x.shape[-1]}"

        original_shape = x.shape  # 保存原始形状

        # 展平处理（支持矩阵输入）
        if self.flatten:
            x_flat = x.view(-1, self.input_dim)  # [N, input_dim]
        else:
            # 确保输入是2D（[batch, input_dim]）
            assert x.dim() == 2, f"When flatten=False, input must be 2D, got {x.dim()}D"
            x_flat = x  # [batch, input_dim]

        N = x_flat.size(0)

        # 1. 门控网络生成专家概率
        gate_logits = self.gate(x_flat)  # [N, num_experts]

        # 2. 可微分采样专家权重（Gumbel-Softmax）
        expert_weights = F.gumbel_softmax(gate_logits, tau=self.tau, hard=False)  # [N, num_experts]

        # 3. 每个专家生成高斯参数
        mus = torch.stack([expert(x_flat) for expert in self.expert_mu], dim=1)  # [N, num_experts, input_dim]
        logvars = torch.stack([expert(x_flat) for expert in self.expert_logvar], dim=1)  # [N, num_experts, input_dim]

        # 4. 重参数化采样
        eps = torch.randn_like(mus)
        expert_samples = mus + eps * torch.exp(0.5 * logvars)  # [N, num_experts, input_dim]

        # 5. 加权组合采样结果
        expert_weights = expert_weights.unsqueeze(-1)  # [N, num_experts, 1]
        output_flat = (expert_weights * expert_samples).sum(dim=1)  # [N, input_dim]

        # 恢复原始形状
        if self.flatten:
            output = output_flat.view(*original_shape[:-1], self.input_dim)
        else:
            output = output_flat  # 保持[batch, input_dim]

        return output


# 使用示例
if __name__ == "__main__":
    # 参数设置
    input_dim = 32
    num_experts = 4

    # 测试矩阵输入（flatten=True）
    model_matrix = ReparamGaussianMoE(input_dim, flatten=True)
    x_matrix = torch.randn(4, 10, input_dim)  # [batch, seq_len, input_dim]
    output_matrix = model_matrix(x_matrix)
    print("矩阵输入输出形状:", output_matrix.shape)  # [4, 10, 32]

    # 测试向量输入（flatten=False）
    model_vector = ReparamGaussianMoE(input_dim, flatten=False)
    x_vector = torch.randn(4, input_dim)  # [batch, input_dim]
    output_vector = model_vector(x_vector)
    print("向量输入输出形状:", output_vector.shape)  # [4, 32]