import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import deepspeed

# ReparamGaussianMoE 模型
class ReparamGaussianMoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.gating_network = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        gate_logits = self.gating_network(x)
        gate_weights = F.softmax(gate_logits, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(-2), dim=-1)
        return output

# UNet 模型（简化版）
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_dim=64, use_moe=False):
        super().__init__()
        self.use_moe = use_moe
        if use_moe:
            self.moe_layer = ReparamGaussianMoE(in_channels, hidden_dim)
        self.conv1 = nn.Conv2d(hidden_dim if use_moe else in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        if self.use_moe:
            x = self.moe_layer(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# 训练逻辑
class DummyDataset(Dataset):
    def __init__(self, size=1000, img_size=(3, 32, 32)):
        self.data = torch.randn(size, *img_size)
        self.labels = torch.randn(size, *img_size)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(use_moe=True).to(device)
    model_engine, _, train_dataloader, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        training_data=DummyDataset(),
    )
    criterion = nn.MSELoss()
    epochs = 5
    for epoch in range(epochs):
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            pred = model_engine(x)
            loss = criterion(pred, y)
            model_engine.backward(loss)
            model_engine.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
