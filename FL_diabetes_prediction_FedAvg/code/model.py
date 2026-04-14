"""
Federated Learning for Diabetes Prediction by Ada, April 2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnFunc

class SimpleMLP(nn.Module):
    """
    3层前馈神经网络
    输入：14维
    输出：2分类
    """
    def __init__(self, input_dim=14, hidden_dim=16, num_classes=2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = nnFunc.relu(self.fc1(x))
        x = nnFunc.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 训练/测试工具函数
def train_model(model, train_loader, optimizer, criterion, epochs=1):
    model.train()
    for _ in range(epochs):
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test_model(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    avg_loss = loss / len(test_loader)
    return avg_loss, acc