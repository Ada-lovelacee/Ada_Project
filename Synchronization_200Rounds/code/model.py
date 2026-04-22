"""
Federated Learning for Diabetes Prediction by Ada, April 2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnFunc
from sklearn.metrics import roc_auc_score

# 部分模型超参
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 10


class SimpleMLP(nn.Module):
    """
    3层神经网络（一个隐藏层）
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

# def test_model(model, test_loader, criterion):
#     model.eval()
#     correct = 0
#     total = 0
#     loss = 0.0

#     # 用于计算 AUC
#     all_probs = []
#     all_labels = []

#     with torch.no_grad():
#         for data, labels in test_loader:
#             outputs = model(data)
#             loss += criterion(outputs, labels).item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             # 计算auc用：获取阳性1的概率
#             # probs = torch.softmax(outputs, dim=1)[:, 1]  # 二分类取1的概率
#             # probs = outputs.squeeze()
#             all_probs.extend(outputs.squeeze().cpu().numpy())
#             all_labels.extend(labels.squeeze().cpu().numpy())

#     acc = correct / total
#     avg_loss = loss / len(test_loader)
#     # 计算 AUC
#     try:
#         auc = roc_auc_score(all_labels, all_probs)
#     except:
#         auc = 0.5  # 异常保底
#     return avg_loss, acc, auc


def test_model(model, test_loader, criterion, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0  # 改名更清晰

    all_probs = []   # 预测为正类的概率（AUC用）
    all_labels = []  # 真实标签

    with torch.no_grad():
        for data, labels in test_loader:
            # gpu情形：数据移到设备
            # data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            
            # 累加 loss（每个样本的loss）
            loss = criterion(outputs, labels)
            total_loss += loss.item() * data.size(0)  # 乘batch大小
            
            # 计算 accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 二分类：获取正类概率（必须是概率！不能用logits）
            if outputs.shape[1] == 2:
                # 有2个输出：softmax后取第1类
                probs = torch.softmax(outputs, dim=1)[:, 1]
            else:
                # 单输出：sigmoid
                probs = torch.sigmoid(outputs).squeeze()

            # 保存概率和标签（用于AUC）
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 平均：总loss / 总样本数
    avg_loss = total_loss / total
    acc = correct / total

    # 计算 AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5

    return avg_loss, acc, auc


def save_global_model(parameters):
    """保存FL最终全局模型到文件"""
    model = SimpleMLP()
    params = []
    for p in parameters:
        params.append(torch.tensor(p))
    model.load_state_dict(dict(zip(model.state_dict().keys(), params)))
    torch.save(model.state_dict(), "results/final_global_model.pth")
    print("Ada tell you ====== 全局模型已保存到 results/final_global_model.pth")