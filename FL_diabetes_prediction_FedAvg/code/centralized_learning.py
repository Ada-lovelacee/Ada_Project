"""
Federated Learning for Diabetes Prediction by Ada, April 2026.

Centralized Training
使用全部数据训练，供与联邦学习做对比用：模型、超参、训练逻辑与联邦客户端一致
"""

import torch

import csv, os
import pandas as pd
import matplotlib.pyplot as plt

from data_and_simulation import load_csv_data, get_dataloaders, CSV_FILENAME
from model import SimpleMLP

# ===================== 超参（和联邦客户端一样）=====================
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 10          
DATASET_SPLIT = 0.8  # 80%训练，20%测试
# =====================================================================

# 日志初始化
def init_central_log():
    log_filename = "./results/centralized_metrics.csv"
    with open(log_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_accuracy"])
    return log_filename

# 绘图
def plot_central_metrics():
    df = pd.read_csv("./results/centralized_metrics.csv")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o', color='blue')
    ax1.plot(df["epoch"], df["test_loss"], label="Test Loss", marker='s', color='red')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Centralized Training Loss Trend")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df["epoch"], df["test_accuracy"], label="Test Accuracy", marker='^', color='green')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Centralized Training Accuracy Trend")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("./results/centralized_metrics.png")
    plt.close()
    print("Ada tell you ====== Centralized Learning metrics saved in centralized_metrics.png")

# 训练一个 epoch
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for data, labels in loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
    return total_loss / total_samples

# 评估
def test(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, labels in loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total_samples += data.size(0)
    avg_loss = total_loss / total_samples
    acc = correct / total_samples
    return avg_loss, acc

# 主函数
def run_centralized_training():
    # 1. 加载全部数据
    X, y = load_csv_data(CSV_FILENAME)
    dataset = list(zip(X, y))

    # 2. 全部数据切分训练集/测试集
    train_size = int(len(dataset) * DATASET_SPLIT)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 3. 生成 DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 模型、损失、优化器（和联邦完全一致）
    model = SimpleMLP(input_dim=14, hidden_dim=16, num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 5. 日志
    log_file = init_central_log()

    # 6. 训练
    print("=== Ada tell you: Centralized Learning Training Beginning ===")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, test_loader, criterion)

        print(f"Ada tell you ====== Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")

        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, test_loss, test_acc])

    # 7. 绘图
    plot_central_metrics()
    print("\n=== Ada's Tips: Centralized Learning Completed Successfully ===")

if __name__ == "__main__":
    run_centralized_training()