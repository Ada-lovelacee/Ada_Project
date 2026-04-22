"""
Federated Learning for Diabetes Prediction by Ada, April 2026.

Centralized Training
功能：使用分配得到的训练集数据进行Centralized训练，并在分配得到的测试集上评估性能.
使用场景：（1）在各个Client端独立Centralized训练评估；
         （2）汇总数据进行Centralized训练和评估；
         （1）和（2）供与联邦学习做对比用。
"""

import torch
from sklearn.metrics import roc_auc_score

import csv, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_and_simulation import load_csv_data, get_dataloaders, CSV_FILENAME, TRAIN_PCT,VALID_PCT,TEST_PCT
from model import SimpleMLP, BATCH_SIZE, LR, EPOCHS, test_model


# 固定随机种子保证可复现
np.random.seed(42)
torch.manual_seed(42)

       
DATASET_SPLIT = TRAIN_PCT + VALID_PCT  # 80%Train，10%Validation，10%Test。与FL一致？


# 日志初始化
def init_central_log(location):   
    log_filename = f"./results/{location}_centralized_metrics.csv"
    with open(log_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_accuracy", "test_auc"])
    return log_filename

# 绘图
def plot_central_metrics(location):
    df = pd.read_csv(f"./results/{location}_centralized_metrics.csv")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6), dpi=300)

    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o', color='blue')
    ax1.plot(df["epoch"], df["test_loss"], label="Test Loss", marker='s', color='red')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Centralized Loss Trend")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df["epoch"], df["test_accuracy"], label="Test Accuracy", marker='^', color='green')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Centralized Accuracy Trend")
    ax2.legend()
    ax2.grid(True)

    # 绘制AUC曲线
    ax3.plot(df["epoch"], df["test_auc"], label="Test AUC", marker='+', color='red')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("AUC")
    ax3.set_title("Centralized AUC Trend")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f"./results/{location}_centralized_metrics.png",dpi=300)
    plt.close()
    print(f"Ada tell you ====== Centralized Learning metrics saved in {location}_centralized_metrics.png")

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
    # 用于计算 AUC
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data, labels in loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total_samples += data.size(0)
            # 计算auc用：获取阳性1的概率
            # probs = torch.softmax(outputs, dim=1)[:, 1]  # 二分类取1的概率
            all_probs.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())

    avg_loss = total_loss / total_samples
    acc = correct / total_samples
    # 计算 AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5  # 异常保底
    return avg_loss, acc, auc

def init_data_for_server_centralized_train():
    # 1. 加载全部数据
    X, y, _, _ = load_csv_data(CSV_FILENAME)
    dataset = list(zip(X, y))

    # 2. 全部数据切分训练集/测试集
    train_size = int(len(dataset) * DATASET_SPLIT)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 3. 生成 DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

# 主函数
def run_centralized_training(location, train_loader, test_loader):
    # location: client_0, client_1, ..., client_n, 或者 server代表总的centralized训练

    # 4. 模型、损失、优化器（和联邦完全一致）
    model = SimpleMLP(input_dim=14, hidden_dim=16, num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 5. 日志
    log_file = init_central_log(location)

    # 6. 训练
    print("=============== Ada tell you: Centralized Learning Training Beginning ===============")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc, test_auc = test_model(model, test_loader, criterion)  # 使用model.py中的test_model

        print(f"Ada tell you ====== Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Auc: {test_auc:.4f}")

        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, test_loss, test_acc, test_auc])

    # 7. 绘图
    plot_central_metrics(location)
    print("\n============ Ada's Tips: Centralized Learning Completed Successfully ============")

    return model


if __name__ == "__main__":
    train_loader, test_loader = init_data_for_server_centralized_train()
    cen_model_server = run_centralized_training('server', train_loader, test_loader)

    # 完成server Centralized训练后，用Test数据完成一次测试
    _, _, X_Test, y_Test = load_csv_data(CSV_FILENAME)
    Test_dataset = list(zip(X_Test, y_Test))
    Test_loader = torch.utils.data.DataLoader(Test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    loss_Test, acc_Test, auc_Test = test_model(cen_model_server,Test_loader,torch.nn.CrossEntropyLoss())
    with open('./results/metrics_On_Test.csv', 'a', newline='', encoding='utf-8') as f:    # 追加一行
        writer = csv.writer(f)
        writer.writerow(['server centralized', loss_Test, acc_Test, auc_Test])
    print(f'Ada tell you ====== Server Centralized model metrics On TestData is: loss={loss_Test:.4f} | acc={acc_Test:.4f} | auc={auc_Test:.4f}')
