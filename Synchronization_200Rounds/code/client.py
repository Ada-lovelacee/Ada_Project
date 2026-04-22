"""
Federated Learning for Diabetes Prediction by Ada, April 2026.
"""
import torch, flwr

import sys, os, csv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from data_and_simulation import load_csv_data, dirichlet_partition_data, get_dataloaders, NUM_CLIENTS, DIRICHLET_ALPHA, CSV_FILENAME
from model import SimpleMLP, train_model, test_model, BATCH_SIZE, LR, EPOCHS 

import centralized_learning as CL


# 全局加载数据（只加载一次）
X, y, X_Test, y_Test = load_csv_data(CSV_FILENAME)
client_datasets = dirichlet_partition_data(X, y, NUM_CLIENTS, DIRICHLET_ALPHA)

# 客户端日志记录初始化
def init_client_log(cid):
    """初始化客户端CSV日志文件，写入表头"""
    log_filename = f"./results/client_{cid}_metrics.csv"
    with open(log_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["round", "loss", "acc", "auc"])
    return log_filename

def plot_client_metrics(cid):
    """绘制客户端loss，accuracy， auc趋势图"""
    log_filename = f"./results/client_{cid}_metrics.csv"
    df = pd.read_csv(log_filename)
    
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6), dpi=300)
    
    # 绘制Loss曲线
    ax1.plot(df["round"], df["loss"], label="Loss", marker='o', color='blue')
    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Client {cid} Loss Trend")
    ax1.legend()
    ax1.grid(True)
    
    # 绘制Accuracy曲线
    ax2.plot(df["round"], df["acc"], label="acc", marker='^', color='green')
    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("acc")
    ax2.set_title(f"Client {cid} acc Trend")
    ax2.legend()
    ax2.grid(True)

    # 绘制AUC曲线
    ax3.plot(df["round"], df["auc"], label="auc", marker='+', color='red')
    ax3.set_xlabel("Federated Round")
    ax3.set_ylabel("auc")
    ax3.set_title(f"Client {cid} auc Trend")
    ax3.legend()
    ax3.grid(True)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(f"./results/client_{cid}_metrics.png", dpi=300)
    plt.close()
    print(f"Ada tell you ====== Client {cid} metrics saved in file: client_{cid}_metrics.png")

def train_model(model, train_loader, optimizer, criterion, epochs=1):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for _ in range(epochs):
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    avg_train_loss = total_loss / total_samples
    return avg_train_loss

# 定义Flower客户端
class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, cid, train_loader, test_loader, model):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        self.log_filename = init_client_log(cid)
        self.round = 0  # 记录联邦轮数

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.round += 1
        self.set_parameters(parameters)
        # 训练
        train_model(self.model, self.train_loader, self.optimizer, self.criterion, EPOCHS)
        # 评估获取测试loss、准确率、auc
        # test_loss, test_acc, test_auc = test_model(self.model, self.test_loader, self.criterion)
          
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc, auc = test_model(self.model, self.test_loader, self.criterion)

        # 实时打印
        print(f"\n=== Client {self.cid} Round {self.round} ===")
        print(f"loss: {loss:.4f} | acc: {acc:.4f} | auc: {auc:.4f}")   
        # 写入csv
        with open(self.log_filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([self.round, loss, acc, auc])
 
        return float(loss), len(self.test_loader.dataset), {"acc": float(acc), "loss": float(loss), "auc": float(auc)}

# 启动客户端
def start_client(cid):
    # 分配对应client的数据
    dataset = client_datasets[int(cid)]
    train_loader, test_loader = get_dataloaders(dataset, BATCH_SIZE)
    
    # 初始化模型
    model = SimpleMLP(input_dim=14, hidden_dim=16, num_classes=2)
    
    # 启动Flower客户端
    client = FlowerClient(cid, train_loader, test_loader, model)
    flwr.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
    
    # 训练结束后绘图
    plot_client_metrics(cid)

    # 完成FL训练后，用分到的数据完成一次完整的Centralized训练和测试-CL
    cen_model = CL.run_centralized_training(f'client_{cid}', train_loader, test_loader)
    # 用Centralized模型在Test上评估性能
    Test_dataset = list(zip(X_Test, y_Test))
    Test_loader = torch.utils.data.DataLoader(Test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    loss_Test, acc_Test, auc_Test = test_model(cen_model,Test_loader,torch.nn.CrossEntropyLoss())
    with open('./results/metrics_On_Test.csv', 'a', newline='', encoding='utf-8') as f:    # 追加一行
        writer = csv.writer(f)
        writer.writerow([f'client_{cid} centralized', loss_Test, acc_Test, auc_Test])
    print(f'Ada tell you ====== On client_{cid}, centralized model metrics On TestData is: loss={loss_Test:.4f} | acc={acc_Test:.4f} | auc={auc_Test:.4f}')


if __name__ == "__main__":
    # 在各个终端窗口模拟各个client
    if len(sys.argv) < 2:
        print("Ada ask you ====== pls assign client ID, for example:  python client.py 0")
        sys.exit(1)
    start_client(sys.argv[1])

