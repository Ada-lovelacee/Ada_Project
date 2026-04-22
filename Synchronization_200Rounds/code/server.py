"""
Federated Learning for Diabetes Prediction by Ada, April 2026.
"""
import flwr
from flwr.server.strategy import FedAvg
from flwr.common import Metrics

from flwr.common import parameters_to_ndarrays

import csv, os
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
from data_and_simulation import NUM_CLIENTS, NUM_ROUNDS
from data_and_simulation import load_csv_data, CSV_FILENAME
from model import save_global_model, test_model, SimpleMLP, BATCH_SIZE

# 服务端日志初始化
def init_server_log():
    """初始化服务端CSV日志文件，写入表头"""
    log_filename = "./results/server_global_metrics.csv"
    with open(log_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["round", "global_loss", "global_acc", "global_auc"])
    return log_filename

server_log_filename = init_server_log()
current_round = 0

def plot_server_metrics():
    """绘制服务端全局loss\accuracy\auc趋势图"""
    df = pd.read_csv(server_log_filename)
    
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6), dpi=300)
    
    # 绘制Global Loss曲线
    ax1.plot(df["round"], df["global_loss"], label="Global Loss", marker='o', color='red')
    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Loss")
    ax1.set_title("Server Global Loss Trend")
    ax1.legend()
    ax1.grid(True)
    
    # 绘制Global Accuracy曲线
    ax2.plot(df["round"], df["global_acc"], label="Global Accuracy", marker='^', color='green')
    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Server Global Accuracy Trend")
    ax2.legend()
    ax2.grid(True)

    # 绘制Global Auc曲线
    ax3.plot(df["round"], df["global_auc"], label="Global Auc", marker='+', color='red')
    ax3.set_xlabel("Federated Round")
    ax3.set_ylabel("Auc")
    ax3.set_title("Server Global Auc Trend")
    ax3.legend()
    ax3.grid(True)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(f"./results/server_global_metrics.png",dpi=300)
    plt.close()
    print("\n Ada tell you ====== server globle metrics saved in server_global_metrics.png")

def plot_metrics_on_Test():
    """根据metrics_On_Test.csv绘制loss,acc,auc在不同情形之间的比较曲线"""
    df = pd.read_csv('results/metrics_On_Test.csv')

    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6), dpi=300)

    # 绘制 Loss
    ax1.plot(df["location"], df["loss"], marker='o', color='red')
    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("loss")
    ax1.set_title("loss on Test Dataset")
    ax1.grid(True,alpha=0.2)
    ax1.set_xticklabels([])
    for x, y, label in zip(df.location, df.loss, df.location):
        ax1.text(x, y,          # 标注位置
                str(label),    
                ha='center',   
                va='bottom',   
                fontsize=6,    
                color='black')   
    # 绘制Accuracy
    ax2.plot(df["location"], df["acc"], marker='o', color='green')
    ax2.set_xlabel("Scenario")
    ax2.set_xticklabels([])
    ax2.set_ylabel("acc")
    ax2.set_title("acc on Test Dataset")
    ax2.grid(True,alpha=0.2)
    for x, y, label in zip(df.location, df.acc, df.location):
        ax2.text(x, y,          # 标注位置
                str(label),    
                ha='center',   
                va='bottom',   
                fontsize=6,   
                color='black')   
    # 绘制Auc
    ax3.plot(df["location"], df["auc"], marker='o', color='blue')
    ax3.set_xlabel("Scenario")
    ax3.set_xticklabels([])
    ax3.set_ylabel("auc")
    ax3.set_title("auc on Test Dataset")
    ax3.grid(True,alpha=0.2)
    for x, y, label in zip(df.location, df.auc, df.location):
        ax3.text(x, y,          # 标注位置
                str(label),    
                ha='center',   
                va='bottom',   
                fontsize=6,   
                color='black')   
    # 保存图片
    plt.tight_layout()
    plt.savefig(f"./results/metrics_On_Test.png",dpi=300)
    plt.close()
    print("\n Ada tell you ====== metrics On Test Dataset saved in metrics_On_Test.png")


def eval_FL_model_on_Test(state_dict):
    """在load_csv_data()中分出的独立Test上测试FL后的模型
    使用第round轮FL训练好并保存起来的模型"""
    # 加载FL后的模型
    model = SimpleMLP()
    model.load_state_dict(state_dict)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # 独立Test数据集的data_loader，
    # 和data_simulation.py中的get_dataloaders(dataset, batch_size=16)一致
    _, _, X_Test, y_Test = load_csv_data(CSV_FILENAME)
    Test_dataset = TensorDataset(X_Test, y_Test)
    Test_loader = DataLoader(Test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    loss, acc, auc = test_model(model, Test_loader, criterion)

    print(f"Ada tell you ====== 在独立Test数据集上，FL训练好的模型的： Loss - {loss:.4f}, Accuracy - {acc:.4f}, AUC - {auc:.4f} ")

# 定义：server端如何计算average accuracy和average loss和acu
def weighted_avg(metrics):
    global current_round
    current_round += 1
    
    # 加权计算全局loss
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    global_loss = sum(losses) / sum(examples)
    
    # 加权计算全局accuracy
    accuracies = [num_examples * m["acc"] for num_examples, m in metrics]
    global_acc = sum(accuracies) / sum(examples)

    # 加权计算全局acu
    aucs = [num_examples * m["auc"] for num_examples, m in metrics]
    global_auc = sum(aucs) / sum(examples)
    
    # 实时打印
    print(f"\n====== Ada show: Server Round {current_round} Global Metrics ======")
    print(f"Global Loss: {global_loss:.4f} | Global Accuracy: {global_acc:.4f} | Global Auc: {global_auc:.4f}")
    
    # 写入CSV
    with open(server_log_filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([current_round, global_loss, global_acc, global_auc])
    
    return {"loss": global_loss, "acc": global_acc, "auc": global_auc}

# 自定义策略：自动保存最终全局模型
class FedAvg_save_model(flwr.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_parameters = None      # 最终模型存这里

    def aggregate_fit(self, server_round, results, failures):
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_params is not None:
            self.final_parameters = aggregated_params  # 保存每一轮参数
        return aggregated_params, metrics

# 加载模型参数
def set_model_params(model, parameters):
    params_list = flwr.common.parameters_to_ndarrays(parameters)
    params_dict = zip(model.state_dict().keys(), params_list)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

# 配置FedAvg策略
strategy = FedAvg_save_model(
    fraction_fit=1.0,                     # 每轮用100%客户端训练
    fraction_evaluate=1.0,                # 每轮用100%客户端评估
    min_fit_clients=NUM_CLIENTS,          # 最少NUM_CLIENTS个client参与训练
    min_evaluate_clients=NUM_CLIENTS,     # 最少NUM_CLIENTS个client参与评估
    min_available_clients=NUM_CLIENTS,    # 必须启动NUM_CLIENTS个client才开始
    evaluate_metrics_aggregation_fn=weighted_avg,     # 自定义全局指标聚合
)


# 启动服务端
if __name__ == "__main__":
    print("Ada tell you ====== Federated Learning Server Running")

    # init
    with open('./results/metrics_On_Test.csv', 'w', newline='', encoding='utf-8') as f: 
        writer = csv.writer(f)
        writer.writerow(['location','loss','acc','auc'])

    # 启动服务端训练
    flwr.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=flwr.server.ServerConfig(num_rounds=NUM_ROUNDS)  # 训练轮数
    )
    print("Ada tell you ====== Federated Learning Server Completed.")

    # FL训练结束后绘制服务端指标图
    plot_server_metrics()

    # FL训练后直接获取全局模型
    FL_model = SimpleMLP()
    set_model_params(FL_model, strategy.final_parameters)

    # 完成FL训练后，用FL model 在 Test 数据上完成一次测试
    _, _, X_Test, y_Test = load_csv_data(CSV_FILENAME)
    Test_dataset = list(zip(X_Test, y_Test))
    Test_loader = torch.utils.data.DataLoader(Test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    loss_Test, acc_Test, auc_Test = test_model(FL_model,Test_loader,torch.nn.CrossEntropyLoss())
    with open('./results/metrics_On_Test.csv', 'a', newline='', encoding='utf-8') as f:    # 追加一行
        writer = csv.writer(f)
        writer.writerow(['FL', loss_Test, acc_Test, auc_Test])
    print(f'Ada tell you ====== Ferdered Learning model metrics On TestData is: loss={loss_Test:.4f} | acc={acc_Test:.4f} | auc={auc_Test:.4f}')
    # 绘制在TestData上的各情形的metrics比较图
    plot_metrics_on_Test()

    # print("\n Ada tell you ====== 保存联邦学习好的模型 ======")
    # state_dict = result.arrays.to_torch_state_dict()
    # torch.save(state_dict, "final_model.pt")
    #在独立的Test数据上测试训练好的FL模型  Ada: ToDo==========================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #eval_FL_model_on_Test(state_dict)