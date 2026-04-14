"""
Federated Learning for Diabetes Prediction by Ada, April 2026.
"""
import flwr
from flwr.server.strategy import FedAvg
from flwr.common import Metrics

import csv, os
import matplotlib.pyplot as plt
import pandas as pd

from data_and_simulation import NUM_CLIENTS, NUM_ROUNDS

# 服务端日志初始化
def init_server_log():
    """初始化服务端CSV日志文件，写入表头"""
    log_filename = "./results/server_global_metrics.csv"
    with open(log_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["round", "global_loss", "global_accuracy"])
    return log_filename

server_log_filename = init_server_log()
current_round = 0

# 定义：server端如何计算average accuracy和average loss
def weighted_avg(metrics):
    global current_round
    current_round += 1
    
    # 加权计算全局loss
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    global_loss = sum(losses) / sum(examples)
    
    # 加权计算全局accuracy
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    global_accuracy = sum(accuracies) / sum(examples)
    
    # 实时打印
    print(f"\n=== Ada's Tips: Server Round {current_round} Global Metrics ===")
    print(f"Global Loss: {global_loss:.4f} | Global Accuracy: {global_accuracy:.4f}")
    
    # 写入CSV
    with open(server_log_filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([current_round, global_loss, global_accuracy])
    
    return {"loss": global_loss, "accuracy": global_accuracy}

# 配置FedAvg策略
strategy = FedAvg(
    fraction_fit=0.5,                     # 每轮用50%客户端训练
    fraction_evaluate=1.0,                # 每轮用100%客户端评估
    min_fit_clients=2,                    # 最少2个client参与训练
    min_evaluate_clients=2,               # 最少2个client参与评估
    min_available_clients=NUM_CLIENTS,    # 必须启动NUM_CLIENTS个client才开始
    evaluate_metrics_aggregation_fn=weighted_avg,     # 自定义全局指标聚合
)

def plot_server_metrics():
    """绘制服务端全局loss和accuracy趋势图"""
    df = pd.read_csv(server_log_filename)
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制Global Loss曲线
    ax1.plot(df["round"], df["global_loss"], label="Global Loss", marker='o', color='red')
    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Loss")
    ax1.set_title("Server Global Loss Trend")
    ax1.legend()
    ax1.grid(True)
    
    # 绘制Global Accuracy曲线
    ax2.plot(df["round"], df["global_accuracy"], label="Global Accuracy", marker='^', color='green')
    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Server Global Accuracy Trend")
    ax2.legend()
    ax2.grid(True)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig("server_global_metrics.png")
    plt.close()
    print("\n Ada tell you ====== server globle metrics saved in server_global_metrics.png")

# 启动服务端
if __name__ == "__main__":
    print("Ada Tips ====== Federated Learning Server Running")
    # 启动服务端训练
    flwr.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=flwr.server.ServerConfig(num_rounds=NUM_ROUNDS)  # 训练轮数
    )
    # 训练结束后绘制服务端指标图
    plot_server_metrics()