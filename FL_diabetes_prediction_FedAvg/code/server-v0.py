"""
Federated Learning for Diabetes Prediction by Ada, April 2026.
"""

import flwr
from flwr.server.strategy import FedAvg

from data_and_simulation import NUM_CLIENTS

# 定义：server端如何计算average accuracy
def weighted_avg_acc(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# 配置FedAvg策略
strategy = FedAvg(
    fraction_fit=0.5,                     # 每轮用50%客户端训练
    fraction_evaluate=1.0,                # 每轮用100%客户端评估
    min_fit_clients=2,                    # 最少2个client参与训练
    min_evaluate_clients=2,               # 最少2个client参与评估
    min_available_clients=NUM_CLIENTS,    # 必须启动NUM_CLIENTS个client才开始
    evaluate_metrics_aggregation_fn=weighted_avg_acc,     # 自定义一个平均accuracy
)

# 启动服务端
if __name__ == "__main__":
    print("启动联邦学习服务端 (FedAvg)")
    flwr.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=flwr.server.ServerConfig(num_rounds=10)  # 训练10轮
    )