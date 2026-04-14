"""
Federated Learning for Diabetes Prediction by Ada, April 2026.
"""

import torch
import flwr

from data_and_simulation import load_csv_data, dirichlet_partition_data, get_dataloaders, NUM_CLIENTS, DIRICHLET_ALPHA, CSV_FILENAME
from model import SimpleMLP, train_model, test_model

# 部分模型超参
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 1

# 全局加载数据（只加载一次）
X, y = load_csv_data(CSV_FILENAME)
client_datasets = dirichlet_partition_data(X, y, NUM_CLIENTS, DIRICHLET_ALPHA)

# 定义Flower客户端
class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, train_loader, test_loader, model):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_model(self.model, self.train_loader, self.optimizer, self.criterion, EPOCHS)
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = test_model(self.model, self.test_loader, self.criterion)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}

# 启动客户端
def start_client(cid):
    # 分配对应client的数据
    dataset = client_datasets[int(cid)]
    train_loader, test_loader = get_dataloaders(dataset, BATCH_SIZE)
    
    # 初始化模型
    model = SimpleMLP(input_dim=14, hidden_dim=16, num_classes=2)
    
    # 启动Flower客户端
    client = FlowerClient(train_loader, test_loader, model)
    flwr.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())

if __name__ == "__main__":
    # 在各个终端窗口模拟各个client
    import sys
    start_client(sys.argv[1])

