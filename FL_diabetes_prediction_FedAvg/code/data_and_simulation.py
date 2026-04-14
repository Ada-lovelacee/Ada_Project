"""
Federated Learning for Diabetes Prediction by Ada, April 2026.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


# 配置FL参数（运行前配置）
NUM_CLIENTS = 3
NUM_ROUNDS = 20
DIRICHLET_ALPHA = 0.5    ## 1.0/0.5/0.1
CSV_FILENAME ="diabetes_clean.csv"


# 固定随机种子保证可复现
np.random.seed(42)
torch.manual_seed(42)


# 数据集中的特征变量和响应变量/标签
TARGET_COL = "Diabetes_Type"
LABEL_POSITIVE = "T2D"
LABEL_NEGATIVE = "Not diabetic"

DEMO_FEATURES = [
    "RIDAGEYR__demographics",
]

PHYSICAL_FEATURES = [
    "BMXBMI__response",
    "BMXWAIST__response",
    "BPXPLS__response",
]

LAB_FEATURES = [
    "LBXGLU__response",
    "LBXGH__response",
    "LBDINSI__response",
    "LBDHDD__response",
    "LBDLDL__response",
    "LBXSTR__response",
    "LBDSCHSI__response",
    "LBXSCR__response",
    "VNEGFR__response",
    "LBXCRP__response",
]

# 预测变量/特征变量
ALL_FEATURES = DEMO_FEATURES + PHYSICAL_FEATURES + LAB_FEATURES


def load_csv_data(path):
    """
    读取CSV：ALL_FEATURES是特征，TARGET_COL是二分类标签
    返回标准化后的 X, y
    """
    df = pd.read_csv(path)
    
    # 特征 & 标签
    X = df[ALL_FEATURES].values
    y = df[TARGET_COL].values

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 转tensor
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)  # 分类任务用long

    return X, y

def dirichlet_partition_data(X, y, num_clients, alpha):
    """
    Dirichlet 分布划分数据给多个客户端
    alpha 越小 → 非独立同分布（non-IID）越强
    """
    n_classes = len(torch.unique(y))
    client_indices = [[] for _ in range(num_clients)]

    # 按类别做Dirichlet划分
    for k in range(n_classes):
        idx_k = np.where(y == k)[0]
        np.random.shuffle(idx_k)

        # Dirichlet分布比例
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(idx_k)).astype(int)
        
        # 补齐长度
        proportions[-1] = len(idx_k) - np.sum(proportions[:-1])

        # 分配索引
        current = 0
        for i in range(num_clients):
            client_indices[i].extend(idx_k[current:current + proportions[i]])
            current += proportions[i]

    # 构建每个client的dataset
    client_datasets = []
    for idx in client_indices:
        dataset = TensorDataset(X[idx], y[idx])
        client_datasets.append(dataset)
    
    return client_datasets


def get_dataloaders(dataset, batch_size=32):
    """生成训练/测试dataloader"""
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader