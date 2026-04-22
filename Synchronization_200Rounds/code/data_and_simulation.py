"""
Federated Learning for Diabetes Prediction by Ada, April 2026.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


# 配置数据划分比列参数（PCT-pecentage）
# TRAIN_PCT和VALID_PCT用于训练和调参等
# TEST_PCT用于FL训练中在Server端测试模型性能
TRAIN_PCT = 0.8
VALID_PCT = 0.1
TEST_PCT  = 0.1

# 配置FL参数（运行前配置）
NUM_CLIENTS = 5
NUM_ROUNDS = 200
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
    返回标准化后的 X_FL, y_FL（用于Client端FL训练和调参）；
    csv中用于FL的总比例: TRAIN_PCT + VALID_PCT 
    csv中用于FL训练中在Server端对模型的测试数据比例: TEST_PCT
    同时返回以及X_Test和y_Test用于Server端，独立测试FL每次聚合后的模型性能（也用于Centralized训练后模型的独立测试）
    """
    df = pd.read_csv(path)
    
    # 特征 & 标签
    X = df[ALL_FEATURES].values
    y = df[TARGET_COL].values

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # X_FL和y_FL用于FL训练和调参使用；X_Test和y_Test用于Server端测试FL每次聚合后的独立测试
    X_FL, X_Test, y_FL, y_Test = train_test_split(
        X, y, test_size=TEST_PCT, random_state=42     # , stratify=y
    )

    # 转tensor
    X_FL = torch.tensor(X_FL, dtype=torch.float32)  
    y_FL = torch.tensor(y_FL, dtype=torch.long)  # 分类任务用
    X_Test = torch.tensor(X_Test, dtype=torch.float32)
    y_Test = torch.tensor(y_Test, dtype=torch.long)  # 分类任务用

    return X_FL, y_FL, X_Test, y_Test

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


def get_dataloaders(dataset, batch_size=16):
    """生成训练/测试dataloader"""
    real_train_pct = TRAIN_PCT/(TRAIN_PCT+VALID_PCT)
    train_size = int(real_train_pct * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader