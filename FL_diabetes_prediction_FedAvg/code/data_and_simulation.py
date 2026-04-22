"""
Federated Learning for Diabetes Prediction by Ada, April 2026.

Shared dataset preparation utilities for both centralized and federated runs.
"""

from functools import lru_cache
import csv
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


# Federated experiment configuration
NUM_CLIENTS = 3
NUM_ROUNDS = 20
DIRICHLET_ALPHA = 0.5
CSV_FILENAME = "diabetes_clean.csv"
RANDOM_STATE = 42

# A cleaner experiment split than the original train/test-only setup:
# 70% train, 10% validation, 20% test.
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2


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

ALL_FEATURES = DEMO_FEATURES + PHYSICAL_FEATURES + LAB_FEATURES


def _resolve_csv_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(__file__), path)


def _can_stratify(labels):
    values, counts = np.unique(labels, return_counts=True)
    return len(values) > 1 and counts.min() >= 2


def _safe_train_test_split(indices, labels, test_size, random_state):
    stratify = labels if _can_stratify(labels) else None
    try:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
            shuffle=True,
        )
    except ValueError:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )


def _make_tensor_dataset(features, labels):
    return TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def _dataset_from_indices(features, labels, indices):
    return _make_tensor_dataset(features[indices], labels[indices])


def _count_labels(labels):
    counts = {}
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, label_counts):
        counts[int(label)] = int(count)
    return counts


def load_raw_data(path):
    """Load raw features/labels without scaling."""
    csv_path = _resolve_csv_path(path)
    df = pd.read_csv(csv_path)
    features = df[ALL_FEATURES].to_numpy(dtype=np.float32)
    labels = df[TARGET_COL].to_numpy(dtype=np.int64)
    return features, labels


def dirichlet_partition_indices(labels, num_clients, alpha, random_state=RANDOM_STATE):
    """
    Partition samples across clients with a deterministic Dirichlet split.

    A local RNG keeps every process aligned on the same partition without
    mutating global NumPy state.
    """
    rng = np.random.default_rng(random_state)
    client_indices = [[] for _ in range(num_clients)]

    for class_label in np.unique(labels):
        class_indices = np.where(labels == class_label)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        class_splits = np.split(class_indices, split_points)

        for client_id, split in enumerate(class_splits):
            client_indices[client_id].extend(split.tolist())

    return [np.array(sorted(indices), dtype=np.int64) for indices in client_indices]


def _split_client_indices(indices, labels, client_id, random_state=RANDOM_STATE):
    """Create deterministic train/val/test splits inside one client dataset."""
    holdout_ratio = VAL_RATIO + TEST_RATIO
    client_labels = labels[indices]

    train_idx, holdout_idx = _safe_train_test_split(
        indices,
        client_labels,
        test_size=holdout_ratio,
        random_state=random_state + client_id,
    )

    holdout_labels = labels[holdout_idx]
    test_share_within_holdout = TEST_RATIO / holdout_ratio
    val_idx, test_idx = _safe_train_test_split(
        holdout_idx,
        holdout_labels,
        test_size=test_share_within_holdout,
        random_state=random_state + 100 + client_id,
    )

    return {
        "train_idx": np.array(sorted(train_idx), dtype=np.int64),
        "val_idx": np.array(sorted(val_idx), dtype=np.int64),
        "test_idx": np.array(sorted(test_idx), dtype=np.int64),
    }


def _summarize_split(split_name, labels):
    label_counts = _count_labels(labels)
    return {
        "split": split_name,
        "size": int(len(labels)),
        "label_0": int(label_counts.get(0, 0)),
        "label_1": int(label_counts.get(1, 0)),
    }


@lru_cache(maxsize=4)
def prepare_experiment_data(
    path=CSV_FILENAME,
    num_clients=NUM_CLIENTS,
    alpha=DIRICHLET_ALPHA,
    random_state=RANDOM_STATE,
):
    """
    Build a single deterministic data view shared by centralized and FL runs.

    Workflow:
    1. Partition the raw dataset across clients with a Dirichlet split.
    2. Split each client's data into train/val/test.
    3. Fit a single scaler on the union of client train data only.
    4. Transform train/val/test using that train-only scaler.

    This keeps centralized and federated experiments comparable while avoiding
    the earlier train/test leakage.
    """
    raw_features, labels = load_raw_data(path)
    client_indices = dirichlet_partition_indices(labels, num_clients, alpha, random_state)
    client_index_splits = [
        _split_client_indices(indices, labels, client_id, random_state)
        for client_id, indices in enumerate(client_indices)
    ]

    global_train_idx = np.concatenate([split["train_idx"] for split in client_index_splits])
    global_val_idx = np.concatenate([split["val_idx"] for split in client_index_splits])
    global_test_idx = np.concatenate([split["test_idx"] for split in client_index_splits])

    scaler = StandardScaler()
    scaler.fit(raw_features[global_train_idx])
    scaled_features = scaler.transform(raw_features).astype(np.float32)

    client_datasets = []
    split_summary_rows = []

    for client_id, split in enumerate(client_index_splits):
        client_train_labels = labels[split["train_idx"]]
        client_val_labels = labels[split["val_idx"]]
        client_test_labels = labels[split["test_idx"]]

        client_datasets.append(
            {
                "train": _dataset_from_indices(scaled_features, labels, split["train_idx"]),
                "val": _dataset_from_indices(scaled_features, labels, split["val_idx"]),
                "test": _dataset_from_indices(scaled_features, labels, split["test_idx"]),
                "summary": {
                    "client_id": client_id,
                    "train": _summarize_split("train", client_train_labels),
                    "val": _summarize_split("val", client_val_labels),
                    "test": _summarize_split("test", client_test_labels),
                },
            }
        )

        split_summary_rows.extend(
            [
                {
                    "scope": f"client_{client_id}",
                    **_summarize_split("train", client_train_labels),
                },
                {
                    "scope": f"client_{client_id}",
                    **_summarize_split("val", client_val_labels),
                },
                {
                    "scope": f"client_{client_id}",
                    **_summarize_split("test", client_test_labels),
                },
            ]
        )

    global_datasets = {
        "train": _dataset_from_indices(scaled_features, labels, np.sort(global_train_idx)),
        "val": _dataset_from_indices(scaled_features, labels, np.sort(global_val_idx)),
        "test": _dataset_from_indices(scaled_features, labels, np.sort(global_test_idx)),
    }

    split_summary_rows.extend(
        [
            {"scope": "global", **_summarize_split("train", labels[global_train_idx])},
            {"scope": "global", **_summarize_split("val", labels[global_val_idx])},
            {"scope": "global", **_summarize_split("test", labels[global_test_idx])},
        ]
    )

    return {
        "client_datasets": client_datasets,
        "global_datasets": global_datasets,
        "split_summary_rows": split_summary_rows,
    }


def get_centralized_dataloaders(batch_size=32):
    data = prepare_experiment_data()
    train_loader = DataLoader(data["global_datasets"]["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data["global_datasets"]["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(data["global_datasets"]["test"], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_federated_client_loaders(client_id, batch_size=32):
    data = prepare_experiment_data()
    client_data = data["client_datasets"][int(client_id)]
    train_loader = DataLoader(client_data["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(client_data["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(client_data["test"], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_global_eval_loaders(batch_size=256):
    data = prepare_experiment_data()
    val_loader = DataLoader(data["global_datasets"]["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(data["global_datasets"]["test"], batch_size=batch_size, shuffle=False)
    return val_loader, test_loader


def write_split_summary(output_path):
    """Persist a human-readable summary of every split for the report."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rows = prepare_experiment_data()["split_summary_rows"]
    fieldnames = ["scope", "split", "size", "label_0", "label_1"]

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Backward-compatible helpers kept for the older prototype scripts.
# ---------------------------------------------------------------------------

def load_csv_data(path):
    """Legacy helper kept for older prototype scripts."""
    features, labels = load_raw_data(path)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return (
        torch.tensor(scaled_features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def dirichlet_partition_data(features, labels, num_clients, alpha):
    """Legacy helper kept for older prototype scripts."""
    if torch.is_tensor(labels):
        label_array = labels.cpu().numpy()
    else:
        label_array = np.asarray(labels)

    client_indices = dirichlet_partition_indices(label_array, num_clients, alpha)
    client_datasets = []

    for indices in client_indices:
        if torch.is_tensor(features):
            client_features = features[indices]
        else:
            client_features = torch.tensor(features[indices], dtype=torch.float32)

        if torch.is_tensor(labels):
            client_labels = labels[indices]
        else:
            client_labels = torch.tensor(label_array[indices], dtype=torch.long)

        client_datasets.append(TensorDataset(client_features, client_labels))

    return client_datasets


def get_dataloaders(dataset, batch_size=32):
    """Legacy train/test-only split kept for older prototype scripts."""
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
