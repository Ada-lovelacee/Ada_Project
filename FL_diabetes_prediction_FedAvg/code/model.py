"""
Federated Learning for Diabetes Prediction by Ada, April 2026.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as nnFunc


LOG_METRIC_FIELDS = (
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "tn",
    "fp",
    "fn",
    "tp",
)


class SimpleMLP(nn.Module):
    """
    Three-layer MLP for binary classification on 14 tabular features.
    """

    def __init__(self, input_dim=14, hidden_dim=16, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = nnFunc.relu(self.fc1(x))
        x = nnFunc.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model_parameters(model):
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def set_model_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {key: torch.tensor(value) for key, value in params_dict}
    model.load_state_dict(state_dict, strict=True)


def train_model(model, train_loader, optimizer, criterion, epochs=1):
    """Train for one or more local epochs and return the average train loss."""
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

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples if total_samples else float("nan")


def _compute_binary_metrics(y_true, y_pred, y_score):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else float("nan")

    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_model(model, data_loader, criterion):
    """
    Evaluate the current model and return loss plus report-friendly metrics.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []
    all_positive_probs = []

    with torch.no_grad():
        for data, labels in data_loader:
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_positive_probs.extend(probabilities[:, 1].cpu().numpy().tolist())

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_predictions, dtype=np.int64)
    y_score = np.asarray(all_positive_probs, dtype=np.float32)

    metrics = _compute_binary_metrics(y_true, y_pred, y_score)
    metrics["loss"] = total_loss / total_samples if total_samples else float("nan")
    metrics["num_samples"] = int(total_samples)
    return metrics


def flatten_metrics(metrics, prefix=None):
    """Flatten a metrics dict into CSV-friendly scalar fields."""
    flattened = {}
    for field in LOG_METRIC_FIELDS:
        key = f"{prefix}_{field}" if prefix else field
        flattened[key] = metrics[field]
    return flattened


def test_model(model, test_loader, criterion):
    """
    Legacy helper kept for the older prototype code paths.
    """
    metrics = evaluate_model(model, test_loader, criterion)
    return metrics["loss"], metrics["accuracy"]
