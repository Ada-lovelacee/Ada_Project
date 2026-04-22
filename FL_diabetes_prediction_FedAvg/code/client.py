"""
Federated Learning for Diabetes Prediction by Ada, April 2026.
"""

import csv
import os
import sys

import flwr
import matplotlib.pyplot as plt
import pandas as pd
import torch

from data_and_simulation import get_federated_client_loaders
from model import (
    SimpleMLP,
    evaluate_model,
    flatten_metrics,
    get_model_parameters,
    set_model_parameters,
    train_model,
)


BATCH_SIZE = 16
LR = 0.001
EPOCHS = 10

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _client_log_path(cid):
    return os.path.join(RESULTS_DIR, f"client_{cid}_metrics.csv")


def init_client_log(cid):
    _ensure_results_dir()
    fieldnames = [
        "round",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_roc_auc",
        "val_tn",
        "val_fp",
        "val_fn",
        "val_tp",
    ]

    with open(_client_log_path(cid), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def append_client_log(cid, round_id, train_loss, val_metrics):
    row = {"round": round_id, "train_loss": train_loss}
    row.update(flatten_metrics(val_metrics, prefix="val"))

    with open(_client_log_path(cid), "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


def plot_client_metrics(cid):
    log_path = _client_log_path(cid)
    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(df["round"], df["train_loss"], label="Train Loss", marker="o", color="blue")
    axes[0].plot(df["round"], df["val_loss"], label="Val Loss", marker="s", color="red")
    axes[0].set_xlabel("Federated Round")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Client {cid} Local Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df["round"], df["val_accuracy"], label="Val Accuracy", marker="^", color="green")
    axes[1].plot(df["round"], df["val_f1"], label="Val F1", marker="d", color="orange")
    axes[1].set_xlabel("Federated Round")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"Client {cid} Validation Accuracy and F1")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(df["round"], df["val_roc_auc"], label="Val ROC-AUC", marker="o", color="purple")
    axes[2].set_xlabel("Federated Round")
    axes[2].set_ylabel("AUC")
    axes[2].set_title(f"Client {cid} Validation ROC-AUC")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"client_{cid}_metrics.png"))
    plt.close()


class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, cid, train_loader, val_loader, model):
        self.cid = cid
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        self.round = 0
        init_client_log(cid)

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def set_parameters(self, parameters):
        set_model_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.round += 1
        self.set_parameters(parameters)

        train_loss = train_model(
            self.model,
            self.train_loader,
            self.optimizer,
            self.criterion,
            epochs=EPOCHS,
        )
        val_metrics = evaluate_model(self.model, self.val_loader, self.criterion)
        append_client_log(self.cid, self.round, train_loss, val_metrics)

        print(f"\n=== Client {self.cid} Round {self.round} ===")
        print(
            "Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            "Val Acc: {val_acc:.4f} | Val Precision: {val_precision:.4f} | "
            "Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}".format(
                train_loss=train_loss,
                val_loss=val_metrics["loss"],
                val_acc=val_metrics["accuracy"],
                val_precision=val_metrics["precision"],
                val_recall=val_metrics["recall"],
                val_f1=val_metrics["f1"],
                val_auc=val_metrics["roc_auc"],
            )
        )

        return get_model_parameters(self.model), len(self.train_loader.dataset), {
            "train_loss": float(train_loss),
            "val_f1": float(val_metrics["f1"]),
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = evaluate_model(self.model, self.val_loader, self.criterion)
        return float(metrics["loss"]), len(self.val_loader.dataset), {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "roc_auc": float(metrics["roc_auc"]),
        }


def start_client(cid):
    train_loader, val_loader, _ = get_federated_client_loaders(cid, batch_size=BATCH_SIZE)
    model = SimpleMLP(input_dim=14, hidden_dim=16, num_classes=2)
    client = FlowerClient(cid, train_loader, val_loader, model)
    flwr.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
    plot_client_metrics(cid)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please assign a client ID, for example: python client.py 0")
        sys.exit(1)

    start_client(sys.argv[1])
