"""
Federated Learning for Diabetes Prediction by Ada, April 2026.

Centralized training baseline with cleaner train/val/test splits and
report-ready classification metrics.
"""

import copy
import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from data_and_simulation import get_centralized_dataloaders, write_split_summary
from model import SimpleMLP, evaluate_model, flatten_metrics, train_model


BATCH_SIZE = 16
LR = 0.001
EPOCHS = 10
SELECTION_METRIC = "f1"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CENTRALIZED_LOG_PATH = os.path.join(RESULTS_DIR, "centralized_metrics.csv")
CENTRALIZED_TEST_PATH = os.path.join(RESULTS_DIR, "centralized_test_metrics.csv")
SPLIT_SUMMARY_PATH = os.path.join(RESULTS_DIR, "data_split_summary.csv")


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def init_central_log():
    _ensure_results_dir()
    fieldnames = [
        "epoch",
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

    with open(CENTRALIZED_LOG_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

    return fieldnames


def write_epoch_log(epoch, train_loss, val_metrics):
    row = {"epoch": epoch, "train_loss": train_loss}
    row.update(flatten_metrics(val_metrics, prefix="val"))

    with open(CENTRALIZED_LOG_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


def save_test_metrics(rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(CENTRALIZED_TEST_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_central_metrics():
    df = pd.read_csv(CENTRALIZED_LOG_PATH)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o", color="blue")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker="s", color="red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Centralized Training Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df["epoch"], df["val_accuracy"], label="Val Accuracy", marker="^", color="green")
    axes[1].plot(df["epoch"], df["val_f1"], label="Val F1", marker="d", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Centralized Validation Accuracy and F1")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(df["epoch"], df["val_roc_auc"], label="Val ROC-AUC", marker="o", color="purple")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].set_title("Centralized Validation ROC-AUC")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "centralized_metrics.png"))
    plt.close()


def _should_update_best(best_metrics, current_metrics):
    if best_metrics is None:
        return True

    current_key = current_metrics[SELECTION_METRIC]
    best_key = best_metrics[SELECTION_METRIC]

    if current_key > best_key:
        return True

    return current_key == best_key and current_metrics["loss"] < best_metrics["loss"]


def run_centralized_training():
    train_loader, val_loader, test_loader = get_centralized_dataloaders(batch_size=BATCH_SIZE)

    model = SimpleMLP(input_dim=14, hidden_dim=16, num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    init_central_log()
    write_split_summary(SPLIT_SUMMARY_PATH)

    best_state = None
    best_epoch = None
    best_val_metrics = None

    print("=== Centralized learning begins with train/val/test splits ===")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_model(model, train_loader, optimizer, criterion, epochs=1)
        val_metrics = evaluate_model(model, val_loader, criterion)

        write_epoch_log(epoch, train_loss, val_metrics)

        if _should_update_best(best_val_metrics, val_metrics):
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)

        print(
            "Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | "
            "Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            "Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | "
            "Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}".format(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_metrics["loss"],
                val_acc=val_metrics["accuracy"],
                val_precision=val_metrics["precision"],
                val_recall=val_metrics["recall"],
                val_f1=val_metrics["f1"],
                val_auc=val_metrics["roc_auc"],
            )
        )

    final_test_metrics = evaluate_model(model, test_loader, criterion)
    test_rows = [
        {
            "selection": "final",
            "source_epoch": EPOCHS,
            **flatten_metrics(final_test_metrics),
        }
    ]

    if best_state is not None:
        best_model = SimpleMLP(input_dim=14, hidden_dim=16, num_classes=2)
        best_model.load_state_dict(best_state)
        best_test_metrics = evaluate_model(best_model, test_loader, criterion)
        test_rows.append(
            {
                "selection": f"best_val_{SELECTION_METRIC}",
                "source_epoch": best_epoch,
                **flatten_metrics(best_test_metrics),
            }
        )

    save_test_metrics(test_rows)
    plot_central_metrics()

    print("\n=== Centralized learning completed ===")
    for row in test_rows:
        print(
            "{selection} | Epoch {source_epoch} | Test Loss: {loss:.4f} | "
            "Acc: {accuracy:.4f} | Precision: {precision:.4f} | "
            "Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f} | "
            "CM: [[{tn}, {fp}], [{fn}, {tp}]]".format(**row)
        )


if __name__ == "__main__":
    run_centralized_training()
