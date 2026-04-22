"""
Federated Learning for Diabetes Prediction by Ada, April 2026.
"""

import csv
import os

import flwr
import matplotlib.pyplot as plt
import pandas as pd
import torch
from flwr.server.strategy import FedAvg

try:
    from flwr.common import parameters_to_ndarrays
except ImportError:  # Compatibility with older Flower releases.
    from flwr.common import parameters_to_weights as parameters_to_ndarrays

from data_and_simulation import NUM_CLIENTS, NUM_ROUNDS, get_global_eval_loaders, write_split_summary
from model import SimpleMLP, evaluate_model, flatten_metrics, set_model_parameters


SELECTION_METRIC = "f1"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SERVER_LOG_PATH = os.path.join(RESULTS_DIR, "server_global_metrics.csv")
SERVER_TEST_PATH = os.path.join(RESULTS_DIR, "server_test_metrics.csv")
SPLIT_SUMMARY_PATH = os.path.join(RESULTS_DIR, "data_split_summary.csv")
EVAL_BATCH_SIZE = 256


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def init_server_log():
    _ensure_results_dir()
    fieldnames = [
        "round",
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

    with open(SERVER_LOG_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def append_server_log(round_id, metrics):
    row = {"round": round_id}
    row.update(flatten_metrics(metrics, prefix="val"))

    with open(SERVER_LOG_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


def save_test_metrics(rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(SERVER_TEST_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_server_metrics():
    df = pd.read_csv(SERVER_LOG_PATH)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(df["round"], df["val_loss"], label="Val Loss", marker="o", color="red")
    axes[0].set_xlabel("Federated Round")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Server Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df["round"], df["val_accuracy"], label="Val Accuracy", marker="^", color="green")
    axes[1].plot(df["round"], df["val_f1"], label="Val F1", marker="d", color="orange")
    axes[1].set_xlabel("Federated Round")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Server Validation Accuracy and F1")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(df["round"], df["val_roc_auc"], label="Val ROC-AUC", marker="o", color="purple")
    axes[2].set_xlabel("Federated Round")
    axes[2].set_ylabel("AUC")
    axes[2].set_title("Server Validation ROC-AUC")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "server_global_metrics.png"))
    plt.close()


def _make_model():
    return SimpleMLP(input_dim=14, hidden_dim=16, num_classes=2)


def _clone_parameters(parameters):
    return [array.copy() for array in parameters]


def _should_update_best(best_metrics, current_metrics):
    if best_metrics is None:
        return True

    current_key = current_metrics[SELECTION_METRIC]
    best_key = best_metrics[SELECTION_METRIC]

    if current_key > best_key:
        return True

    return current_key == best_key and current_metrics["loss"] < best_metrics["loss"]


def _evaluate_with_parameters(parameters, data_loader, criterion):
    model = _make_model()
    set_model_parameters(model, parameters)
    return evaluate_model(model, data_loader, criterion)


def build_server_strategy():
    init_server_log()
    write_split_summary(SPLIT_SUMMARY_PATH)

    val_loader, test_loader = get_global_eval_loaders(batch_size=EVAL_BATCH_SIZE)
    criterion = torch.nn.CrossEntropyLoss()

    tracking = {
        "best_parameters": None,
        "best_round": None,
        "best_val_metrics": None,
        "final_parameters": None,
        "final_round": None,
    }

    def evaluate_global_model(server_round, parameters, config):
        ndarrays = parameters_to_ndarrays(parameters)
        val_metrics = _evaluate_with_parameters(ndarrays, val_loader, criterion)

        tracking["final_parameters"] = _clone_parameters(ndarrays)
        tracking["final_round"] = server_round

        if _should_update_best(tracking["best_val_metrics"], val_metrics):
            tracking["best_parameters"] = _clone_parameters(ndarrays)
            tracking["best_round"] = server_round
            tracking["best_val_metrics"] = dict(val_metrics)

        append_server_log(server_round, val_metrics)

        print(
            "\n=== Server Round {round_id} Validation Metrics ===\n"
            "Val Loss: {loss:.4f} | Val Acc: {accuracy:.4f} | "
            "Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | "
            "Val F1: {f1:.4f} | Val AUC: {roc_auc:.4f} | "
            "CM: [[{tn}, {fp}], [{fn}, {tp}]]".format(
                round_id=server_round,
                **val_metrics,
            )
        )

        return float(val_metrics["loss"]), {
            "accuracy": float(val_metrics["accuracy"]),
            "precision": float(val_metrics["precision"]),
            "recall": float(val_metrics["recall"]),
            "f1": float(val_metrics["f1"]),
            "roc_auc": float(val_metrics["roc_auc"]),
            "tn": int(val_metrics["tn"]),
            "fp": int(val_metrics["fp"]),
            "fn": int(val_metrics["fn"]),
            "tp": int(val_metrics["tp"]),
        }

    strategy = FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        min_evaluate_clients=0,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_global_model,
    )

    return strategy, tracking, test_loader, criterion


def run_server():
    strategy, tracking, test_loader, criterion = build_server_strategy()

    print("Federated learning server running with centralized validation/test evaluation")
    flwr.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=flwr.server.ServerConfig(num_rounds=NUM_ROUNDS),
    )

    test_rows = []
    if tracking["final_parameters"] is not None:
        final_test_metrics = _evaluate_with_parameters(
            tracking["final_parameters"],
            test_loader,
            criterion,
        )
        test_rows.append(
            {
                "selection": "final",
                "source_round": tracking["final_round"],
                **flatten_metrics(final_test_metrics),
            }
        )

    if tracking["best_parameters"] is not None:
        best_test_metrics = _evaluate_with_parameters(
            tracking["best_parameters"],
            test_loader,
            criterion,
        )
        test_rows.append(
            {
                "selection": f"best_val_{SELECTION_METRIC}",
                "source_round": tracking["best_round"],
                **flatten_metrics(best_test_metrics),
            }
        )

    save_test_metrics(test_rows)
    plot_server_metrics()

    print("\n=== Federated server evaluation on global test split ===")
    for row in test_rows:
        print(
            "{selection} | Round {source_round} | Test Loss: {loss:.4f} | "
            "Acc: {accuracy:.4f} | Precision: {precision:.4f} | "
            "Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f} | "
            "CM: [[{tn}, {fp}], [{fn}, {tp}]]".format(**row)
        )


if __name__ == "__main__":
    run_server()
