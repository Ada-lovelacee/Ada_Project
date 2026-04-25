from __future__ import annotations

import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def build_overview_payload(sync_code_dir: Path, results_dir: Path) -> dict:
    metadata = load_project_metadata(sync_code_dir)
    server_rounds = load_server_rounds(results_dir / "server_global_metrics.csv")
    client_rounds = load_client_rounds(results_dir)
    test_metrics = load_test_metrics(results_dir / "metrics_On_Test.csv")
    artifacts = load_artifacts(results_dir)
    discovered_client_count = max(client_rounds.keys(), default=-1) + 1
    total_clients = max(metadata["total_clients"], discovered_client_count)

    round_count = len(server_rounds)
    total_rounds = metadata["total_rounds"]
    latest_server_round = server_rounds[-1] if server_rounds else {}
    overall_status = determine_overall_status(round_count, total_rounds)

    clients = build_client_summaries(
        client_rounds=client_rounds,
        current_round=round_count,
        total_clients=total_clients,
        overall_status=overall_status,
    )
    avg_client_accuracy = mean_value(
        [client["last_accuracy"] for client in clients if client["last_accuracy"] is not None]
    )
    avg_client_auc = mean_value(
        [client["last_auc"] for client in clients if client["last_auc"] is not None]
    )

    rounds = build_round_summaries(server_rounds, client_rounds)

    return {
        "task": {
            "name": metadata["project_name"],
            "model_name": metadata["model_name"],
            "status": overall_status,
            "global_accuracy": latest_server_round.get("global_accuracy", 0.0),
            "global_loss": latest_server_round.get("global_loss", 0.0),
            "global_auc": latest_server_round.get("global_auc", 0.0),
            "progress": clamp_progress(round_count, total_rounds),
            "round_count": round_count,
            "total_rounds": total_rounds,
            "alpha": metadata["alpha"],
            "dataset_file": metadata["dataset_file"],
            "results_dir": str(results_dir),
        },
        "summary": {
            "total_clients": total_clients,
            "reported_clients": sum(client["status"] != "idle" for client in clients),
            "round_count": round_count,
            "avg_client_accuracy": avg_client_accuracy,
            "avg_client_auc": avg_client_auc,
            "artifact_count": len(artifacts),
        },
        "clients": clients,
        "rounds": rounds,
        "test_metrics": test_metrics,
        "artifacts": artifacts,
    }


def load_project_metadata(sync_code_dir: Path) -> dict:
    data_script = read_text_file(sync_code_dir / "data_and_simulation.py")
    model_script = read_text_file(sync_code_dir / "model.py")

    return {
        "project_name": sync_code_dir.parent.name,
        "model_name": extract_model_name(model_script),
        "total_clients": extract_int_constant(data_script, "NUM_CLIENTS", 0),
        "total_rounds": extract_int_constant(data_script, "NUM_ROUNDS", 0),
        "alpha": extract_float_constant(data_script, "DIRICHLET_ALPHA", 0.0),
        "dataset_file": extract_string_constant(data_script, "CSV_FILENAME", "unknown"),
    }


def extract_model_name(script_text: str) -> str:
    match = re.search(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\(", script_text, re.MULTILINE)
    return match.group(1) if match else "UnknownModel"


def extract_int_constant(script_text: str, name: str, default: int) -> int:
    match = re.search(rf"^{name}\s*=\s*(\d+)", script_text, re.MULTILINE)
    return int(match.group(1)) if match else default


def extract_float_constant(script_text: str, name: str, default: float) -> float:
    match = re.search(rf"^{name}\s*=\s*([0-9]+(?:\.[0-9]+)?)", script_text, re.MULTILINE)
    return float(match.group(1)) if match else default


def extract_string_constant(script_text: str, name: str, default: str) -> str:
    match = re.search(rf'^{name}\s*=\s*"([^"]+)"', script_text, re.MULTILINE)
    return match.group(1) if match else default


def load_server_rounds(csv_path: Path) -> list[dict]:
    rows = read_csv_rows(csv_path)
    rounds = []
    for row in rows:
        rounds.append(
            {
                "round_no": safe_int(row.get("round")),
                "global_loss": safe_float(row.get("global_loss")),
                "global_accuracy": safe_float(row.get("global_acc")),
                "global_auc": safe_float(row.get("global_auc")),
            }
        )
    return [item for item in rounds if item["round_no"] > 0]


def load_client_rounds(results_dir: Path) -> dict[int, list[dict]]:
    client_rounds: dict[int, list[dict]] = {}
    for csv_path in sorted(results_dir.glob("client_*_metrics.csv")):
        match = re.fullmatch(r"client_(\d+)_metrics\.csv", csv_path.name)
        if not match:
            continue

        client_id = int(match.group(1))
        rows = read_csv_rows(csv_path)
        parsed_rows = []
        for row in rows:
            parsed_rows.append(
                {
                    "round_no": safe_int(row.get("round")),
                    "loss": safe_float(row.get("loss")),
                    "acc": safe_float(row.get("acc")),
                    "auc": safe_float(row.get("auc")),
                }
            )

        client_rounds[client_id] = [item for item in parsed_rows if item["round_no"] > 0]

    return client_rounds


def load_test_metrics(csv_path: Path) -> list[dict]:
    rows = read_csv_rows(csv_path)
    metrics = []
    for row in rows:
        metrics.append(
            {
                "location": row.get("location", "unknown"),
                "loss": safe_float(row.get("loss")),
                "acc": safe_float(row.get("acc")),
                "auc": safe_float(row.get("auc")),
            }
        )
    return metrics


def load_artifacts(results_dir: Path) -> list[dict]:
    artifacts = []
    for path in sorted(results_dir.glob("*.png")):
        artifacts.append(
            {
                "filename": path.name,
                "name": path.stem.replace("_", " ").title(),
                "updated_at": format_timestamp(datetime.fromtimestamp(path.stat().st_mtime)),
                "size_kb": round(path.stat().st_size / 1024, 1),
            }
        )
    return artifacts


def build_client_summaries(
    client_rounds: dict[int, list[dict]],
    current_round: int,
    total_clients: int,
    overall_status: str,
) -> list[dict]:
    clients = []
    for client_id in range(total_clients):
        rows = client_rounds.get(client_id, [])
        latest = rows[-1] if rows else {}

        if not rows:
            status = "idle"
        elif overall_status == "completed":
            status = "completed"
        elif latest.get("round_no", 0) >= current_round > 0:
            status = "reporting"
        else:
            status = "waiting"

        clients.append(
            {
                "id": client_id,
                "name": f"Client {client_id}",
                "status": status,
                "last_round": latest.get("round_no"),
                "last_loss": latest.get("loss"),
                "last_accuracy": latest.get("acc"),
                "last_auc": latest.get("auc"),
            }
        )

    return clients


def build_round_summaries(server_rounds: list[dict], client_rounds: dict[int, list[dict]]) -> list[dict]:
    per_round_client_metrics: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"acc": [], "auc": [], "loss": []}
    )

    for rows in client_rounds.values():
        for row in rows:
            if row["round_no"] <= 0:
                continue
            per_round_client_metrics[row["round_no"]]["acc"].append(row["acc"])
            per_round_client_metrics[row["round_no"]]["auc"].append(row["auc"])
            per_round_client_metrics[row["round_no"]]["loss"].append(row["loss"])

    rounds = []
    for round_item in server_rounds:
        round_no = round_item["round_no"]
        client_metrics = per_round_client_metrics.get(round_no, {})
        rounds.append(
            {
                "round_no": round_no,
                "participants": len(client_metrics.get("acc", [])),
                "avg_client_accuracy": mean_value(client_metrics.get("acc", [])),
                "avg_client_auc": mean_value(client_metrics.get("auc", [])),
                "avg_client_loss": mean_value(client_metrics.get("loss", [])),
                "global_accuracy": round_item["global_accuracy"],
                "global_loss": round_item["global_loss"],
                "global_auc": round_item["global_auc"],
            }
        )
    return rounds


def determine_overall_status(round_count: int, total_rounds: int) -> str:
    if round_count <= 0:
        return "not_started"
    if total_rounds > 0 and round_count >= total_rounds:
        return "completed"
    return "running"


def clamp_progress(round_count: int, total_rounds: int) -> int:
    if total_rounds <= 0:
        return 0
    return max(0, min(int(round_count / total_rounds * 100), 100))


def read_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def safe_float(value: str | float | int | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def safe_int(value: str | int | None) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def mean_value(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")
