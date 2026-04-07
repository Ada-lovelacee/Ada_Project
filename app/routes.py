from datetime import datetime

from flask import Blueprint, jsonify, render_template

from .extensions import db
from .models import ClientNode, TrainingRound, TrainingTask


main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/api/overview")
def api_overview():
    task = TrainingTask.query.order_by(TrainingTask.id.desc()).first()
    clients = ClientNode.query.order_by(ClientNode.id.asc()).all()
    rounds = (
        TrainingRound.query.order_by(TrainingRound.round_no.asc()).limit(20).all()
    )

    payload = {
        "task": {
            "name": task.name,
            "model_name": task.model_name,
            "status": task.status,
            "global_accuracy": task.global_accuracy,
            "global_loss": task.global_loss,
            "progress": task.progress,
        },
        "summary": {
            "total_clients": len(clients),
            "online_clients": sum(client.status != "offline" for client in clients),
            "round_count": len(rounds),
            "avg_latency": round(
                sum(client.latency_ms for client in clients if client.latency_ms)
                / max(sum(client.latency_ms > 0 for client in clients), 1),
                1,
            ),
        },
        "clients": [
            {
                "id": client.id,
                "name": client.name,
                "location": client.location,
                "status": client.status,
                "data_size": client.data_size,
                "last_accuracy": client.last_accuracy,
                "latency_ms": client.latency_ms,
                "contribution_score": client.contribution_score,
                "updated_at": client.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for client in clients
        ],
        "rounds": [
            {
                "round_no": item.round_no,
                "participants": item.participants,
                "avg_local_accuracy": item.avg_local_accuracy,
                "global_accuracy": item.global_accuracy,
                "global_loss": item.global_loss,
                "aggregation_time": item.aggregation_time,
                "created_at": item.created_at.strftime("%H:%M"),
            }
            for item in rounds
        ],
    }
    return jsonify(payload)


@main_bp.route("/api/simulate", methods=["POST"])
def api_simulate():
    task = TrainingTask.query.order_by(TrainingTask.id.desc()).first()
    last_round = TrainingRound.query.order_by(TrainingRound.round_no.desc()).first()
    clients = ClientNode.query.order_by(ClientNode.id.asc()).all()

    next_round = 1 if not last_round else last_round.round_no + 1
    base_accuracy = last_round.global_accuracy if last_round else 0.75
    base_loss = last_round.global_loss if last_round else 0.6

    global_accuracy = min(round(base_accuracy + 0.008, 4), 0.985)
    global_loss = max(round(base_loss - 0.018, 4), 0.03)

    round_record = TrainingRound(
        round_no=next_round,
        participants=sum(client.status != "offline" for client in clients),
        avg_local_accuracy=max(round(global_accuracy - 0.01, 4), 0.0),
        global_accuracy=global_accuracy,
        global_loss=global_loss,
        aggregation_time=max(round(1.8 - next_round * 0.04, 2), 0.6),
        created_at=datetime.utcnow(),
    )
    db.session.add(round_record)

    for index, client in enumerate(clients):
        if client.status == "offline" and next_round % 3 != 0:
            continue

        client.status = "training" if index % 2 == next_round % 2 else "online"
        client.last_accuracy = max(round(global_accuracy - 0.004 * index, 4), 0.0)
        client.latency_ms = 42 + index * 9 + (next_round % 4) * 3
        client.contribution_score = round(max(0.14, 0.33 - index * 0.03), 2)
        client.updated_at = datetime.utcnow()

    task.global_accuracy = global_accuracy
    task.global_loss = global_loss
    task.progress = min(task.progress + 6, 100)
    if task.progress == 100:
        task.status = "completed"

    db.session.commit()
    return jsonify({"message": f"已生成第 {next_round} 轮聚合结果"})
