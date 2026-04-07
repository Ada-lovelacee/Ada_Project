from datetime import datetime, timedelta

from .extensions import db
from .models import ClientNode, TrainingRound, TrainingTask


def seed_demo_data():
    if TrainingTask.query.first():
        return

    task = TrainingTask(
        name="医疗影像分类联邦实验",
        model_name="ResNet18",
        status="running",
        global_accuracy=0.912,
        global_loss=0.148,
        progress=72,
    )
    db.session.add(task)

    clients = [
        ClientNode(
            name="华东医院节点",
            location="上海",
            status="online",
            data_size=12500,
            last_accuracy=0.903,
            latency_ms=46,
            contribution_score=0.31,
            updated_at=datetime.utcnow() - timedelta(minutes=3),
        ),
        ClientNode(
            name="华南医院节点",
            location="广州",
            status="online",
            data_size=10800,
            last_accuracy=0.894,
            latency_ms=58,
            contribution_score=0.28,
            updated_at=datetime.utcnow() - timedelta(minutes=5),
        ),
        ClientNode(
            name="西部医院节点",
            location="成都",
            status="training",
            data_size=9700,
            last_accuracy=0.887,
            latency_ms=73,
            contribution_score=0.22,
            updated_at=datetime.utcnow() - timedelta(minutes=2),
        ),
        ClientNode(
            name="东北医院节点",
            location="沈阳",
            status="offline",
            data_size=8200,
            last_accuracy=0.842,
            latency_ms=0,
            contribution_score=0.19,
            updated_at=datetime.utcnow() - timedelta(minutes=18),
        ),
    ]
    db.session.add_all(clients)

    rounds = [
        TrainingRound(
            round_no=1,
            participants=3,
            avg_local_accuracy=0.742,
            global_accuracy=0.768,
            global_loss=0.624,
            aggregation_time=1.8,
            created_at=datetime.utcnow() - timedelta(hours=6),
        ),
        TrainingRound(
            round_no=2,
            participants=4,
            avg_local_accuracy=0.801,
            global_accuracy=0.823,
            global_loss=0.441,
            aggregation_time=1.7,
            created_at=datetime.utcnow() - timedelta(hours=5),
        ),
        TrainingRound(
            round_no=3,
            participants=4,
            avg_local_accuracy=0.854,
            global_accuracy=0.872,
            global_loss=0.292,
            aggregation_time=1.6,
            created_at=datetime.utcnow() - timedelta(hours=4),
        ),
        TrainingRound(
            round_no=4,
            participants=3,
            avg_local_accuracy=0.881,
            global_accuracy=0.894,
            global_loss=0.214,
            aggregation_time=1.5,
            created_at=datetime.utcnow() - timedelta(hours=3),
        ),
        TrainingRound(
            round_no=5,
            participants=4,
            avg_local_accuracy=0.903,
            global_accuracy=0.912,
            global_loss=0.148,
            aggregation_time=1.4,
            created_at=datetime.utcnow() - timedelta(hours=2),
        ),
    ]
    db.session.add_all(rounds)
    db.session.commit()
