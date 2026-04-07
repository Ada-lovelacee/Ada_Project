from datetime import datetime

from .extensions import db


class TrainingTask(db.Model):
    __tablename__ = "training_tasks"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    model_name = db.Column(db.String(120), nullable=False)
    status = db.Column(db.String(40), nullable=False, default="running")
    global_accuracy = db.Column(db.Float, nullable=False, default=0.0)
    global_loss = db.Column(db.Float, nullable=False, default=0.0)
    progress = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class ClientNode(db.Model):
    __tablename__ = "client_nodes"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    location = db.Column(db.String(120), nullable=False)
    status = db.Column(db.String(40), nullable=False, default="online")
    data_size = db.Column(db.Integer, nullable=False, default=0)
    last_accuracy = db.Column(db.Float, nullable=False, default=0.0)
    latency_ms = db.Column(db.Integer, nullable=False, default=0)
    contribution_score = db.Column(db.Float, nullable=False, default=0.0)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class TrainingRound(db.Model):
    __tablename__ = "training_rounds"

    id = db.Column(db.Integer, primary_key=True)
    round_no = db.Column(db.Integer, nullable=False)
    participants = db.Column(db.Integer, nullable=False)
    avg_local_accuracy = db.Column(db.Float, nullable=False)
    global_accuracy = db.Column(db.Float, nullable=False)
    global_loss = db.Column(db.Float, nullable=False)
    aggregation_time = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
