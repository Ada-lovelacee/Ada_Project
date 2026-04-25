import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    DASHBOARD_REFRESH_MS = int(os.getenv("DASHBOARD_REFRESH_MS", "5000"))
    SYNC_PROJECT_DIR = BASE_DIR / "Synchronization_200Rounds"
    SYNC_CODE_DIR = SYNC_PROJECT_DIR / "code"
    SYNC_RESULTS_DIR = SYNC_CODE_DIR / "results"
