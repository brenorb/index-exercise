"""Configuration settings for the system."""

from pathlib import Path

# Dataset configuration
DATASET_NAME = "ScaleAI/PRBench"
DATASET_CACHE_DIR = Path("./data/prbench_cache")
AVAILABLE_SPLITS = ["finance", "legal", "finance_hard", "legal_hard"]

# Codex configuration
CODEX_TIMEOUT = 30  # seconds
CODEX_DEFAULT_MODEL = "gpt-5-codex"

# System configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

