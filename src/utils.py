"""Shared configuration and reproducibility helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


RANDOM_SEED = 42
TARGET_SAMPLE_RATE = 16_000
TARGET_AUDIO_DURATION_SECONDS = 10.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "MLEndHWII_sample_800"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_DIR = RESULTS_DIR / "features"
REPORTS_DIR = RESULTS_DIR / "classification_reports"
METADATA_PATH = RESULTS_DIR / "metadata.csv"
SPLITS_PATH = RESULTS_DIR / "splits.csv"
METRICS_ORIGINAL_PATH = RESULTS_DIR / "metrics_original.csv"
METRICS_IMPROVED_PATH = RESULTS_DIR / "metrics_improved.csv"
METRICS_COMPARISON_PATH = RESULTS_DIR / "metrics_comparison.csv"


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_project_dirs() -> None:
    """Create expected output directories if they do not already exist."""
    for path in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR, FEATURES_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def relative_to_project(path: Path) -> str:
    """Return a stable, project-relative path string."""
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()
