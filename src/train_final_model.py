"""Train and save the final validation-selected Random Forest model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from features import extract_summary_features
from utils import (
    FEATURES_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    RESULTS_DIR,
    SPLITS_PATH,
    ensure_project_dirs,
    set_seed,
)


DURATION_SECONDS = 10.0
FEATURE_SET = "combined"
MODEL_PATH = MODELS_DIR / "final_random_forest.joblib"
METRICS_PATH = RESULTS_DIR / "final_random_forest_metrics.csv"
REPORT_PATH = REPORTS_DIR / "final_random_forest_test.txt"
FIGURE_PATH = FIGURES_DIR / "confusion_matrix_final_random_forest.png"


def cache_path() -> Path:
    return FEATURES_DIR / "final_combined_10s.npz"


def load_or_extract_features(splits: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    path = cache_path()
    if path.exists():
        cached = np.load(path, allow_pickle=True)
        return cached["X"], cached["y"]

    features = []
    labels = []
    for index, row in splits.iterrows():
        features.append(
            extract_summary_features(
                row["filepath"],
                duration_seconds=DURATION_SECONDS,
                feature_set=FEATURE_SET,
            )
        )
        labels.append(row["song_label"])
        if (index + 1) % 100 == 0:
            print(f"Extracted final-model features for {index + 1}/{len(splits)} files")

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    return X, y


def build_final_model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=3,
        class_weight=None,
        n_jobs=-1,
        random_state=seed,
    )


def metric_row(split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "model": "final_random_forest",
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "features": FEATURE_SET,
        "duration_seconds": DURATION_SECONDS,
        "notes": "validation-selected diagnostic Random Forest; subject-aware split",
    }


def save_outputs(
    model: RandomForestClassifier,
    labels: list[str],
    y_test: np.ndarray,
    test_pred: np.ndarray,
    seed: int,
) -> None:
    joblib.dump(
        {
            "model": model,
            "labels": labels,
            "feature_set": FEATURE_SET,
            "duration_seconds": DURATION_SECONDS,
            "random_seed": seed,
        },
        MODEL_PATH,
    )

    REPORT_PATH.write_text(classification_report(y_test, test_pred, zero_division=0), encoding="utf-8")

    matrix = confusion_matrix(y_test, test_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, cmap="Greens", xticks_rotation=45, colorbar=False)
    ax.set_title("Final Random Forest test confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    ensure_project_dirs()
    set_seed(args.seed)

    splits = pd.read_csv(args.splits)
    X, y = load_or_extract_features(splits)
    train_mask = splits["split"].to_numpy() == "train"
    val_mask = splits["split"].to_numpy() == "val"
    test_mask = splits["split"].to_numpy() == "test"
    labels = sorted(pd.unique(y))

    model = build_final_model(args.seed)
    model.fit(X[train_mask], y[train_mask])

    val_pred = model.predict(X[val_mask])
    test_pred = model.predict(X[test_mask])
    rows = pd.DataFrame(
        [
            metric_row("val", y[val_mask], val_pred),
            metric_row("test", y[test_mask], test_pred),
        ]
    )
    rows.to_csv(METRICS_PATH, index=False)
    save_outputs(model, labels, y[test_mask], test_pred, args.seed)

    print(rows.to_string(index=False))
    print(f"Saved final model to {MODEL_PATH.relative_to(MODELS_DIR.parent)}")


if __name__ == "__main__":
    main()
