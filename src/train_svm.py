"""Train a validation-selected SVM baseline on corrected subject-aware splits."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import extract_summary_features
from utils import (
    FEATURES_DIR,
    FIGURES_DIR,
    METRICS_COMPARISON_PATH,
    METRICS_ORIGINAL_PATH,
    RANDOM_SEED,
    REPORTS_DIR,
    RESULTS_DIR,
    SPLITS_PATH,
    ensure_project_dirs,
    set_seed,
)


DURATION_SECONDS = 10.0
FEATURE_SET = "combined"
METRICS_SVM_PATH = RESULTS_DIR / "metrics_svm.csv"


def cache_path() -> Path:
    return FEATURES_DIR / "diagnostic_combined_10s.npz"


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
            print(f"Extracted SVM features for {index + 1}/{len(splits)} files")

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    return X, y


def build_svm(params: dict[str, object], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=str(params["kernel"]),
                    C=float(params["C"]),
                    gamma=params.get("gamma", "scale"),
                    class_weight=params["class_weight"],
                    random_state=seed,
                ),
            ),
        ]
    )


def metric_row(
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    chosen_params: str,
    notes: str,
) -> dict[str, object]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "model": "svm",
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "chosen_params": chosen_params,
        "notes": notes,
    }


def format_params(params: dict[str, object]) -> str:
    return "; ".join(f"{key}={value}" for key, value in params.items())


def candidate_params() -> list[dict[str, object]]:
    candidates = []
    for C, class_weight in product([0.01, 0.1, 1.0, 10.0], [None, "balanced"]):
        candidates.append({"kernel": "linear", "C": C, "gamma": "scale", "class_weight": class_weight})
    for C, gamma, class_weight in product([0.01, 0.1, 1.0, 10.0], ["scale", "auto"], [None, "balanced"]):
        candidates.append({"kernel": "rbf", "C": C, "gamma": gamma, "class_weight": class_weight})
    return candidates


def select_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[Pipeline, dict[str, object], pd.DataFrame]:
    rows = []
    best_model = None
    best_params: dict[str, object] | None = None
    best_score = -np.inf

    for params in candidate_params():
        model = build_svm(params, seed)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        row = metric_row(
            "val_candidate",
            y_val,
            pred,
            format_params(params),
            "validation candidate; duration=10s; feature_set=combined; subject-aware split",
        )
        rows.append(row)

        if row["macro_f1"] > best_score:
            best_score = float(row["macro_f1"])
            best_model = model
            best_params = params

    if best_model is None or best_params is None:
        raise RuntimeError("No SVM candidate was selected.")

    return best_model, best_params, pd.DataFrame(rows)


def save_test_artifacts(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> None:
    (REPORTS_DIR / "classification_report_svm.txt").write_text(
        classification_report(y_true, y_pred, zero_division=0), encoding="utf-8"
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, cmap="Purples", xticks_rotation=45, colorbar=False)
    ax.set_title("SVM test confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_svm.png", dpi=200)
    plt.close(fig)


def update_comparison(svm_rows: pd.DataFrame) -> None:
    comparison_rows = []

    if METRICS_ORIGINAL_PATH.exists():
        original = pd.read_csv(METRICS_ORIGINAL_PATH)
        for model_name in ["logreg", "rf", "cnn"]:
            rows = original[original["model"] == model_name].copy()
            rows["model"] = f"original_{model_name}"
            comparison_rows.append(rows)

    diagnostic_path = RESULTS_DIR / "split_diagnostics.csv"
    if diagnostic_path.exists():
        diagnostic = pd.read_csv(diagnostic_path)
        diagnostic_rf = diagnostic[
            (diagnostic["diagnostic_split"] == "subject_aware") & (diagnostic["model"] == "rf")
        ].copy()
        diagnostic_rf["model"] = "diagnostic_rf"
        comparison_rows.append(diagnostic_rf)

    svm = svm_rows[svm_rows["split"].isin(["val", "test"])].copy()
    comparison_rows.append(svm)

    combined = pd.concat(comparison_rows, ignore_index=True, sort=False)
    columns = ["model", "split", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    combined = combined[columns]
    order = ["original_logreg", "original_rf", "original_cnn", "diagnostic_rf", "svm"]
    combined["model"] = pd.Categorical(combined["model"], categories=order, ordered=True)
    combined = combined.sort_values(["model", "split"])
    combined.to_csv(METRICS_COMPARISON_PATH, index=False)


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

    model, best_params, candidate_metrics = select_svm(X[train_mask], y[train_mask], X[val_mask], y[val_mask], args.seed)
    chosen_params = format_params(best_params)
    notes = "selected by validation macro_f1; duration=10s; feature_set=combined; subject-aware split"

    val_pred = model.predict(X[val_mask])
    test_pred = model.predict(X[test_mask])
    selected_rows = pd.DataFrame(
        [
            metric_row("val", y[val_mask], val_pred, chosen_params, notes),
            metric_row("test", y[test_mask], test_pred, chosen_params, notes),
        ]
    )

    metrics_out = pd.concat([candidate_metrics, selected_rows], ignore_index=True)
    metrics_out.to_csv(METRICS_SVM_PATH, index=False)
    save_test_artifacts(y[test_mask], test_pred, labels)
    update_comparison(selected_rows)

    print("Best SVM params:")
    print(chosen_params)
    print("\nSelected validation/test metrics:")
    print(selected_rows.to_string(index=False))


if __name__ == "__main__":
    main()
