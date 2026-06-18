"""Train improved Logistic Regression and Random Forest baselines."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_summary_features
from utils import (
    FEATURES_DIR,
    FIGURES_DIR,
    METRICS_COMPARISON_PATH,
    METRICS_IMPROVED_PATH,
    METRICS_ORIGINAL_PATH,
    RANDOM_SEED,
    REPORTS_DIR,
    SPLITS_PATH,
    ensure_project_dirs,
    set_seed,
)


def cache_path(augment_train: bool) -> Path:
    suffix = "augmented_train" if augment_train else "clean"
    return FEATURES_DIR / f"improved_summary_{suffix}.npz"


def load_or_extract_features(splits: pd.DataFrame, augment_train: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = cache_path(augment_train)
    if path.exists():
        cached = np.load(path, allow_pickle=True)
        return cached["X"], cached["y"]

    rng = np.random.default_rng(seed)
    features = []
    labels = []
    for index, row in splits.iterrows():
        apply_aug = augment_train and row["split"] == "train"
        features.append(extract_summary_features(row["filepath"], apply_aug=apply_aug, rng=rng))
        labels.append(row["song_label"])
        if (index + 1) % 100 == 0:
            print(f"Extracted summary features for {index + 1}/{len(splits)} files")

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    return X, y


def metric_row(model: str, split: str, y_true: np.ndarray, y_pred: np.ndarray, notes: str) -> dict[str, object]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "model": model,
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "notes": notes,
    }


def save_report(model: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    path = REPORTS_DIR / f"improved_{model}_{split}.txt"
    path.write_text(classification_report(y_true, y_pred, zero_division=0), encoding="utf-8")


def save_confusion_matrix(model: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, cmap="Greens", xticks_rotation=45, colorbar=False)
    ax.set_title(f"Improved {model} test confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_matrix_improved_{model}.png", dpi=200)
    plt.close(fig)


def validation_macro_f1(model: object, X_val: np.ndarray, y_val: np.ndarray) -> float:
    pred = model.predict(X_val)
    return precision_recall_fscore_support(y_val, pred, average="macro", zero_division=0)[2]


def select_logreg(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int) -> tuple[object, str]:
    best_model = None
    best_score = -np.inf
    best_notes = ""
    for C, class_weight, pca_option in product([0.01, 0.1, 1.0, 10.0], [None, "balanced"], [None, 0.95]):
        steps = [("scaler", StandardScaler())]
        if pca_option is not None:
            steps.append(("pca", PCA(n_components=pca_option, random_state=seed)))
        steps.append(
            (
                "logreg",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            )
        )
        model = Pipeline(steps)
        model.fit(X_train, y_train)
        score = validation_macro_f1(model, X_val, y_val)
        if score > best_score:
            best_score = score
            best_model = model
            best_notes = f"C={C}; class_weight={class_weight}; pca={pca_option}; val_macro_f1={score:.4f}"

    if best_model is None:
        raise RuntimeError("No Logistic Regression model was selected.")
    return best_model, best_notes


def select_rf(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int) -> tuple[object, str]:
    best_model = None
    best_score = -np.inf
    best_notes = ""
    for n_estimators, max_depth, min_samples_leaf, class_weight in product(
        [300, 600],
        [None, 10, 20],
        [1, 3, 5],
        [None, "balanced_subsample"],
    ):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        score = validation_macro_f1(model, X_val, y_val)
        if score > best_score:
            best_score = score
            best_model = model
            best_notes = (
                f"n_estimators={n_estimators}; max_depth={max_depth}; "
                f"min_samples_leaf={min_samples_leaf}; class_weight={class_weight}; val_macro_f1={score:.4f}"
            )

    if best_model is None:
        raise RuntimeError("No Random Forest model was selected.")
    return best_model, best_notes


def write_metrics_comparison() -> None:
    if not METRICS_ORIGINAL_PATH.exists() or not METRICS_IMPROVED_PATH.exists():
        return

    original = pd.read_csv(METRICS_ORIGINAL_PATH).assign(run="original")
    improved = pd.read_csv(METRICS_IMPROVED_PATH).assign(run="improved")
    combined = pd.concat([original, improved], ignore_index=True, sort=False)
    combined["model"] = combined["run"] + "_" + combined["model"]
    combined = combined[
        ["model", "split", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    ]
    order = [
        "original_logreg",
        "improved_logreg",
        "original_rf",
        "improved_rf",
        "original_cnn",
        "improved_cnn",
    ]
    combined["model"] = pd.Categorical(combined["model"], categories=order, ordered=True)
    combined = combined.sort_values(["model", "split"])
    combined.to_csv(METRICS_COMPARISON_PATH, index=False)


def append_metrics(rows: list[dict[str, object]]) -> None:
    new_metrics = pd.DataFrame(rows)
    if METRICS_IMPROVED_PATH.exists():
        previous = pd.read_csv(METRICS_IMPROVED_PATH)
        previous = previous[~previous["model"].isin(new_metrics["model"].unique())]
        new_metrics = pd.concat([previous, new_metrics], ignore_index=True)
    new_metrics.to_csv(METRICS_IMPROVED_PATH, index=False)
    write_metrics_comparison()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--augment-train", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()
    set_seed(args.seed)
    splits = pd.read_csv(args.splits)
    X, y = load_or_extract_features(splits, augment_train=args.augment_train, seed=args.seed)

    train_mask = splits["split"].to_numpy() == "train"
    val_mask = splits["split"].to_numpy() == "val"
    test_mask = splits["split"].to_numpy() == "test"
    labels = sorted(pd.unique(y))

    selected = {
        "logreg": select_logreg(X[train_mask], y[train_mask], X[val_mask], y[val_mask], args.seed),
        "rf": select_rf(X[train_mask], y[train_mask], X[val_mask], y[val_mask], args.seed),
    }

    rows = []
    for model_name, (model, notes) in selected.items():
        print(f"Selected {model_name}: {notes}")
        for split, mask in [("val", val_mask), ("test", test_mask)]:
            pred = model.predict(X[mask])
            rows.append(metric_row(model_name, split, y[mask], pred, notes))
            save_report(model_name, split, y[mask], pred)
            if split == "test":
                save_confusion_matrix(model_name, y[mask], pred, labels)

    append_metrics(rows)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
