"""Train original coursework-style classical baselines on corrected splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from features import extract_classical_features
from utils import (
    FEATURES_DIR,
    FIGURES_DIR,
    METRICS_ORIGINAL_PATH,
    RANDOM_SEED,
    REPORTS_DIR,
    SPLITS_PATH,
    ensure_project_dirs,
    set_seed,
)


def _cache_path(augment_train: bool) -> Path:
    suffix = "augmented_train" if augment_train else "clean"
    return FEATURES_DIR / f"original_mfcc_{suffix}.npz"


def load_or_extract_features(splits: pd.DataFrame, augment_train: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    cache_path = _cache_path(augment_train)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached["X"], cached["y"]

    rng = np.random.default_rng(seed)
    features = []
    labels = []
    for index, row in splits.iterrows():
        apply_aug = augment_train and row["split"] == "train"
        features.append(extract_classical_features(row["filepath"], apply_aug=apply_aug, rng=rng))
        labels.append(row["song_label"])
        if (index + 1) % 100 == 0:
            print(f"Extracted MFCC features for {index + 1}/{len(splits)} files")

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def metric_row(model_name: str, split_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def save_report(model_name: str, split_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    report = classification_report(y_true, y_pred, zero_division=0)
    (REPORTS_DIR / f"original_{model_name}_{split_name}.txt").write_text(report, encoding="utf-8")


def save_confusion_matrix(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"Original {model_name} test confusion matrix")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def append_metrics(rows: list[dict[str, object]]) -> None:
    new_metrics = pd.DataFrame(rows)
    if METRICS_ORIGINAL_PATH.exists():
        previous = pd.read_csv(METRICS_ORIGINAL_PATH)
        previous = previous[~previous["model"].isin(new_metrics["model"].unique())]
        new_metrics = pd.concat([previous, new_metrics], ignore_index=True)
    new_metrics.to_csv(METRICS_ORIGINAL_PATH, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--augment-train",
        action="store_true",
        help="Apply a deterministic approximation of the notebook's training-only augmentation.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    set_seed(args.seed)
    splits = pd.read_csv(args.splits)
    X, y = load_or_extract_features(splits, augment_train=args.augment_train, seed=args.seed)

    train_mask = splits["split"].to_numpy() == "train"
    val_mask = splits["split"].to_numpy() == "val"
    test_mask = splits["split"].to_numpy() == "test"
    labels = sorted(pd.unique(y))

    models = {
        "logreg": LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial"),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=args.seed,
        ),
    }

    rows = []
    for model_name, model in models.items():
        print(f"Training {model_name}")
        model.fit(X[train_mask], y[train_mask])
        for split_name, mask in [("val", val_mask), ("test", test_mask)]:
            predictions = model.predict(X[mask])
            rows.append(metric_row(model_name, split_name, y[mask], predictions))
            save_report(model_name, split_name, y[mask], predictions)
            if split_name == "test":
                output = FIGURES_DIR / f"confusion_matrix_original_{model_name}.png"
                save_confusion_matrix(model_name, y[mask], predictions, labels, output)

    append_metrics(rows)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
