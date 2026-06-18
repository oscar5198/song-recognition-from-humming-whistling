"""Train a smaller improved CNN on corrected log-mel splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers
from tensorflow.keras.utils import to_categorical

from features import CNN_TARGET_FRAMES, N_MELS, extract_cnn_features
from train_improved_baselines import write_metrics_comparison
from utils import (
    FEATURES_DIR,
    FIGURES_DIR,
    METRICS_IMPROVED_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    SPLITS_PATH,
    ensure_project_dirs,
    set_seed,
)


def cache_path(augment_train: bool) -> Path:
    suffix = "augmented_train" if augment_train else "clean"
    return FEATURES_DIR / f"improved_logmel_{suffix}.npz"


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
        features.append(extract_cnn_features(row["filepath"], apply_aug=apply_aug, rng=rng))
        labels.append(row["song_label"])
        if (index + 1) % 100 == 0:
            print(f"Extracted improved log-mel features for {index + 1}/{len(splits)} files")

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    return X, y


def normalize_from_train(X: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    mean = X[train_mask].mean(axis=(0, 1, 2), keepdims=True)
    std = X[train_mask].std(axis=(0, 1, 2), keepdims=True)
    return ((X - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def build_improved_cnn(input_shape: tuple[int, int, int], num_classes: int) -> models.Model:
    weight_decay = 1e-4
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


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


def save_outputs(
    history: tf.keras.callbacks.History,
    labels: list[str],
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
) -> None:
    (REPORTS_DIR / "improved_cnn_val.txt").write_text(
        classification_report(y_val_true, y_val_pred, zero_division=0), encoding="utf-8"
    )
    (REPORTS_DIR / "improved_cnn_test.txt").write_text(
        classification_report(y_test_true, y_test_pred, zero_division=0), encoding="utf-8"
    )

    matrix = confusion_matrix(y_test_true, y_test_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, cmap="Greens", xticks_rotation=45, colorbar=False)
    ax.set_title("Improved CNN test confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_improved_cnn.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history.get("loss", []), label="train")
    axes[0].plot(history.history.get("val_loss", []), label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history.get("accuracy", []), label="train")
    axes[1].plot(history.history.get("val_accuracy", []), label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves_improved_cnn.png", dpi=200)
    plt.close(fig)


def append_metrics(rows: list[dict[str, object]]) -> None:
    new_metrics = pd.DataFrame(rows)
    if METRICS_IMPROVED_PATH.exists():
        previous = pd.read_csv(METRICS_IMPROVED_PATH)
        previous = previous[previous["model"] != "cnn"]
        new_metrics = pd.concat([previous, new_metrics], ignore_index=True)
    new_metrics.to_csv(METRICS_IMPROVED_PATH, index=False)
    write_metrics_comparison()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augment-train", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()
    set_seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    splits = pd.read_csv(args.splits)
    X, y = load_or_extract_features(splits, augment_train=args.augment_train, seed=args.seed)
    train_mask = splits["split"].to_numpy() == "train"
    val_mask = splits["split"].to_numpy() == "val"
    test_mask = splits["split"].to_numpy() == "test"
    X = normalize_from_train(X, train_mask)

    encoder = LabelEncoder()
    encoder.fit(y)
    labels = list(encoder.classes_)
    y_train = to_categorical(encoder.transform(y[train_mask]), len(labels))
    y_val = to_categorical(encoder.transform(y[val_mask]), len(labels))

    model = build_improved_cnn((N_MELS, CNN_TARGET_FRAMES, 1), len(labels))
    checkpoint = MODELS_DIR / "improved_cnn_best.keras"
    history = model.fit(
        X[train_mask],
        y_train,
        validation_data=(X[val_mask], y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
            callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True, verbose=1),
        ],
        verbose=2,
    )

    val_pred = encoder.inverse_transform(np.argmax(model.predict(X[val_mask], verbose=0), axis=1))
    test_pred = encoder.inverse_transform(np.argmax(model.predict(X[test_mask], verbose=0), axis=1))
    notes = (
        "logmel train-stat normalization; conv filters 16/32/64; "
        "dropout 0.15/0.20/0.25/0.35; Adam lr=5e-4; early stopping patience=8"
    )
    rows = [
        metric_row("cnn", "val", y[val_mask], val_pred, notes),
        metric_row("cnn", "test", y[test_mask], test_pred, notes),
    ]
    save_outputs(history, labels, y[val_mask], val_pred, y[test_mask], test_pred)
    append_metrics(rows)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
