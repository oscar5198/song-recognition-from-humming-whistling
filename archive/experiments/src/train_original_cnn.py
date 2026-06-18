"""Train the original coursework-style CNN on corrected splits."""

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
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.utils import to_categorical

from features import CNN_TARGET_FRAMES, N_MELS, extract_cnn_features
from utils import (
    FEATURES_DIR,
    FIGURES_DIR,
    METRICS_ORIGINAL_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    SPLITS_PATH,
    ensure_project_dirs,
    set_seed,
)


def _cache_path(augment_train: bool) -> Path:
    suffix = "augmented_train" if augment_train else "clean"
    return FEATURES_DIR / f"original_logmel_{suffix}.npz"


def load_or_extract_features(splits: pd.DataFrame, augment_train: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    cache_path = _cache_path(augment_train)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached["X"], cached["y"]

    rng = np.random.default_rng(seed)
    specs = []
    labels = []
    for index, row in splits.iterrows():
        apply_aug = augment_train and row["split"] == "train"
        specs.append(extract_cnn_features(row["filepath"], apply_aug=apply_aug, rng=rng))
        labels.append(row["song_label"])
        if (index + 1) % 100 == 0:
            print(f"Extracted log-mel features for {index + 1}/{len(splits)} files")

    X = np.stack(specs).astype(np.float32)
    y = np.asarray(labels)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def build_cnn(input_shape: tuple[int, int, int], num_classes: int) -> models.Model:
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding="same", activation=None)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


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


def save_outputs(
    history: tf.keras.callbacks.History,
    labels: list[str],
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
) -> list[dict[str, object]]:
    (REPORTS_DIR / "original_cnn_val.txt").write_text(
        classification_report(y_val_true, y_val_pred, zero_division=0), encoding="utf-8"
    )
    (REPORTS_DIR / "original_cnn_test.txt").write_text(
        classification_report(y_test_true, y_test_pred, zero_division=0), encoding="utf-8"
    )

    matrix = confusion_matrix(y_test_true, y_test_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Original CNN test confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_original_cnn.png", dpi=200)
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
    fig.savefig(FIGURES_DIR / "training_curves_original_cnn.png", dpi=200)
    plt.close(fig)

    return [
        metric_row("cnn", "val", y_val_true, y_val_pred),
        metric_row("cnn", "test", y_test_true, y_test_pred),
    ]


def append_metrics(rows: list[dict[str, object]]) -> None:
    new_metrics = pd.DataFrame(rows)
    if METRICS_ORIGINAL_PATH.exists():
        previous = pd.read_csv(METRICS_ORIGINAL_PATH)
        previous = previous[previous["model"] != "cnn"]
        new_metrics = pd.concat([previous, new_metrics], ignore_index=True)
    new_metrics.to_csv(METRICS_ORIGINAL_PATH, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--augment-train",
        action="store_true",
        help="Apply a deterministic approximation of the notebook's training-only augmentation.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    set_seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    splits = pd.read_csv(args.splits)
    X, y = load_or_extract_features(splits, augment_train=args.augment_train, seed=args.seed)

    train_mask = splits["split"].to_numpy() == "train"
    val_mask = splits["split"].to_numpy() == "val"
    test_mask = splits["split"].to_numpy() == "test"

    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    labels = list(label_encoder.classes_)

    y_train = to_categorical(label_encoder.transform(y[train_mask]), len(labels))
    y_val = to_categorical(label_encoder.transform(y[val_mask]), len(labels))

    model = build_cnn((N_MELS, CNN_TARGET_FRAMES, 1), len(labels))
    checkpoint_path = MODELS_DIR / "original_cnn_best.keras"
    history = model.fit(
        X[train_mask],
        y_train,
        validation_data=(X[val_mask], y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
            callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
        ],
        verbose=2,
    )

    val_pred = label_encoder.inverse_transform(np.argmax(model.predict(X[val_mask], verbose=0), axis=1))
    test_pred = label_encoder.inverse_transform(np.argmax(model.predict(X[test_mask], verbose=0), axis=1))

    rows = save_outputs(history, labels, y[val_mask], val_pred, y[test_mask], test_pred)
    append_metrics(rows)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
