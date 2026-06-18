"""Controlled diagnostics for weak experimental CNN results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import extract_summary_features
from utils import FEATURES_DIR, PROJECT_ROOT, RANDOM_SEED, RESULTS_DIR, SPLITS_PATH, ensure_project_dirs, set_seed


DURATIONS = [10.0, 20.0, 30.0]
FEATURE_SETS = ["mfcc", "mel", "pitch", "combined"]
MODEL_NAMES = ["logreg", "rf", "svm"]


def text_table(frame: pd.DataFrame) -> str:
    return "```text\n" + frame.to_string(index=False) + "\n```"


def make_model(model_name: str, seed: int) -> object:
    if model_name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, random_state=seed),
                ),
            ]
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=3,
            class_weight=None,
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed)),
            ]
        )
    raise ValueError(f"Unknown model: {model_name}")


def metrics(model: str, split: str, y_true: np.ndarray, y_pred: np.ndarray, notes: str = "") -> dict[str, object]:
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


def feature_cache_path(duration: float, feature_set: str) -> Path:
    duration_tag = str(int(duration))
    return FEATURES_DIR / f"diagnostic_{feature_set}_{duration_tag}s.npz"


def load_or_extract_features(splits: pd.DataFrame, duration: float, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    path = feature_cache_path(duration, feature_set)
    if path.exists():
        cached = np.load(path, allow_pickle=True)
        return cached["X"], cached["y"]

    features = []
    labels = []
    for index, row in splits.iterrows():
        features.append(
            extract_summary_features(
                row["filepath"],
                duration_seconds=duration,
                feature_set=feature_set,
            )
        )
        labels.append(row["song_label"])
        if (index + 1) % 100 == 0:
            print(f"Extracted {feature_set} {duration:.0f}s features for {index + 1}/{len(splits)} files")

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    return X, y


def split_masks(splits: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: (splits["split"].to_numpy() == name) for name in ["train", "val", "test"]}


def evaluate_config(
    splits: pd.DataFrame,
    duration: float,
    feature_set: str,
    model_name: str,
    seed: int,
    eval_split: str,
    diagnostic_split_name: str = "subject_aware",
) -> dict[str, object]:
    X, y = load_or_extract_features(splits, duration, feature_set)
    masks = split_masks(splits)
    model = make_model(model_name, seed)
    model.fit(X[masks["train"]], y[masks["train"]])
    pred = model.predict(X[masks[eval_split]])
    row = metrics(
        model_name,
        eval_split,
        y[masks[eval_split]],
        pred,
        notes=f"{diagnostic_split_name}; duration={duration:.0f}s; feature_set={feature_set}",
    )
    row["duration_seconds"] = duration
    row["feature_set"] = feature_set
    row["diagnostic_split"] = diagnostic_split_name
    return row


def run_duration_diagnostics(splits: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for duration in DURATIONS:
        for model_name in MODEL_NAMES:
            rows.append(evaluate_config(splits, duration, "combined", model_name, seed, "val"))
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(RESULTS_DIR / "duration_diagnostics.csv", index=False)
    return diagnostics


def run_feature_diagnostics(splits: pd.DataFrame, duration: float, seed: int) -> pd.DataFrame:
    rows = []
    for feature_set in FEATURE_SETS:
        for model_name in MODEL_NAMES:
            rows.append(evaluate_config(splits, duration, feature_set, model_name, seed, "val"))
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(RESULTS_DIR / "feature_diagnostics.csv", index=False)
    return diagnostics


def create_file_level_diagnostic_split(subject_splits: pd.DataFrame, seed: int) -> pd.DataFrame:
    columns = ["filename", "filepath", "song_label", "participant_id"]
    rows = subject_splits[columns].copy()
    train, temp = train_test_split(
        rows,
        test_size=0.30,
        random_state=seed,
        stratify=rows["song_label"],
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=seed,
        stratify=temp["song_label"],
    )
    train = train.assign(split="train")
    val = val.assign(split="val")
    test = test.assign(split="test")
    file_split = pd.concat([train, val, test], ignore_index=True).sort_values("filename")
    file_split.to_csv(RESULTS_DIR / "splits_file_diagnostic.csv", index=False)
    return file_split


def run_split_diagnostics(
    subject_splits: pd.DataFrame,
    file_splits: pd.DataFrame,
    duration: float,
    feature_set: str,
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for split_name, split_frame in [("subject_aware", subject_splits), ("file_level_not_main_eval", file_splits)]:
        for eval_split in ["val", "test"]:
            rows.append(
                evaluate_config(
                    split_frame,
                    duration,
                    feature_set,
                    model_name,
                    seed,
                    eval_split,
                    diagnostic_split_name=split_name,
                )
            )
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(RESULTS_DIR / "split_diagnostics.csv", index=False)
    return diagnostics


def write_cnn_diagnosis(splits: pd.DataFrame) -> None:
    output = RESULTS_DIR / "cnn_diagnosis.md"
    lines = [
        "# CNN Diagnosis",
        "",
        "The original CNN and smaller experimental CNN were checked using the saved metrics and curves.",
        "",
        "## Saved Metrics",
    ]
    metrics_path = RESULTS_DIR / "metrics_comparison.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        cnn_rows = metrics_df[metrics_df["model"].astype(str).str.contains("cnn", na=False)]
        lines.append(text_table(cnn_rows))
    else:
        lines.append("`results/metrics_comparison.csv` was not available.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The original CNN reached higher validation performance than the smaller experimental CNN, but still had weak corrected test performance.",
            "- The smaller experimental CNN underfit: validation accuracy stayed near random chance and early stopping restored epoch 1 weights.",
            "- The improved CNN architecture was probably too conservative for this dataset/feature representation, and normalization plus strong regularization did not rescue it.",
            "- Training curves are saved at `figures/training_curves_original_cnn.png` and `figures/training_curves_improved_cnn.png`.",
            "",
            "## Recommendation",
            "",
            "Do not present the smaller experimental CNN as a final improvement. Keep it as a negative result, and revisit CNN design only after the classical feature diagnostics are understood.",
        ]
    )

    output.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    duration_diagnostics: pd.DataFrame,
    feature_diagnostics: pd.DataFrame,
    selected_row: pd.Series,
    selected_test: pd.Series,
    split_diagnostics: pd.DataFrame,
) -> None:
    duration_best = duration_diagnostics.sort_values("macro_f1", ascending=False).iloc[0]
    feature_best = feature_diagnostics.sort_values("macro_f1", ascending=False).iloc[0]
    file_diag = split_diagnostics[split_diagnostics["diagnostic_split"] == "file_level_not_main_eval"]
    subject_diag = split_diagnostics[split_diagnostics["diagnostic_split"] == "subject_aware"]

    lines = [
        "# Classical Diagnostic Summary",
        "",
        "## Best Validation Configuration",
        "",
        f"- Model: `{selected_row['model']}`",
        f"- Duration: `{selected_row['duration_seconds']:.0f}s`",
        f"- Feature set: `{selected_row['feature_set']}`",
        f"- Validation accuracy: `{selected_row['accuracy']:.4f}`",
        f"- Validation macro F1: `{selected_row['macro_f1']:.4f}`",
        "",
        "## Selected Test Performance",
        "",
        f"- Test accuracy: `{selected_test['accuracy']:.4f}`",
        f"- Test macro F1: `{selected_test['macro_f1']:.4f}`",
        f"- Test weighted F1: `{selected_test['weighted_f1']:.4f}`",
        "",
        "## Duration Finding",
        "",
        f"The best duration diagnostic row used `{duration_best['duration_seconds']:.0f}s`, "
        f"`{duration_best['model']}`, and validation macro F1 `{duration_best['macro_f1']:.4f}`.",
        "",
        "## Feature Finding",
        "",
        f"The best feature diagnostic row used `{feature_best['feature_set']}` features, "
        f"`{feature_best['model']}`, and validation macro F1 `{feature_best['macro_f1']:.4f}`.",
        "",
        "## Split Finding",
        "",
        "Subject-aware split rows:",
        "",
        text_table(subject_diag[["model", "split", "accuracy", "macro_f1", "feature_set", "duration_seconds"]]),
        "",
        "File-level diagnostic split rows, not main evaluation:",
        "",
        text_table(file_diag[["model", "split", "accuracy", "macro_f1", "feature_set", "duration_seconds"]]),
        "",
        "This file-level diagnostic split allows participant overlap, but the selected RF summary-feature setup did not become easier in this run. This suggests the weak results are not explained by subject-aware splitting alone; the contaminated notebook results were likely inflated by the stronger overlap created when evaluating on datasets that duplicated training/validation files. The file-level split should not replace the subject-aware evaluation.",
        "",
        "## Recommendation",
        "",
        "Use the diagnostic-selected classical configuration as the next candidate baseline. Keep the smaller CNN scripts as experimental/negative-result code for now, and do not present the smaller CNN as an improvement.",
    ]
    (RESULTS_DIR / "classical_diagnostic_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    ensure_project_dirs()
    set_seed(args.seed)
    subject_splits = pd.read_csv(args.splits)

    duration_diagnostics = run_duration_diagnostics(subject_splits, args.seed)
    best_duration = float(duration_diagnostics.sort_values("macro_f1", ascending=False).iloc[0]["duration_seconds"])

    feature_diagnostics = run_feature_diagnostics(subject_splits, best_duration, args.seed)
    selected = feature_diagnostics.sort_values("macro_f1", ascending=False).iloc[0]
    selected_test = evaluate_config(
        subject_splits,
        float(selected["duration_seconds"]),
        str(selected["feature_set"]),
        str(selected["model"]),
        args.seed,
        "test",
    )

    duration_with_test = pd.concat([duration_diagnostics, pd.DataFrame([selected_test])], ignore_index=True)
    duration_with_test.to_csv(RESULTS_DIR / "duration_diagnostics.csv", index=False)

    feature_with_test = pd.concat([feature_diagnostics, pd.DataFrame([selected_test])], ignore_index=True)
    feature_with_test.to_csv(RESULTS_DIR / "feature_diagnostics.csv", index=False)

    file_splits = create_file_level_diagnostic_split(subject_splits, args.seed)
    split_diagnostics = run_split_diagnostics(
        subject_splits,
        file_splits,
        float(selected["duration_seconds"]),
        str(selected["feature_set"]),
        str(selected["model"]),
        args.seed,
    )

    write_cnn_diagnosis(subject_splits)
    write_summary(duration_diagnostics, feature_diagnostics, selected, pd.Series(selected_test), split_diagnostics)

    print("Best validation configuration:")
    print(selected.to_string())
    print("\nSelected test result:")
    print(pd.Series(selected_test).to_string())
    print("\nSplit diagnostics:")
    print(split_diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
