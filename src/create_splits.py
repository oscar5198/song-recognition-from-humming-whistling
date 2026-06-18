"""Create subject-aware train/validation/test splits from metadata.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import METADATA_PATH, RANDOM_SEED, SPLITS_PATH, relative_to_project, set_seed


SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
N_SEARCH_ITERATIONS = 5_000


def _class_distribution(frame: pd.DataFrame, labels: list[str]) -> np.ndarray:
    counts = frame["song_label"].value_counts().reindex(labels, fill_value=0).to_numpy(dtype=float)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(labels), dtype=float)
    return counts / total


def _score_split(metadata: pd.DataFrame, assignments: dict[str, str], labels: list[str]) -> float:
    total_rows = len(metadata)
    global_distribution = _class_distribution(metadata, labels)
    score = 0.0

    for split_name, target_fraction in SPLIT_FRACTIONS.items():
        split_frame = metadata[metadata["participant_id"].map(assignments) == split_name]
        actual_fraction = len(split_frame) / total_rows
        size_penalty = abs(actual_fraction - target_fraction)
        balance_penalty = np.abs(_class_distribution(split_frame, labels) - global_distribution).mean()
        empty_class_penalty = (split_frame["song_label"].nunique() < len(labels)) * 0.25
        score += size_penalty + balance_penalty + empty_class_penalty

    return float(score)


def _assignment_from_order(metadata: pd.DataFrame, participants: np.ndarray) -> dict[str, str]:
    participant_sizes = metadata.groupby("participant_id").size().to_dict()
    target_sizes = {name: fraction * len(metadata) for name, fraction in SPLIT_FRACTIONS.items()}
    split_counts = {name: 0 for name in SPLIT_FRACTIONS}
    assignments: dict[str, str] = {}

    for participant_id in participants:
        size = participant_sizes[participant_id]
        best_split = min(
            SPLIT_FRACTIONS,
            key=lambda split_name: (
                (split_counts[split_name] + size - target_sizes[split_name]) / target_sizes[split_name],
                split_counts[split_name],
            ),
        )
        assignments[participant_id] = best_split
        split_counts[best_split] += size

    return assignments


def choose_subject_aware_split(
    metadata: pd.DataFrame,
    seed: int = RANDOM_SEED,
    n_iterations: int = N_SEARCH_ITERATIONS,
) -> dict[str, str]:
    if metadata[["participant_id", "song_label"]].isna().any().any():
        missing = metadata[metadata[["participant_id", "song_label"]].isna().any(axis=1)]
        raise ValueError(
            "Cannot create subject-aware splits because participant_id or song_label "
            f"is missing for {len(missing)} rows."
        )

    participants = metadata["participant_id"].drop_duplicates().to_numpy()
    labels = sorted(metadata["song_label"].unique())
    rng = np.random.default_rng(seed)

    best_assignments: dict[str, str] | None = None
    best_score = float("inf")

    for _ in range(n_iterations):
        shuffled = rng.permutation(participants)
        assignments = _assignment_from_order(metadata, shuffled)
        score = _score_split(metadata, assignments, labels)
        if score < best_score:
            best_score = score
            best_assignments = assignments

    if best_assignments is None:
        raise RuntimeError("Failed to build split assignments.")

    return best_assignments


def validate_no_subject_overlap(splits: pd.DataFrame) -> None:
    participant_split_counts = splits.groupby("participant_id")["split"].nunique()
    overlapping = participant_split_counts[participant_split_counts > 1]
    if not overlapping.empty:
        raise AssertionError(
            "Subject leakage detected. Participants in multiple splits: "
            + ", ".join(overlapping.index.astype(str))
        )


def create_splits(metadata_path: Path = METADATA_PATH, seed: int = RANDOM_SEED) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    required_columns = ["filename", "filepath", "song_label", "participant_id"]
    missing_columns = [column for column in required_columns if column not in metadata.columns]
    if missing_columns:
        raise ValueError(f"metadata.csv is missing required columns: {missing_columns}")

    set_seed(seed)
    assignments = choose_subject_aware_split(metadata, seed=seed)
    splits = metadata[required_columns].copy()
    splits["split"] = splits["participant_id"].map(assignments)
    validate_no_subject_overlap(splits)

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH, help="Input metadata CSV.")
    parser.add_argument("--output", type=Path, default=SPLITS_PATH, help="Output splits CSV.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Fixed random seed.")
    args = parser.parse_args()

    splits = create_splits(args.metadata, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.output, index=False)

    print(f"Wrote {len(splits)} rows to {relative_to_project(args.output)}")
    print("Split counts:")
    print(splits["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    print("Unique participants per split:")
    print(splits.groupby("split")["participant_id"].nunique().reindex(["train", "val", "test"]).to_string())
    validate_no_subject_overlap(splits)
    print("Subject overlap check: passed")


if __name__ == "__main__":
    main()
