"""Build file-level metadata for the local MLEndHWII 800-file dataset."""

from __future__ import annotations

import argparse
import re
import wave
from pathlib import Path

import pandas as pd

from utils import DATASET_DIR, METADATA_PATH, ensure_project_dirs, relative_to_project


FILENAME_PATTERN = re.compile(
    r"^(?P<participant_id>S\d+)_(?P<recording_type>hum|whistle)_"
    r"(?P<take_number>\d+)_(?P<song_label>.+)\.wav$",
    re.IGNORECASE,
)


def parse_filename(path: Path) -> dict[str, object]:
    """Parse dataset labels encoded in filenames such as S100_hum_2_Married.wav."""
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        return {
            "song_label": None,
            "participant_id": None,
            "recording_type": None,
            "take_number": None,
        }

    values = match.groupdict()
    return {
        "song_label": values["song_label"],
        "participant_id": values["participant_id"].upper(),
        "recording_type": values["recording_type"].lower(),
        "take_number": int(values["take_number"]),
    }


def read_audio_info(path: Path) -> dict[str, object]:
    """Read basic audio properties without loading the full waveform."""
    try:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            frames = wav_file.getnframes()
    except wave.Error:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                f"Could not read {path.name} with the standard wave module, "
                "and optional dependency soundfile is not installed."
            ) from exc

        info = sf.info(str(path))
        sample_rate = int(info.samplerate)
        num_channels = int(info.channels)
        frames = int(info.frames)

    duration_seconds = float(frames) / float(sample_rate)
    return {
        "sample_rate": int(sample_rate),
        "num_channels": int(num_channels),
        "duration_seconds": duration_seconds,
    }


def build_metadata(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            "Expected local audio under data/MLEndHWII_sample_800/."
        )

    rows = []
    for path in sorted(dataset_dir.glob("*.wav")):
        row = {
            "filename": path.name,
            "filepath": relative_to_project(path),
            **parse_filename(path),
            **read_audio_info(path),
        }
        rows.append(row)

    if not rows:
        raise ValueError(f"No .wav files found in {dataset_dir}.")

    columns = [
        "filename",
        "filepath",
        "song_label",
        "participant_id",
        "recording_type",
        "take_number",
        "sample_rate",
        "num_channels",
        "duration_seconds",
    ]
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory containing the 800-file .wav dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=METADATA_PATH,
        help="Path to write metadata CSV.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    metadata = build_metadata(args.dataset_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output, index=False)

    print(f"Wrote {len(metadata)} rows to {relative_to_project(args.output)}")
    print(f"Participants: {metadata['participant_id'].nunique()}")
    print(f"Song labels: {metadata['song_label'].nunique()}")


if __name__ == "__main__":
    main()
