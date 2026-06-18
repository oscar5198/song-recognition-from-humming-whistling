"""Original-coursework style audio preprocessing and feature extraction."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.signal import resample_poly

from utils import PROJECT_ROOT, TARGET_AUDIO_DURATION_SECONDS, TARGET_SAMPLE_RATE


N_FFT = 1024
HOP_LENGTH = 512
N_MFCC = 30
N_MELS = 64
CNN_TARGET_FRAMES = 312
MEL_FMIN = 40


def resolve_audio_path(filepath: str | Path) -> Path:
    path = Path(filepath)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_audio(filepath: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono float32 audio."""
    sample_rate, audio = wavfile.read(resolve_audio_path(filepath))
    audio = np.asarray(audio)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_value = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)

    return audio, int(sample_rate)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak == 0:
        return audio.astype(np.float32)
    return (audio / peak).astype(np.float32)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def pad_or_truncate(
    audio: np.ndarray,
    target_samples: int = int(TARGET_SAMPLE_RATE * TARGET_AUDIO_DURATION_SECONDS),
) -> np.ndarray:
    if len(audio) > target_samples:
        return audio[:target_samples].astype(np.float32)
    if len(audio) < target_samples:
        return np.pad(audio, (0, target_samples - len(audio)), mode="constant").astype(np.float32)
    return audio.astype(np.float32)


def augment_audio(audio: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate the notebook's one-of pitch/time/noise augmentation."""
    choice = rng.choice(["pitch", "stretch", "noise"])
    if choice == "noise":
        return (audio + 0.005 * rng.standard_normal(len(audio))).astype(np.float32)

    if choice == "stretch":
        rate = 1.0 + rng.uniform(-0.15, 0.15)
        stretched_len = max(1, int(round(len(audio) / rate)))
        stretched = resample_poly(audio, stretched_len, len(audio))
        return pad_or_truncate(stretched)

    steps = rng.uniform(-2.0, 2.0)
    factor = 2.0 ** (steps / 12.0)
    shifted_len = max(1, int(round(len(audio) / factor)))
    shifted = resample_poly(audio, shifted_len, len(audio))
    shifted = pad_or_truncate(shifted)
    return resample_poly(shifted, len(audio), len(shifted)).astype(np.float32)


def preprocess_audio(
    filepath: str | Path,
    apply_aug: bool = False,
    rng: np.random.Generator | None = None,
    duration_seconds: float = TARGET_AUDIO_DURATION_SECONDS,
) -> np.ndarray:
    audio, sample_rate = load_audio(filepath)
    audio = normalize_audio(audio)
    audio = resample_audio(audio, sample_rate, TARGET_SAMPLE_RATE)
    target_samples = int(TARGET_SAMPLE_RATE * duration_seconds)
    audio = pad_or_truncate(audio, target_samples=target_samples)

    if apply_aug:
        if rng is None:
            rng = np.random.default_rng()
        audio = augment_audio(audio, TARGET_SAMPLE_RATE, rng)

    return audio.astype(np.float32)


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(
    sample_rate: int = TARGET_SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = MEL_FMIN,
    fmax: float | None = None,
) -> np.ndarray:
    if fmax is None:
        fmax = sample_rate / 2

    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center > left:
            filters[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters


def _power_spectrogram(audio: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> np.ndarray:
    padded = np.pad(audio, (n_fft // 2, n_fft // 2), mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length][:n_frames]
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=1)
    power = np.abs(spectrum) ** 2
    return power.T.astype(np.float32)


def extract_logmel(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE, n_mels: int = N_MELS) -> np.ndarray:
    power = _power_spectrogram(audio)
    mel_basis = _mel_filterbank(sample_rate=sample_rate, n_mels=n_mels)
    mel = np.maximum(mel_basis @ power, 1e-10)
    log_mel = 10.0 * np.log10(mel) - 10.0 * np.log10(np.max(mel))
    return log_mel.astype(np.float32)


def extract_mfcc(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    log_mel = extract_logmel(audio, sample_rate=sample_rate, n_mels=128)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]
    return mfcc.flatten().astype(np.float32)


def extract_mfcc_matrix(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    log_mel = extract_logmel(audio, sample_rate=sample_rate, n_mels=128)
    return dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc].astype(np.float32)


def _summary_stats(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.mean(matrix, axis=1),
            np.std(matrix, axis=1),
        ]
    ).astype(np.float32)


def _chroma_from_power(power: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / sample_rate)
    chroma = np.zeros((12, power.shape[1]), dtype=np.float32)
    valid = frequencies > 0
    midi = np.rint(69 + 12 * np.log2(frequencies[valid] / 440.0)).astype(int)
    pitch_classes = np.mod(midi, 12)

    for pitch_class in range(12):
        chroma[pitch_class] = power[valid][pitch_classes == pitch_class].sum(axis=0)

    frame_energy = np.maximum(chroma.sum(axis=0, keepdims=True), 1e-10)
    return chroma / frame_energy


def _spectral_shape_features(power: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / sample_rate).astype(np.float32)
    energy = np.maximum(power.sum(axis=0), 1e-10)
    centroid = (frequencies[:, None] * power).sum(axis=0) / energy
    bandwidth = np.sqrt((((frequencies[:, None] - centroid[None, :]) ** 2) * power).sum(axis=0) / energy)

    cumulative_energy = np.cumsum(power, axis=0)
    rolloff_threshold = 0.85 * energy
    rolloff_indices = np.argmax(cumulative_energy >= rolloff_threshold[None, :], axis=0)
    rolloff = frequencies[rolloff_indices]

    shape = np.vstack([centroid, bandwidth, rolloff])
    return np.concatenate([shape.mean(axis=1), shape.std(axis=1), np.median(shape, axis=1)]).astype(np.float32)


def _dominant_frequency_features(power: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / sample_rate).astype(np.float32)
    valid = (frequencies >= 50.0) & (frequencies <= 2000.0)
    valid_power = power[valid]
    valid_frequencies = frequencies[valid]
    peak_indices = np.argmax(valid_power, axis=0)
    dominant = valid_frequencies[peak_indices]
    confidence = valid_power[peak_indices, np.arange(valid_power.shape[1])] / np.maximum(valid_power.sum(axis=0), 1e-10)
    contour = np.vstack([dominant, confidence])
    return np.concatenate([contour.mean(axis=1), contour.std(axis=1), np.median(contour, axis=1)]).astype(np.float32)


def pad_or_crop_spectrogram(spec: np.ndarray, target_frames: int = CNN_TARGET_FRAMES) -> np.ndarray:
    n_mels, frames = spec.shape
    if frames == target_frames:
        return spec.astype(np.float32)
    if frames > target_frames:
        return spec[:, :target_frames].astype(np.float32)

    pad_width = target_frames - frames
    return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=float(spec.min())).astype(
        np.float32
    )


def extract_classical_features(
    filepath: str | Path,
    apply_aug: bool = False,
    rng: np.random.Generator | None = None,
    duration_seconds: float = TARGET_AUDIO_DURATION_SECONDS,
) -> np.ndarray:
    audio = preprocess_audio(filepath, apply_aug=apply_aug, rng=rng, duration_seconds=duration_seconds)
    return extract_mfcc(audio, TARGET_SAMPLE_RATE)


def extract_summary_features(
    filepath: str | Path,
    apply_aug: bool = False,
    rng: np.random.Generator | None = None,
    duration_seconds: float = TARGET_AUDIO_DURATION_SECONDS,
    feature_set: str = "combined",
) -> np.ndarray:
    """Extract compact statistics for improved classical ML models."""
    audio = preprocess_audio(filepath, apply_aug=apply_aug, rng=rng, duration_seconds=duration_seconds)
    power = _power_spectrogram(audio)
    mfcc = extract_mfcc_matrix(audio, TARGET_SAMPLE_RATE)
    mfcc_delta = np.gradient(mfcc, axis=1)
    log_mel = extract_logmel(audio, TARGET_SAMPLE_RATE, n_mels=N_MELS)
    chroma = _chroma_from_power(power, TARGET_SAMPLE_RATE)

    feature_groups = {
        "mfcc": [_summary_stats(mfcc), _summary_stats(mfcc_delta)],
        "mel": [_summary_stats(log_mel)],
        "pitch": [_summary_stats(chroma), _dominant_frequency_features(power, TARGET_SAMPLE_RATE)],
        "combined": [
            _summary_stats(mfcc),
            _summary_stats(mfcc_delta),
            _summary_stats(log_mel),
            _summary_stats(chroma),
            _dominant_frequency_features(power, TARGET_SAMPLE_RATE),
            _spectral_shape_features(power, TARGET_SAMPLE_RATE),
        ],
    }
    if feature_set not in feature_groups:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Expected one of {sorted(feature_groups)}.")

    return np.concatenate(feature_groups[feature_set]).astype(np.float32)


def extract_cnn_features(
    filepath: str | Path,
    apply_aug: bool = False,
    rng: np.random.Generator | None = None,
    duration_seconds: float = TARGET_AUDIO_DURATION_SECONDS,
) -> np.ndarray:
    audio = preprocess_audio(filepath, apply_aug=apply_aug, rng=rng, duration_seconds=duration_seconds)
    spec = extract_logmel(audio, TARGET_SAMPLE_RATE, n_mels=N_MELS)
    return pad_or_crop_spectrogram(spec, CNN_TARGET_FRAMES)[..., np.newaxis]
