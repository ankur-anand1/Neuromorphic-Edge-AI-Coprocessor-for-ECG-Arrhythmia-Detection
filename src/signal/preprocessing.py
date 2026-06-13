"""
ECG preprocessing: bandpass filtering, R-peak detection, beat segmentation.

Implements standard ELL205 signal-chain operations for real-time edge inference.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Butterworth bandpass filter (0.5–40 Hz typical for ECG)."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    return np.array([filtfilt(b, a, s) for s in signal])


def detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """Pan-Tompkins-style R-peak detection (simplified)."""
    if signal.ndim > 1:
        return np.array([detect_r_peaks(s, fs) for s in signal], dtype=object)

    filtered = bandpass_filter(signal, fs)
    diff = np.diff(filtered, prepend=filtered[0])
    squared = diff ** 2

    min_distance = int(0.2 * fs)  # 200 ms refractory period
    peaks, _ = find_peaks(squared, distance=min_distance, height=np.mean(squared))
    return peaks


def extract_beats(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
    window_before: float = 0.25,
    window_after: float = 0.45,
) -> np.ndarray:
    """Extract fixed-length beat windows centered on R-peaks."""
    before = int(window_before * fs)
    after = int(window_after * fs)
    beat_len = before + after
    beats = []

    for peak in r_peaks:
        start = peak - before
        end = peak + after
        if start >= 0 and end <= len(signal):
            beats.append(signal[start:end])

    if not beats:
        return np.array([])

    # Pad/truncate to uniform length
    max_len = max(len(b) for b in beats)
    uniform = np.zeros((len(beats), max_len))
    for i, b in enumerate(beats):
        uniform[i, :len(b)] = b
    return uniform


def extract_features(beats: np.ndarray, n_features: int = 16) -> np.ndarray:
    """
    Extract fixed feature vector from each beat via DCT compression.

    Maps continuous ECG morphology to a low-dimensional vector suitable
    for crossbar inference (16 features → 16 wordlines).
    """
    from scipy.fft import dct

    features = np.zeros((len(beats), n_features))
    for i, beat in enumerate(beats):
        normalized = (beat - beat.mean()) / (beat.std() + 1e-9)
        coeffs = dct(normalized, norm="ortho")[:n_features]
        features[i] = coeffs
    return features
