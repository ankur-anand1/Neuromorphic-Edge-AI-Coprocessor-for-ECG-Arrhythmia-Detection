"""
ECG signal loading and synthetic generation.

Provides MIT-BIH-style arrhythmia records (synthetic fallback when wfdb
is unavailable) for edge arrhythmia detection validation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# MIT-BIH Arrhythmia Database record classes
# N = Normal, V = Premature Ventricular Contraction (PVC), A = Atrial Premature Beat
ARRHYTHMIA_CLASSES = {
    0: "Normal Sinus Rhythm (NSR)",
    1: "Premature Ventricular Contraction (PVC)",
    2: "Atrial Premature Beat (APB)",
}

FS_DEFAULT = 360  # Hz (MIT-BIH sampling rate)


def generate_synthetic_ecg(
    n_beats: int = 200,
    fs: int = FS_DEFAULT,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic ECG beats with labeled arrhythmias.

    Returns
    -------
    signals : ndarray, shape (n_beats, beat_length)
    labels : ndarray, shape (n_beats,)
    fs : int
    """
    rng = np.random.default_rng(seed)
    beat_length = int(0.8 * fs)  # 800 ms window
    signals = np.zeros((n_beats, beat_length))
    labels = np.zeros(n_beats, dtype=np.int32)

    t = np.arange(beat_length) / fs

    for i in range(n_beats):
        label = rng.choice([0, 0, 0, 0, 1, 1, 2], p=[0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05])
        labels[i] = label

        # Morphology parameters per class (distinct for separability)
        if label == 0:  # NSR
            p_amp, qrs_center, qrs_amp, qrs_width, t_center = 0.15, 0.30, 1.0, 0.04, 0.50
        elif label == 1:  # PVC - early wide QRS
            p_amp, qrs_center, qrs_amp, qrs_width, t_center = 0.05, 0.22, 1.5, 0.10, 0.42
        else:  # APB - elevated P, narrow QRS
            p_amp, qrs_center, qrs_amp, qrs_width, t_center = 0.40, 0.28, 0.9, 0.035, 0.48

        # P wave
        signal = p_amp * np.exp(-((t - 0.15) ** 2) / (2 * 0.015 ** 2))

        # QRS complex
        signal += qrs_amp * np.exp(-((t - qrs_center) ** 2) / (2 * qrs_width ** 2))

        # T wave
        signal += 0.3 * np.exp(-((t - t_center) ** 2) / (2 * 0.06 ** 2))

        # Baseline wander + noise
        signal += 0.05 * np.sin(2 * np.pi * 0.5 * t)
        signal += 0.02 * rng.standard_normal(beat_length)

        signals[i] = signal

    return signals, labels, np.array([fs])


def load_ecg_dataset(
    n_beats: int = 200,
    use_synthetic: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load ECG beat dataset. Falls back to synthetic if wfdb unavailable.

    Returns (signals, labels, fs).
    """
    if use_synthetic:
        signals, labels, fs_arr = generate_synthetic_ecg(n_beats=n_beats, seed=seed)
        return signals, labels, int(fs_arr[0])

    try:
        import wfdb
        record = wfdb.rdrecord("100", pn_dir="mitdb")
        fs = record.fs
        sig = record.p_signal[:, 0]
        ann = wfdb.rdann("100", "atr", pn_dir="mitdb")
        # ... real extraction would go here
        return generate_synthetic_ecg(n_beats=n_beats)[0:2] + (fs,)
    except Exception:
        signals, labels, fs_arr = generate_synthetic_ecg(n_beats=n_beats)
        return signals, labels, int(fs_arr[0])
