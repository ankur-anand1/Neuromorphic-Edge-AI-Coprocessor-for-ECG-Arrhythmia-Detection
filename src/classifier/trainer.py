"""
Train crossbar-mapped weights from ECG feature data.

Uses ridge regression to fit a two-layer network, then programs
conductances into the memristor crossbar for analog inference.
"""

from __future__ import annotations

import numpy as np

from ..classifier.arrhythmia_detector import CrossbarClassifier, INPUT_FEATURES, HIDDEN_SIZE, NUM_CLASSES
from ..crossbar.memristor_array import G_HRS, G_LRS


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(labels), n_classes))
    y[np.arange(len(labels)), labels] = 1.0
    return y


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def conductance_to_weights(conductance: np.ndarray, sign_matrix: np.ndarray) -> np.ndarray:
    """Map conductance matrix back to normalized weight matrix."""
    scale = G_LRS - G_HRS
    w = (conductance - G_HRS) / scale
    w = np.clip(w, 0.0, 1.0) * sign_matrix
    return w


def train_classifier(
    classifier: CrossbarClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    reg: float = 0.001,
    epochs: int = 800,
    lr: float = 0.15,
) -> float:
    """
    Train hidden + output layers, program crossbar, return training accuracy.
    """
    n = len(features)
    rng = np.random.default_rng(0)

    W1 = rng.standard_normal((INPUT_FEATURES, HIDDEN_SIZE)) * 0.5
    sign_matrix = np.sign(W1)
    sign_matrix[sign_matrix == 0] = 1.0
    b1 = np.zeros(HIDDEN_SIZE)
    W2 = rng.standard_normal((HIDDEN_SIZE, NUM_CLASSES)) * 0.5
    b2 = np.zeros(NUM_CLASSES)

    Y = _one_hot(labels, NUM_CLASSES)
    # Class weights for imbalanced ECG labels
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    class_weights = n / (NUM_CLASSES * class_counts + 1e-9)
    sample_weights = class_weights[labels]

    for _ in range(epochs):
        z1 = features @ W1 + b1
        h1 = _relu(z1)
        logits = h1 @ W2 + b2
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        d_logits = (probs - Y) * sample_weights[:, np.newaxis] / n
        dW2 = h1.T @ d_logits + reg * W2
        db2 = d_logits.sum(axis=0)
        dh1 = d_logits @ W2.T
        dh1[z1 <= 0] = 0
        dW1 = features.T @ dh1 + reg * W1
        db1 = dh1.sum(axis=0)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    w_abs = np.abs(W1)
    w_max = w_abs.max() + 1e-9
    w_normalized = W1 / w_max

    classifier.crossbar.program_weight_matrix(np.abs(w_normalized))
    classifier.sign_matrix = sign_matrix
    classifier.output_weights = W2.copy()
    classifier.output_bias = b2.copy()
    classifier._w_max = w_max
    classifier._b1 = b1.copy()
    classifier._W1_ideal = W1.copy()

    preds = classifier.predict(features)
    return float((preds == labels).mean())
