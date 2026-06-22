"""
Arrhythmia classifier mapped to resistive crossbar weights.

A 16×8 crossbar implements the first hidden layer; digital head classifies
into 3 arrhythmia classes (NSR, PVC, APB).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..adc.sar_adc import SARADC
from ..crossbar.memristor_array import MemristorCrossbar, G_HRS, G_LRS
from ..crossbar.ohm_law_engine import OhmLawEngine
from ..signal.preprocessing import extract_features


NUM_CLASSES = 3
INPUT_FEATURES = 16
HIDDEN_SIZE = 8


@dataclass
class CrossbarClassifier:
    """
    Single-layer crossbar neural network for arrhythmia classification.

    Architecture: 16 inputs → 8 hidden (crossbar MVM) → 3 outputs (digital)
    """

    crossbar: MemristorCrossbar
    ohm_engine: OhmLawEngine
    sar_adc: SARADC
    output_weights: np.ndarray = field(init=False)
    output_bias: np.ndarray = field(init=False)
    sign_matrix: np.ndarray = field(init=False)
    _w_max: float = field(default=1.0, init=False)
    _b1: np.ndarray = field(init=False)
    _W1_ideal: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.output_weights = np.zeros((HIDDEN_SIZE, NUM_CLASSES))
        self.output_bias = np.zeros(NUM_CLASSES)
        self.sign_matrix = np.ones((INPUT_FEATURES, HIDDEN_SIZE))
        self._b1 = np.zeros(HIDDEN_SIZE)
        self._W1_ideal = np.zeros((INPUT_FEATURES, HIDDEN_SIZE))

    def _effective_weights(self) -> np.ndarray:
        """Map defective conductance back to weight matrix."""
        scale = G_LRS - G_HRS
        g = self.crossbar.effective_conductance()
        w = (g - G_HRS) / scale
        w = np.clip(w, 0.0, 1.0) * self.sign_matrix
        return w * self._w_max

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def forward_digital(self, features: np.ndarray) -> np.ndarray:
        """Inference using effective crossbar weights (includes defect modeling)."""
        single = features.ndim == 1
        if single:
            features = features[np.newaxis, :]

        W_eff = self._effective_weights()
        hidden = self._relu(features @ W_eff + self._b1)
        logits = hidden @ self.output_weights + self.output_bias

        return logits[0] if single else logits

    def forward_analog(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference through full analog crossbar + SAR ADC path.

        Used for cycle-accurate single-beat demo.
        """
        single = features.ndim == 1
        if single:
            features = features[np.newaxis, :]

        batch_size = features.shape[0]
        all_logits = np.zeros((batch_size, NUM_CLASSES))

        for b in range(batch_size):
            v_in = features[b]
            v_norm = v_in / (np.linalg.norm(v_in) + 1e-9)
            v_scaled = v_norm * 0.25  # Map to wordline voltage range

            bitline_voltages, _ = self.ohm_engine.full_mvm(v_scaled)

            # Normalize to ADC full scale
            v_max = np.abs(bitline_voltages).max() + 1e-12
            v_adc_in = (bitline_voltages / v_max) * self.sar_adc.v_ref * 0.9 + self.sar_adc.v_ref * 0.05

            codes, _ = self.sar_adc.convert_batch(v_adc_in)
            hidden_adc = self.sar_adc.dequantize(codes)

            # Calibrate ADC output back to hidden layer scale
            hidden = self._relu((hidden_adc / self.sar_adc.v_ref) * v_max * 100 + self._b1)
            logits = hidden @ self.output_weights + self.output_bias
            all_logits[b] = logits

        return all_logits[0] if single else all_logits

    def predict(self, features: np.ndarray, analog: bool = False) -> np.ndarray:
        forward = self.forward_analog if analog else self.forward_digital
        logits = forward(features)
        if logits.ndim == 1:
            return int(np.argmax(logits))
        return np.argmax(logits, axis=1)

    def predict_proba(self, features: np.ndarray, analog: bool = False) -> np.ndarray:
        forward = self.forward_analog if analog else self.forward_digital
        logits = forward(features)
        if logits.ndim == 1:
            return self._softmax(logits)
        return np.array([self._softmax(l) for l in logits])

    def accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        preds = self.predict(features)
        return float((preds == labels).mean())


@dataclass
class ArrhythmiaDetector:
    """
    End-to-end arrhythmia detection pipeline for the neuromorphic coprocessor.
    """

    classifier: CrossbarClassifier
    fs: int = 360

    def detect_from_beats(self, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = extract_features(beats, n_features=INPUT_FEATURES)
        preds = self.classifier.predict(features)
        probas = np.array([self.classifier.predict_proba(f) for f in features])
        return preds, probas

    def detect(self, beats: np.ndarray, labels: Optional[np.ndarray] = None, analog: bool = False) -> dict:
        features = extract_features(beats, n_features=INPUT_FEATURES)
        preds = self.classifier.predict(features, analog=analog)
        probas = np.array([self.classifier.predict_proba(f, analog=analog) for f in features])

        result = {
            "predictions": preds,
            "probabilities": probas,
            "n_beats": len(beats),
        }
        if labels is not None:
            result["accuracy"] = float((preds == labels).mean())
            result["labels"] = labels
        return result
