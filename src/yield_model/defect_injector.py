"""
Manufacturing yield model for resistive crossbar arrays.

Injects open-circuit (stuck-open) defects at statistically distributed
locations and evaluates classification accuracy degradation vs. defect rate.
This validates fault-tolerant design for edge medical inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..crossbar.memristor_array import MemristorCrossbar, G_HRS, G_LRS


@dataclass
class FaultToleranceReport:
    """Results of a yield Monte Carlo trial."""

    defect_rate: float
    n_defects: int
    accuracy_baseline: float
    accuracy_defective: float
    accuracy_drop_pct: float
    passes_threshold: bool
    threshold: float = 0.90


@dataclass
class YieldModel:
    """
    Monte Carlo yield simulator for crossbar defect injection.
    """

    crossbar: MemristorCrossbar
    accuracy_threshold: float = 0.90
    _ideal_conductance: Optional[np.ndarray] = field(default=None, init=False)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    def save_ideal_state(self) -> None:
        """Snapshot conductance before defect injection."""
        self._ideal_conductance = self.crossbar.conductance.copy()

    def restore_ideal_state(self) -> None:
        """Restore crossbar to pre-defect state."""
        if self._ideal_conductance is not None:
            self.crossbar.conductance[:] = self._ideal_conductance
        self.crossbar.defect_mask[:] = False

    def inject_defects(self, defect_fraction: float) -> int:
        """Reset and inject fresh defects at given fraction."""
        self.restore_ideal_state()
        return self.crossbar.apply_defects(defect_fraction, self.rng)

    def compute_mvm_error(
        self,
        weights: np.ndarray,
        inputs: np.ndarray,
        defect_fraction: float,
    ) -> float:
        """Relative MVM error caused by defects."""
        ideal = weights @ inputs.T

        self.inject_defects(defect_fraction)
        g_defect = self.crossbar.effective_conductance()

        scale = G_LRS - G_HRS
        w_defect = (g_defect - G_HRS) / scale
        w_defect = np.clip(w_defect, 0, 1) * np.sign(weights)
        defective = w_defect @ inputs.T

        rel_error = np.linalg.norm(ideal - defective, axis=0) / (np.linalg.norm(ideal, axis=0) + 1e-9)
        return float(rel_error.mean())

    def evaluate_accuracy_impact(
        self,
        predict_fn,
        test_inputs: np.ndarray,
        test_labels: np.ndarray,
        defect_fraction: float,
        baseline_accuracy: Optional[float] = None,
    ) -> FaultToleranceReport:
        """Measure classification accuracy before and after defect injection."""
        if baseline_accuracy is None:
            self.restore_ideal_state()
            baseline_preds = predict_fn(test_inputs)
            baseline_accuracy = float((baseline_preds == test_labels).mean())

        n_defects = self.inject_defects(defect_fraction)
        defective_preds = predict_fn(test_inputs)
        defective_accuracy = float((defective_preds == test_labels).mean())

        drop = (baseline_accuracy - defective_accuracy) / max(baseline_accuracy, 1e-9) * 100

        return FaultToleranceReport(
            defect_rate=defect_fraction,
            n_defects=n_defects,
            accuracy_baseline=baseline_accuracy,
            accuracy_defective=defective_accuracy,
            accuracy_drop_pct=drop,
            passes_threshold=defective_accuracy >= self.accuracy_threshold,
            threshold=self.accuracy_threshold,
        )

    def monte_carlo_study(
        self,
        predict_fn,
        test_inputs: np.ndarray,
        test_labels: np.ndarray,
        defect_rates: list[float],
        n_trials: int = 10,
    ) -> list[dict]:
        """Run Monte Carlo yield study across defect rates."""
        results = []
        for rate in defect_rates:
            accuracies = []
            passes = 0
            for trial in range(n_trials):
                self.rng = np.random.default_rng(42 + trial)
                report = self.evaluate_accuracy_impact(
                    predict_fn, test_inputs, test_labels, rate
                )
                accuracies.append(report.accuracy_defective)
                passes += int(report.passes_threshold)

            results.append({
                "defect_rate_pct": rate * 100,
                "mean_accuracy": float(np.mean(accuracies)),
                "std_accuracy": float(np.std(accuracies)),
                "pass_rate": passes / n_trials,
                "threshold": self.accuracy_threshold,
            })
        return results
