"""
Analog resistive crossbar array (Compute-in-Memory substrate).

Each cell stores conductance G_ij (Siemens). Wordlines carry input voltages;
bitlines accumulate currents via Kirchhoff's current law. This is the physical
basis for instant matrix-vector multiply in neuromorphic inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ConductanceState(Enum):
    """Memristor programming states mapped to conductance ranges."""

    LRS = "low_resistance"   # ~1e-4 S  (weight ≈ +1)
    HRS = "high_resistance"  # ~1e-6 S  (weight ≈ 0)
    OPEN = "open_circuit"    # G = 0     (manufacturing defect)


# Physical conductance bounds (typical ReRAM / PCM crossbar literature)
G_LRS = 1.0e-4   # Siemens
G_HRS = 1.0e-6   # Siemens
G_MIN = 1.0e-7
G_MAX = 2.0e-4


@dataclass
class MemristorCrossbar:
    """
    M×N resistive crossbar array.

    Parameters
    ----------
    rows, cols : int
        Array dimensions (wordlines × bitlines).
    conductance : ndarray, optional
        Initial conductance matrix (S). If None, initialized to HRS.
    """

    rows: int
    cols: int
    conductance: np.ndarray = field(init=False)
    defect_mask: np.ndarray = field(init=False)  # True = open-circuit defect

    def __post_init__(self) -> None:
        self.conductance = np.full((self.rows, self.cols), G_HRS, dtype=np.float64)
        self.defect_mask = np.zeros((self.rows, self.cols), dtype=bool)

    def program_weight_matrix(self, weights: np.ndarray) -> None:
        """
        Map normalized neural weights [-1, 1] to differential conductance pairs.

        Positive weights use LRS on the + branch; negative weights use LRS on
        the − branch (differential signaling cancels common-mode noise).
        """
        assert weights.shape == (self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                if self.defect_mask[i, j]:
                    self.conductance[i, j] = 0.0
                    continue
                w = float(np.clip(weights[i, j], -1.0, 1.0))
                # Linear mapping: |w| → conductance between HRS and LRS
                g = G_HRS + abs(w) * (G_LRS - G_HRS)
                self.conductance[i, j] = g

    def apply_defects(self, defect_fraction: float, rng: Optional[np.random.Generator] = None) -> int:
        """
        Inject open-circuit (stuck-open) defects at random cells.

        Returns the number of defective cells.
        """
        rng = rng or np.random.default_rng()
        n_defects = int(round(defect_fraction * self.rows * self.cols))
        flat_indices = rng.choice(self.rows * self.cols, size=n_defects, replace=False)
        self.defect_mask.flat[flat_indices] = True
        for idx in flat_indices:
            r, c = divmod(idx, self.cols)
            self.conductance[r, c] = 0.0
        return n_defects

    def read_conductance(self) -> np.ndarray:
        """Return a copy of the conductance matrix."""
        return self.conductance.copy()

    def effective_conductance(self) -> np.ndarray:
        """Conductance after defect masking."""
        g = self.conductance.copy()
        g[self.defect_mask] = 0.0
        return g

    @property
    def defect_rate(self) -> float:
        return float(self.defect_mask.sum()) / (self.rows * self.cols)

    def summary(self) -> dict:
        return {
            "shape": (self.rows, self.cols),
            "total_cells": self.rows * self.cols,
            "defective_cells": int(self.defect_mask.sum()),
            "defect_rate_pct": round(self.defect_rate * 100, 3),
            "g_mean_S": float(self.conductance[self.conductance > 0].mean()) if (self.conductance > 0).any() else 0.0,
        }
