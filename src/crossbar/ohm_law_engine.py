"""
Ohm's Law matrix multiplication engine for the resistive crossbar.

Analog MVM:  I_out = G_eff @ V_in

Each cycle represents one wordline pulse width (T_pulse). Bitline currents
are integrated on capacitors and read out through the SAR ADC front-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .memristor_array import MemristorCrossbar


@dataclass
class OhmLawEngine:
    """
    Cycle-accurate analog compute engine wrapping the memristor crossbar.

    Attributes
    ----------
    crossbar : MemristorCrossbar
    v_read : float
        Wordline read voltage (V).
    r_line : float
        Access transistor + metal line resistance (Ω), adds IR drop.
    c_bitline : float
        Bitline integration capacitance (F).
    pulse_width_s : float
        Wordline pulse duration per MAC cycle (s).
    """

    crossbar: MemristorCrossbar
    v_read: float = 0.3          # 300 mV read voltage (low-power edge)
    r_line: float = 500.0        # 500 Ω line resistance
    c_bitline: float = 50.0e-15  # 50 fF bitline cap
    pulse_width_s: float = 10.0e-9  # 10 ns pulse

    cycle_count: int = field(default=0, init=False)
    last_currents: Optional[np.ndarray] = field(default=None, init=False)
    last_voltages: Optional[np.ndarray] = field(default=None, init=False)

    def mac_cycle(self, v_in: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Execute one analog MAC cycle: apply voltages, compute bitline currents.

        Parameters
        ----------
        v_in : ndarray, shape (rows,)
            Wordline input voltages (V).

        Returns
        -------
        currents : ndarray, shape (cols,)
            Bitline currents after IR-drop correction (A).
        cycles : int
            Cycles consumed (always 1 for single-pulse MVM).
        """
        assert v_in.shape == (self.crossbar.rows,)
        self.cycle_count += 1

        g_eff = self.crossbar.effective_conductance()

        # IR drop on wordlines: effective voltage at each cell
        v_cell = v_in[:, np.newaxis] * self.v_read / (1.0 + self.r_line * g_eff.sum(axis=1, keepdims=True))

        # Ohm's Law: I_j = Σ_i G_ij * V_i  (KCL at each bitline)
        currents = (g_eff * v_cell).sum(axis=0)

        # Parasitic leakage (modeled as additive noise floor)
        leakage = 1.0e-12 * np.random.randn(self.crossbar.cols)
        currents = currents + leakage

        self.last_currents = currents.copy()
        self.last_voltages = v_in.copy()
        return currents, 1

    def integrate_and_read(self, currents: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Integrate bitline currents on capacitors → voltage for ADC sampling.

        V_bl = (I * T_pulse) / C_bitline
        """
        v_bl = (currents * self.pulse_width_s) / self.c_bitline
        return v_bl, 1

    def full_mvm(self, v_in: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Complete analog MVM: MAC + integration.

        Returns bitline voltages and total cycles consumed.
        """
        currents, c1 = self.mac_cycle(v_in)
        voltages, c2 = self.integrate_and_read(currents)
        return voltages, c1 + c2

    def reset_cycles(self) -> None:
        self.cycle_count = 0
