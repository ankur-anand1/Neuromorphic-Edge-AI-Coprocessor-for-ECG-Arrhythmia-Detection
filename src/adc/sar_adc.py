"""
Successive Approximation Register (SAR) ADC model.

Models the mixed-signal boundary: continuous bitline voltages are sampled
into discrete digital codes. Includes INL/DNL, comparator offset, and
conversion latency in cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class QuantizationResult:
    """Output of one SAR ADC conversion."""

    digital_code: int
    analog_reconstructed: float
    quantization_error: float
    cycles: int


@dataclass
class SARADC:
    """
    N-bit SAR ADC with configurable reference and conversion cycles.

    Parameters
    ----------
    resolution_bits : int
        ADC resolution (default 10-bit for crossbar readout).
    v_ref : float
        Full-scale reference voltage (V).
    conversion_cycles : int
        Clock cycles per conversion (typical SAR: N+2 cycles).
    comparator_offset_v : float
        Static comparator offset (V).
    inl_lsb : float
        Integral non-linearity (LSB peak).
    """

    resolution_bits: int = 10
    v_ref: float = 1.0
    conversion_cycles: int = 12
    comparator_offset_v: float = 0.0
    inl_lsb: float = 0.5

    total_conversions: int = field(default=0, init=False)
    _ladder: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n_levels = 2 ** self.resolution_bits
        self._ladder = np.linspace(0, self.v_ref, n_levels)

    @property
    def lsb(self) -> float:
        return self.v_ref / (2 ** self.resolution_bits)

    @property
    def max_code(self) -> int:
        return (2 ** self.resolution_bits) - 1

    def convert(self, v_analog: float) -> QuantizationResult:
        """
        Perform one SAR conversion with INL distortion and comparator offset.
        """
        self.total_conversions += 1
        v = v_analog + self.comparator_offset_v
        v = np.clip(v, 0.0, self.v_ref - self.lsb)

        # Ideal code
        code = int(round(v / self.lsb))
        code = np.clip(code, 0, self.max_code)

        # INL: sinusoidal distortion across input range
        inl_error = self.inl_lsb * self.lsb * np.sin(2 * np.pi * code / self.max_code)
        v_recon = self._ladder[code] + inl_error
        q_error = v_analog - v_recon

        return QuantizationResult(
            digital_code=code,
            analog_reconstructed=v_recon,
            quantization_error=q_error,
            cycles=self.conversion_cycles,
        )

    def convert_batch(self, voltages: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Convert a vector of analog voltages.

        Returns digital codes and total cycles (channels converted sequentially).
        """
        codes = np.zeros(len(voltages), dtype=np.int32)
        total_cycles = 0
        for i, v in enumerate(voltages):
            result = self.convert(float(v))
            codes[i] = result.digital_code
            total_cycles += result.cycles
        return codes, total_cycles

    def dequantize(self, codes: np.ndarray) -> np.ndarray:
        """Map digital codes back to analog voltage estimates."""
        clipped = np.clip(codes, 0, self.max_code)
        return self._ladder[clipped]

    def snr_db(self, signal_rms: float, noise_rms: Optional[float] = None) -> float:
        """Theoretical SNR from quantization (6.02N + 1.76 dB) or measured."""
        if noise_rms is None:
            return 6.02 * self.resolution_bits + 1.76
        if noise_rms == 0:
            return float("inf")
        return 20 * np.log10(signal_rms / noise_rms)
