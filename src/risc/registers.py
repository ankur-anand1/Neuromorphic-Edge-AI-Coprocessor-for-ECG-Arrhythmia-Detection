"""
16-bit RISC register file and special-purpose coprocessor registers.

General-purpose: R0–R7 (R0 hardwired to 0)
Special:         PC, SP, STATUS, CIM_CTRL, ADC_CTRL, EVENT_MASK
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


NUM_GPR = 8
R0_INDEX = 0  # Always zero


@dataclass
class RegisterFile:
    """16-bit register file for the edge neuromorphic coprocessor."""

    gpr: list[int] = field(default_factory=lambda: [0] * NUM_GPR)
    pc: int = 0x0000
    sp: int = 0xFF00
    status: int = 0x0000

    # Coprocessor control registers (memory-mapped)
    cim_ctrl: int = 0x0000    # bit0=START, bit1=DONE, bit2=DEFECT_EN
    adc_ctrl: int = 0x0000    # bit0=START, bit1=DONE
    event_mask: int = 0x0000  # Interrupt enable bits
    event_pending: int = 0x0000

    # Memory-mapped I/O results
    cim_result_addr: int = 0x8000
    adc_result_addr: int = 0x8100

    def read(self, reg: int) -> int:
        if reg == R0_INDEX:
            return 0
        if 0 <= reg < NUM_GPR:
            return self.gpr[reg] & 0xFFFF
        raise ValueError(f"Invalid GPR index: {reg}")

    def write(self, reg: int, value: int) -> None:
        if reg == R0_INDEX:
            return  # R0 is read-only
        if 0 <= reg < NUM_GPR:
            self.gpr[reg] = value & 0xFFFF

    def set_flag(self, bit: int) -> None:
        self.status |= (1 << bit)

    def clear_flag(self, bit: int) -> None:
        self.status &= ~(1 << bit)

    def flag_set(self, bit: int) -> bool:
        return bool(self.status & (1 << bit))

    # Status flag bits
    FLAG_ZERO = 0
    FLAG_CARRY = 1
    FLAG_CIM_DONE = 2
    FLAG_ADC_DONE = 3
    FLAG_ARRHYTHMIA = 4

    def snapshot(self) -> dict:
        return {
            "pc": f"0x{self.pc:04X}",
            "sp": f"0x{self.sp:04X}",
            "gpr": [f"0x{v:04X}" for v in self.gpr],
            "status": f"0x{self.status:04X}",
            "cim_ctrl": f"0x{self.cim_ctrl:04X}",
        }
