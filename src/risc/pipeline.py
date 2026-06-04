"""
5-stage cycle-accurate pipeline: IF → ID → EX → MEM → WB

Models pipeline hazards, stalls on coprocessor operations, and event-driven
wake-up from WFI (Wait-For-Interrupt) for edge power gating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .isa import Instruction, Opcode


class PipelineStage(Enum):
    IF = auto()
    ID = auto()
    EX = auto()
    MEM = auto()
    WB = auto()


@dataclass
class PipelineRegister:
    """Inter-stage pipeline register."""

    valid: bool = False
    instruction: Optional[Instruction] = None
    pc: int = 0
    alu_result: int = 0
    mem_addr: int = 0
    mem_data: int = 0
    rd: int = 0
    write_back: bool = False


@dataclass
class CycleAccuratePipeline:
    """
    5-stage in-order pipeline with coprocessor stall logic.

    Coprocessor instructions (CIM_START, ADC_START) stall the pipeline
    until the mixed-signal unit raises its DONE flag (event-driven).
    """

    if_id: PipelineRegister = field(default_factory=PipelineRegister)
    id_ex: PipelineRegister = field(default_factory=PipelineRegister)
    ex_mem: PipelineRegister = field(default_factory=PipelineRegister)
    mem_wb: PipelineRegister = field(default_factory=PipelineRegister)

    total_cycles: int = field(default=0, init=False)
    instructions_retired: int = field(default=0, init=False)
    stall_cycles: int = field(default=0, init=False)
    cim_stall_cycles: int = field(default=0, init=False)
    adc_stall_cycles: int = field(default=0, init=False)

    stalled: bool = field(default=False, init=False)
    stall_reason: str = field(default="", init=False)
    waiting_for_event: bool = field(default=False, init=False)

    def tick(self) -> None:
        """Advance pipeline by one clock cycle."""
        self.total_cycles += 1
        if self.stalled:
            self.stall_cycles += 1

    def stall_pipeline(self, reason: str) -> None:
        self.stalled = True
        self.stall_reason = reason

    def release_stall(self) -> None:
        self.stalled = False
        self.stall_reason = ""

    def is_coprocessor_op(self, instr: Instruction) -> bool:
        return instr.opcode in (Opcode.CIM_START, Opcode.ADC_START)

    def retire(self) -> None:
        self.instructions_retired += 1

    def stats(self) -> dict:
        return {
            "total_cycles": self.total_cycles,
            "instructions_retired": self.instructions_retired,
            "cpi": round(self.total_cycles / max(1, self.instructions_retired), 2),
            "stall_cycles": self.stall_cycles,
            "cim_stall_cycles": self.cim_stall_cycles,
            "adc_stall_cycles": self.adc_stall_cycles,
            "ipc": round(self.instructions_retired / max(1, self.total_cycles), 3),
        }
