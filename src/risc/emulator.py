"""
Cycle-accurate 16-bit RISC emulator with event-driven coprocessor integration.

The emulator orchestrates instruction fetch-decode-execute while delegating
analog compute (CiM MVM) and mixed-signal conversion (SAR ADC) to hardware
models that consume real cycle budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..adc.sar_adc import SARADC
from ..crossbar.ohm_law_engine import OhmLawEngine
from .isa import Instruction, Opcode, decode_instruction, encode_instruction
from .pipeline import CycleAccuratePipeline
from .registers import RegisterFile


MEMORY_SIZE = 0x10000  # 64 KB address space


@dataclass
class RISCEmulator:
    """
    Full 16-bit RISC coprocessor emulator.

    Parameters
    ----------
    ohm_engine : OhmLawEngine
        Analog crossbar compute engine.
    sar_adc : SARADC
        Mixed-signal ADC front-end.
    program : list[int], optional
        Initial instruction memory (16-bit words).
    """

    ohm_engine: OhmLawEngine
    sar_adc: SARADC
    program: list[int] = field(default_factory=list)
    memory: np.ndarray = field(init=False)
    regs: RegisterFile = field(default_factory=RegisterFile)
    pipeline: CycleAccuratePipeline = field(default_factory=CycleAccuratePipeline)

    halted: bool = field(default=False, init=False)
    trace: list[str] = field(default_factory=list, init=False)
    enable_trace: bool = False

    # Callbacks for event-driven wake-up
    on_cim_complete: Optional[Callable] = field(default=None, init=False)
    on_adc_complete: Optional[Callable] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.memory = np.zeros(MEMORY_SIZE, dtype=np.uint16)
        self._load_program()

    def _load_program(self) -> None:
        for i, word in enumerate(self.program):
            if i < MEMORY_SIZE:
                self.memory[i] = word & 0xFFFF
        # Pad remainder with HALT to prevent runaway execution
        halt_word = encode_instruction(Opcode.HALT)
        for i in range(len(self.program), min(len(self.program) + 4, MEMORY_SIZE)):
            self.memory[i] = halt_word

    def load_program_at(self, address: int, instructions: list[int]) -> None:
        for i, word in enumerate(instructions):
            self.memory[address + i] = word & 0xFFFF

    def fetch(self) -> tuple[Instruction, int]:
        pc = self.regs.pc
        word = int(self.memory[pc])
        instr = decode_instruction(word)
        return instr, pc

    def execute_alu(self, instr: Instruction) -> int:
        rs1 = self.regs.read(instr.rs1)
        rs2 = self.regs.read(instr.rs2)
        imm = instr.imm
        op = instr.opcode
        result = 0

        if op == Opcode.ADD:
            result = (rs1 + rs2) & 0xFFFF
        elif op == Opcode.SUB:
            result = (rs1 - rs2) & 0xFFFF
        elif op == Opcode.AND:
            result = rs1 & rs2
        elif op == Opcode.OR:
            result = rs1 | rs2
        elif op == Opcode.XOR:
            result = rs1 ^ rs2
        elif op == Opcode.SHL:
            result = (rs1 << (rs2 & 0xF)) & 0xFFFF
        elif op == Opcode.SHR:
            result = (rs1 >> (rs2 & 0xF)) & 0xFFFF
        elif op == Opcode.CMP:
            diff = rs1 - rs2
            if diff == 0:
                self.regs.set_flag(RegisterFile.FLAG_ZERO)
            else:
                self.regs.clear_flag(RegisterFile.FLAG_ZERO)
            result = diff & 0xFFFF
        elif op == Opcode.LDI:
            result = imm & 0xFF

        return result

    def execute_cim_start(self) -> int:
        """Trigger analog MVM using input vector stored in R1 (scaled) and weight rows from memory."""
        rows = self.ohm_engine.crossbar.rows
        v_in = np.zeros(rows, dtype=np.float64)
        for i in range(rows):
            raw = self.regs.read(1) if i == 0 else int(self.memory[self.regs.cim_result_addr + i])
            v_in[i] = (raw / 32768.0)  # Normalize 16-bit fixed-point to voltage scale

        voltages, cycles = self.ohm_engine.full_mvm(v_in)

        # Store bitline voltages in memory-mapped region
        for j, v in enumerate(voltages):
            addr = self.regs.cim_result_addr + j
            code = int(np.clip(v * 1000, 0, 0xFFFF))  # mV → raw
            self.memory[addr] = code

        self.regs.set_flag(RegisterFile.FLAG_CIM_DONE)
        self.regs.event_pending |= 0x01
        if self.on_cim_complete:
            self.on_cim_complete(voltages, cycles)
        return cycles

    def execute_adc_start(self) -> int:
        """Convert bitline voltages through SAR ADC."""
        cols = self.ohm_engine.crossbar.cols
        voltages = np.zeros(cols, dtype=np.float64)
        for j in range(cols):
            raw = int(self.memory[self.regs.cim_result_addr + j])
            voltages[j] = raw / 1000.0  # mV → V

        codes, cycles = self.sar_adc.convert_batch(voltages)

        for j, code in enumerate(codes):
            self.memory[self.regs.adc_result_addr + j] = int(code)

        self.regs.set_flag(RegisterFile.FLAG_ADC_DONE)
        self.regs.event_pending |= 0x02
        if self.on_adc_complete:
            self.on_adc_complete(codes, cycles)
        return cycles

    def step(self) -> bool:
        """
        Execute one instruction (simplified single-cycle for non-coprocessor ops).

        Returns False if halted.
        """
        if self.halted:
            return False

        self.pipeline.tick()

        if self.pipeline.waiting_for_event:
            if self.regs.event_pending & self.regs.event_mask:
                self.pipeline.waiting_for_event = False
                self.regs.event_pending = 0
                self.pipeline.release_stall()
            else:
                return True  # Still sleeping

        instr, pc = self.fetch()

        if self.enable_trace:
            self.trace.append(f"[{self.pipeline.total_cycles:5d}] PC=0x{pc:04X}  {instr.mnemonic}")

        op = instr.opcode

        if op == Opcode.HALT:
            self.halted = True
            return False

        if op == Opcode.WFI:
            self.pipeline.waiting_for_event = True
            self.regs.event_mask = 0x03  # CiM + ADC events
            return True

        if op == Opcode.NOP:
            self.regs.pc = (pc + 1) & 0xFFFF
            self.pipeline.retire()
            return True

        if op == Opcode.EVT_CLR:
            self.regs.event_pending = 0
            self.regs.clear_flag(RegisterFile.FLAG_CIM_DONE)
            self.regs.clear_flag(RegisterFile.FLAG_ADC_DONE)
            self.regs.pc = (pc + 1) & 0xFFFF
            self.pipeline.retire()
            return True

        if op == Opcode.CIM_START:
            cim_cycles = self.execute_cim_start()
            for _ in range(cim_cycles - 1):
                self.pipeline.tick()
                self.pipeline.cim_stall_cycles += 1
            self.regs.pc = (pc + 1) & 0xFFFF
            self.pipeline.retire()
            return True

        if op == Opcode.ADC_START:
            adc_cycles = self.execute_adc_start()
            for _ in range(adc_cycles - 1):
                self.pipeline.tick()
                self.pipeline.adc_stall_cycles += 1
            self.regs.pc = (pc + 1) & 0xFFFF
            self.pipeline.retire()
            return True

        if op == Opcode.JMP:
            self.regs.pc = instr.imm & 0xFFFF
            self.pipeline.retire()
            return True

        if op == Opcode.BEQ:
            if self.regs.flag_set(RegisterFile.FLAG_ZERO):
                self.regs.pc = instr.imm & 0xFFFF
            else:
                self.regs.pc = (pc + 1) & 0xFFFF
            self.pipeline.retire()
            return True

        if op == Opcode.BNE:
            if not self.regs.flag_set(RegisterFile.FLAG_ZERO):
                self.regs.pc = instr.imm & 0xFFFF
            else:
                self.regs.pc = (pc + 1) & 0xFFFF
            self.pipeline.retire()
            return True

        if op == Opcode.LD:
            addr = (self.regs.read(instr.rs1) + instr.imm) & 0xFFFF
            value = int(self.memory[addr])
            self.regs.write(instr.rd, value)
            self.regs.pc = (pc + 1) & 0xFFFF
            self.pipeline.retire()
            return True

        if op == Opcode.ST:
            addr = (self.regs.read(instr.rs1) + instr.imm) & 0xFFFF
            self.memory[addr] = self.regs.read(instr.rd) & 0xFFFF
            self.regs.pc = (pc + 1) & 0xFFFF
            self.pipeline.retire()
            return True

        # ALU ops
        result = self.execute_alu(instr)
        self.regs.write(instr.rd, result)
        self.regs.pc = (pc + 1) & 0xFFFF
        self.pipeline.retire()
        return True

    def run(self, max_cycles: int = 10000) -> dict:
        """Run until HALT or max_cycles."""
        while not self.halted and self.pipeline.total_cycles < max_cycles:
            self.step()
        return self.pipeline.stats()
