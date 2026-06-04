"""
16-bit RISC instruction set architecture (ISA).

Custom ISA for neuromorphic edge coprocessor with CiM/ADC coprocessor instructions.
All instructions are 16 bits (halfword aligned).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Opcode(IntEnum):
    # Arithmetic / Logic
    ADD = 0x0
    SUB = 0x1
    AND = 0x2
    OR = 0x3
    XOR = 0x4
    SHL = 0x5
    SHR = 0x6
    CMP = 0x7

    # Memory
    LD = 0x8
    ST = 0x9
    LDI = 0xA   # Load immediate (8-bit)

    # Control flow
    JMP = 0xB
    BEQ = 0xC   # Branch if zero flag
    BNE = 0xD

    # Coprocessor (mixed-signal boundary)
    CIM_START = 0xE   # Trigger crossbar MVM
    ADC_START = 0xF   # Trigger SAR ADC conversion

    # System
    NOP = 0x10
    HALT = 0x11
    WFI = 0x12        # Wait-for-interrupt (event-driven)
    EVT_CLR = 0x13    # Clear event pending


@dataclass(frozen=True)
class Instruction:
    """Decoded 16-bit instruction."""

    opcode: Opcode
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0
    raw: int = 0

    @property
    def mnemonic(self) -> str:
        op = self.opcode
        if op in (Opcode.LDI, Opcode.JMP, Opcode.BEQ, Opcode.BNE):
            return f"{op.name} R{self.rd}, #{self.imm & 0xFF}"
        if op in (Opcode.CIM_START, Opcode.ADC_START, Opcode.WFI, Opcode.HALT, Opcode.NOP, Opcode.EVT_CLR):
            return op.name
        return f"{op.name} R{self.rd}, R{self.rs1}, R{self.rs2}"


def decode_instruction(word: int) -> Instruction:
    """Decode a 16-bit instruction word."""
    word &= 0xFFFF
    opcode_val = (word >> 12) & 0xF
    rd = (word >> 9) & 0x7
    rs1 = (word >> 6) & 0x7
    rs2 = (word >> 3) & 0x7
    imm = word & 0xFF

    # Extended opcodes (upper nibble = 0x1X)
    if opcode_val == 0xE and (word >> 8) == 0x1E:
        ext = (word >> 4) & 0xF
        try:
            opcode = Opcode(ext + 0x10)
        except ValueError:
            opcode = Opcode.NOP
        return Instruction(opcode=opcode, raw=word)

    try:
        opcode = Opcode(opcode_val)
    except ValueError:
        opcode = Opcode.NOP

    return Instruction(opcode=opcode, rd=rd, rs1=rs1, rs2=rs2, imm=imm, raw=word)


def encode_instruction(opcode: Opcode, rd: int = 0, rs1: int = 0, rs2: int = 0, imm: int = 0) -> int:
    """Encode fields into a 16-bit instruction word."""
    if opcode.value >= 0x10:
        return (0x1E00 | ((opcode.value - 0x10) << 4)) & 0xFFFF
    return (
        ((opcode.value & 0xF) << 12)
        | ((rd & 0x7) << 9)
        | ((rs1 & 0x7) << 6)
        | ((rs2 & 0x7) << 3)
        | (imm & 0xFF)
    ) & 0xFFFF
