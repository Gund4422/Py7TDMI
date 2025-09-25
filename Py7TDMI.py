"""
Py7TDMI - Inti-License V2
Copyright (C) 2025 Intiha (aka Gund4422)

This software is licensed under Inti-License Version 2 (V2).
You may use, modify, and redistribute it, but all derivatives must remain under Inti-License V2.
Please credit Intiha / Gund4422 and ARM Holdings for the original CPU design.
"""

import numpy as np
from numba import njit

# --------------------------
# ARM7TDMI CPU Constants & Helpers
# --------------------------

# ARM Condition Codes Mapping
COND_CODES = {
    0x0: 'EQ', 0x1: 'NE', 0x2: 'CS', 0x3: 'CC',
    0x4: 'MI', 0x5: 'PL', 0x6: 'VS', 0x7: 'VC',
    0x8: 'HI', 0x9: 'LS', 0xA: 'GE', 0xB: 'LT',
    0xC: 'GT', 0xD: 'LE', 0xE: 'AL'
}

# ARM Data Processing Opcodes Mapping
DP_OPCODES = {
    0x0: 'AND', 0x1: 'EOR', 0x2: 'SUB', 0x3: 'RSB',
    0x4: 'ADD', 0x5: 'ADC', 0x6: 'SBC', 0x7: 'RSC',
    0x8: 'TST', 0x9: 'TEQ', 0xA: 'CMP', 0xB: 'CMN',
    0xC: 'ORR', 0xD: 'MOV', 0xE: 'BIC', 0xF: 'MVN'
}

# --------------------------
# Condition Check Function
# --------------------------
@njit
def check_condition(cond, cpsr):
    N = (cpsr >> 31) & 1
    Z = (cpsr >> 30) & 1
    C = (cpsr >> 29) & 1
    V = (cpsr >> 28) & 1

    if cond == 0x0: return Z == 1  # EQ
    if cond == 0x1: return Z == 0  # NE
    if cond == 0x2: return C == 1  # CS
    if cond == 0x3: return C == 0  # CC
    if cond == 0x4: return N == 1  # MI
    if cond == 0x5: return N == 0  # PL
    if cond == 0x6: return V == 1  # VS
    if cond == 0x7: return V == 0  # VC
    if cond == 0x8: return C == 1 and Z == 0  # HI
    if cond == 0x9: return C == 0 or Z == 1   # LS
    if cond == 0xA: return N == V              # GE
    if cond == 0xB: return N != V              # LT
    if cond == 0xC: return Z == 0 and N == V  # GT
    if cond == 0xD: return Z == 1 or N != V   # LE
    return True  # AL (Always)

# --------------------------
# Barrel Shifter Function
# --------------------------
@njit
def barrel_shift(val, shift_type, shift_amount, cpsr):
    C = (cpsr >> 29) & 1
    result = val
    carry = C

    if shift_type == 0:  # LSL
        if shift_amount == 0:
            result = val
        else:
            result = (val << shift_amount) & 0xFFFFFFFF
            carry = (val >> (32 - shift_amount)) & 1

    elif shift_type == 1:  # LSR
        if shift_amount == 0:
            result = 0
            carry = (val >> 31) & 1
        else:
            result = (val >> shift_amount) & 0xFFFFFFFF
            carry = (val >> (shift_amount - 1)) & 1

    elif shift_type == 2:  # ASR
        if shift_amount == 0:
            carry = (val >> 31) & 1
            result = 0xFFFFFFFF if (val & 0x80000000) else 0
        else:
            result = np.int32(val) >> shift_amount
            carry = (val >> (shift_amount - 1)) & 1

    elif shift_type == 3:  # ROR
        if shift_amount == 0:
            result = ((C << 31) | (val >> 1)) & 0xFFFFFFFF
            carry = val & 1
        else:
            result = ((val >> shift_amount) | (val << (32 - shift_amount))) & 0xFFFFFFFF
            carry = (val >> (shift_amount - 1)) & 1

    return result, carry

# --------------------------
# ARM7TDMI CPU Class
# --------------------------
class ARM7TDMI:
    def __init__(self, memory_array, trace=False):
        # General-purpose registers R0-R15
        self.regs = np.zeros(16, dtype=np.uint32)

        # Current Program Status Register
        self.cpsr = np.uint32(0)

        # Saved Program Status Register
        self.spsr = np.uint32(0)

        # Thumb mode flag
        self.thumb = False

        # Program Counter
        self.pc = np.uint32(0x08000000)

        # Memory array (byte-addressable)
        self.memory = memory_array.astype(np.uint8)

        # Trace flag (for printing instructions)
        self.trace = trace

        # Internal instruction counter
        self.instr_count = 0

        # Debug logs
        self.debug_log = []

    # --------------------------
    # Step a batch of instructions
    # --------------------------
    def step_batch(self, n=1000):
        self.regs, self.pc, self.cpsr = jit_full_cpu(
            self.memory, self.regs, self.pc, self.cpsr, n, self.thumb, self.trace
        )

# --------------------------
# Full JIT CPU Execution Function
# --------------------------
@njit
def jit_full_cpu(memory, regs, pc, cpsr, n, thumb, trace):
    # Preallocate mnemonics for trace printing
    mnemonics = np.array([
        'AND','EOR','SUB','RSB','ADD','ADC','SBC','RSC',
        'TST','TEQ','CMP','CMN','ORR','MOV','BIC','MVN'
    ])

    for _ in range(n):
        curr_pc = pc

        # --------------------------
        # Fetch Instruction
        # --------------------------
        if thumb:
            instr = np.uint32(memory[pc] | (memory[pc+1] << 8))
            pc += 2
        else:
            instr = np.uint32(
                memory[pc] |
                (memory[pc+1] << 8) |
                (memory[pc+2] << 16) |
                (memory[pc+3] << 24)
            )
            pc += 4

        # --------------------------
        # Condition Check
        # --------------------------
        cond = (instr >> 28) & 0xF
        if not check_condition(cond, cpsr):
            continue

        # --------------------------
        # Decode Data Processing Fields
        # --------------------------
        opcode = (instr >> 21) & 0xF
        rd = (instr >> 12) & 0xF
        rn = (instr >> 16) & 0xF
        imm_flag = (instr >> 25) & 1
        imm = instr & 0xFFF
        rm = instr & 0xF

        # --------------------------
        # Trace / Debug Print
        # --------------------------
        if trace:
            if imm_flag:
                print(f"PC={hex(curr_pc)}: {mnemonics[opcode]} R{rd}, #{imm}")
            else:
                print(f"PC={hex(curr_pc)}: {mnemonics[opcode]} R{rd}, R{rn}, R{rm}")

        # --------------------------
        # Execute Data Processing Instructions
        # --------------------------
        val2 = imm if imm_flag else regs[rm]
        carry_in = (cpsr >> 29) & 1
        result = 0

        if opcode == 0x0:  # AND
            result = regs[rn] & val2
        elif opcode == 0x1:  # EOR
            result = regs[rn] ^ val2
        elif opcode == 0x2:  # SUB
            result = (regs[rn] - val2) & 0xFFFFFFFF
        elif opcode == 0x3:  # RSB
            result = (val2 - regs[rn]) & 0xFFFFFFFF
        elif opcode == 0x4:  # ADD
            result = (regs[rn] + val2) & 0xFFFFFFFF
        elif opcode == 0x5:  # ADC
            result = (regs[rn] + val2 + carry_in) & 0xFFFFFFFF
        elif opcode == 0x6:  # SBC
            result = (regs[rn] - val2 - (1 - carry_in)) & 0xFFFFFFFF
        elif opcode == 0x7:  # RSC
            result = (val2 - regs[rn] - (1 - carry_in)) & 0xFFFFFFFF
        elif opcode == 0x8:  # TST
            result = regs[rn] & val2
        elif opcode == 0x9:  # TEQ
            result = regs[rn] ^ val2
        elif opcode == 0xA:  # CMP
            result = (regs[rn] - val2) & 0xFFFFFFFF
        elif opcode == 0xB:  # CMN
            result = (regs[rn] + val2) & 0xFFFFFFFF
        elif opcode == 0xC:  # ORR
            result = regs[rn] | val2
        elif opcode == 0xD:  # MOV
            result = val2
        elif opcode == 0xE:  # BIC
            result = regs[rn] & (~val2)
        elif opcode == 0xF:  # MVN
            result = (~val2) & 0xFFFFFFFF

        # Update destination register for DP (except TST/TEQ/CMP/CMN)
        if opcode not in [0x8,0x9,0xA,0xB]:
            regs[rd] = result

        # Update CPSR N/Z flags
        cpsr = (cpsr & 0x0FFFFFFF) | ((result >> 31) << 31) | ((result == 0) << 30)

        # --------------------------
        # Branch Instructions (B / BL)
        # --------------------------
        if (instr & 0x0F000000) == 0x0A000000:  # B
            offset = instr & 0x00FFFFFF
            if offset & 0x00800000: offset |= 0xFF000000
            pc = (pc + (offset << 2)) & 0xFFFFFFFF

        elif (instr & 0x0F000000) == 0x0B000000:  # BL
            offset = instr & 0x00FFFFFF
            if offset & 0x00800000: offset |= 0xFF000000
            regs[14] = pc  # LR
            pc = (pc + (offset << 2)) & 0xFFFFFFFF

        # --------------------------
        # Load/Store Word (simplified)
        # --------------------------
        elif (instr & 0x0C000000) == 0x04000000:
            addr = regs[rn]
            if ((instr >> 20) & 1):  # LDR
                regs[rd] = (
                    memory[addr] |
                    (memory[addr+1]<<8) |
                    (memory[addr+2]<<16) |
                    (memory[addr+3]<<24)
                )
            else:  # STR
                val = regs[rd]
                memory[addr] = val & 0xFF
                memory[addr+1] = (val>>8) & 0xFF
                memory[addr+2] = (val>>16) & 0xFF
                memory[addr+3] = (val>>24) & 0xFF

    return regs, pc, cpsr
