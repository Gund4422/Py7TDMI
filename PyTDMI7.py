# Py7TDMI7
# Copyright (C) 2025 Intiha
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
import numpy as np
from numba import njit

# --------------------------
# ARM7TDMI CPU
# --------------------------
class ARM7TDMI:
    def __init__(self, memory_array, trace=False):
        self.regs = np.zeros(16, dtype=np.uint32)  # r0-r15
        self.cpsr = np.uint32(0)
        self.thumb = False
        self.memory = memory_array.astype(np.uint8)
        self.trace = trace

    def step_batch(self, n=1000):
        self.regs, self.cpsr = jit_full_cpu(self.memory, self.regs, self.cpsr, n, self.thumb, self.trace)

# --------------------------
# Condition codes helpers
# --------------------------
@njit
def check_condition(cond, cpsr):
    N = (cpsr >> 31) & 1
    Z = (cpsr >> 30) & 1
    C = (cpsr >> 29) & 1
    V = (cpsr >> 28) & 1
    # ARM condition codes
    if cond == 0x0: return Z==1
    if cond == 0x1: return Z==0
    if cond == 0x2: return C==1
    if cond == 0x3: return C==0
    if cond == 0x4: return N==1
    if cond == 0x5: return N==0
    if cond == 0x6: return V==1
    if cond == 0x7: return V==0
    if cond == 0x8: return C==1 and Z==0
    if cond == 0x9: return C==0 or Z==1
    if cond == 0xA: return N==V
    if cond == 0xB: return N!=V
    if cond == 0xC: return Z==0 and N==V
    if cond == 0xD: return Z==1 or N!=V
    return True

# --------------------------
# Barrel Shifter
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
# Full JIT CPU
# --------------------------
@njit
def jit_full_cpu(memory, regs, cpsr, n, thumb, trace):
    mnemonics = np.array([
        "AND","EOR","SUB","RSB","ADD","ADC","SBC","RSC",
        "TST","TEQ","CMP","CMN","ORR","MOV","BIC","MVN"
    ])
    
    for _ in range(n):
        pc = regs[15]

        # ---- FETCH ----
        if thumb:
            instr = np.uint32(memory[pc] | (memory[pc+1]<<8))
            regs[15] = pc+2
        else:
            instr = np.uint32(
                memory[pc] | (memory[pc+1]<<8) | (memory[pc+2]<<16) | (memory[pc+3]<<24)
            )
            regs[15] = pc+4

        cond = (instr>>28)&0xF
        if not check_condition(cond, cpsr):
            continue

        opcode = (instr >> 21) & 0xF
        rd = (instr >> 12) & 0xF
        rn = (instr >> 16) & 0xF
        imm_flag = (instr >> 25) & 1
        imm = instr & 0xFFF
        rm = instr & 0xF

        # ---- TRACE ----
        if trace:
            if thumb:
                print(f"PC={hex(pc)}: THUMB {hex(instr)}")
            else:
                if opcode < 16:
                    if imm_flag:
                        print(f"PC={hex(pc)}: {mnemonics[opcode]} R{rd}, #{imm}")
                    else:
                        print(f"PC={hex(pc)}: {mnemonics[opcode]} R{rd}, R{rn}, R{rm}")

        # ---- EXECUTE DATA-PROCESSING ----
        val2 = imm if imm_flag else regs[rm]
        carry_in = (cpsr >> 29) & 1
        result = 0

        # AND
        if opcode == 0x0:
            result = regs[rn] & val2
        # EOR
        elif opcode == 0x1:
            result = regs[rn] ^ val2
        # SUB
        elif opcode == 0x2:
            result = (regs[rn] - val2) & 0xFFFFFFFF
        # RSB
        elif opcode == 0x3:
            result = (val2 - regs[rn]) & 0xFFFFFFFF
        # ADD
        elif opcode == 0x4:
            result = (regs[rn] + val2) & 0xFFFFFFFF
        # ADC
        elif opcode == 0x5:
            result = (regs[rn] + val2 + carry_in) & 0xFFFFFFFF
        # SBC
        elif opcode == 0x6:
            result = (regs[rn] - val2 - (1 - carry_in)) & 0xFFFFFFFF
        # RSC
        elif opcode == 0x7:
            result = (val2 - regs[rn] - (1 - carry_in)) & 0xFFFFFFFF
        # TST
        elif opcode == 0x8:
            result = regs[rn] & val2
        # TEQ
        elif opcode == 0x9:
            result = regs[rn] ^ val2
        # CMP
        elif opcode == 0xA:
            result = (regs[rn] - val2) & 0xFFFFFFFF
        # CMN
        elif opcode == 0xB:
            result = (regs[rn] + val2) & 0xFFFFFFFF
        # ORR
        elif opcode == 0xC:
            result = regs[rn] | val2
        # MOV
        elif opcode == 0xD:
            result = val2
        # BIC
        elif opcode == 0xE:
            result = regs[rn] & (~val2)
        # MVN
        elif opcode == 0xF:
            result = (~val2) & 0xFFFFFFFF

        # Update destination register for data-processing (except TST/TEQ/CMP/CMN)
        if opcode not in [0x8,0x9,0xA,0xB]:
            regs[rd] = result

        # Update CPSR flags (N, Z)
        if opcode not in [0x8,0x9,0xA,0xB] or opcode in [0x8,0x9,0xA,0xB]:
            cpsr = (cpsr & 0x0FFFFFFF) | ((result >> 31) << 31) | ((result == 0) << 30)

        # ---- BRANCH ----
        if (instr & 0x0F000000) == 0x0A000000:  # B
            offset = instr & 0x00FFFFFF
            if offset & 0x00800000: offset |= 0xFF000000
            regs[15] = (regs[15] + (offset << 2)) & 0xFFFFFFFF
        elif (instr & 0x0F000000) == 0x0B000000:  # BL
            offset = instr & 0x00FFFFFF
            if offset & 0x00800000: offset |= 0xFF000000
            regs[14] = regs[15]  # LR
            regs[15] = (regs[15] + (offset << 2)) & 0xFFFFFFFF

        # ---- LOAD/STORE WORD ----
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

    return regs, cpsr
