import numpy as np
from numba import njit

# --------------------------
# Disassembler
# --------------------------
class Disassembler:
    ARM_OPCODES = {
        0x0: "AND", 0x1: "EOR", 0x2: "SUB", 0x3: "RSB",
        0x4: "ADD", 0x5: "ADC", 0x6: "SBC", 0x7: "RSC",
        0x8: "TST", 0x9: "TEQ", 0xA: "CMP", 0xB: "CMN",
        0xC: "ORR", 0xD: "MOV", 0xE: "BIC", 0xF: "MVN"
    }

    @staticmethod
    def disassemble_arm(instr):
        cond = (instr >> 28) & 0xF
        opcode = (instr >> 21) & 0xF
        rd = (instr >> 12) & 0xF
        rn = (instr >> 16) & 0xF
        imm_flag = (instr >> 25) & 1
        imm = instr & 0xFFF

        mnemonic = Disassembler.ARM_OPCODES.get(opcode, "UNKNOWN")
        if imm_flag:
            return f"{mnemonic} R{rd}, #{imm}"
        else:
            rm = instr & 0xF
            return f"{mnemonic} R{rd}, R{rn}, R{rm}"

# --------------------------
# ARM7TDMI CPU
# --------------------------
class ARM7TDMI:
    def __init__(self, memory_array, trace=False):
        self.regs = np.zeros(16, dtype=np.uint32)  # r0-r15
        self.cpsr = np.uint32(0)
        self.spsr = np.uint32(0)
        self.thumb = False
        self.pc = np.uint32(0x08000000)
        self.memory = memory_array.astype(np.uint8)
        self.trace = trace
        self.disassembler = Disassembler()

    def step_batch(self, n=1000):
        self.regs, self.pc, self.cpsr = jit_full_cpu(
            self.memory, self.regs, self.pc, self.cpsr, n, self.thumb, self.trace
        )

# --------------------------
# Numba JIT CPU with optional tracing
# --------------------------
@njit
def jit_full_cpu(memory, regs, pc, cpsr, n, thumb, trace):
    regs = regs.astype(np.uint32)
    pc = np.uint32(pc)
    cpsr = np.uint32(cpsr)

    for _ in range(n):
        # Fetch instruction
        if thumb:
            instr = np.uint32(memory[pc] | (memory[pc+1] << np.uint32(8)))
            pc += np.uint32(2)
        else:
            instr = np.uint32(
                memory[pc] |
                (memory[pc+1] << np.uint32(8)) |
                (memory[pc+2] << np.uint32(16)) |
                (memory[pc+3] << np.uint32(24))
            )
            pc += np.uint32(4)

        # Trace (print disassembly)
        if trace:
            # simple disassembler output, limited in njit
            opcode = (instr >> 21) & np.uint32(0xF)
            rd = (instr >> 12) & np.uint32(0xF)
            rn = (instr >> 16) & np.uint32(0xF)
            imm_flag = (instr >> 25) & np.uint32(1)
            imm = instr & np.uint32(0xFFF)
            mnemonic = ["AND","EOR","SUB","RSB","ADD","ADC","SBC","RSC",
                        "TST","TEQ","CMP","CMN","ORR","MOV","BIC","MVN"][opcode]
            if imm_flag:
                print(f"PC={hex(pc-4)}: {mnemonic} R{rd}, #{imm}")
            else:
                rm = instr & np.uint32(0xF)
                print(f"PC={hex(pc-4)}: {mnemonic} R{rd}, R{rn}, R{rm}")

        # Decode & execute (data processing)
        opcode = (instr >> 21) & np.uint32(0xF)
        rd = (instr >> 12) & np.uint32(0xF)
        rn = (instr >> 16) & np.uint32(0xF)
        imm_flag = (instr >> 25) & np.uint32(1)
        imm = instr & np.uint32(0xFFF)

        if opcode == np.uint32(0xD):  # MOV
            regs[rd] = imm if imm_flag else regs[instr & np.uint32(0xF)]
        elif opcode == np.uint32(0x4):  # ADD
            val = imm if imm_flag else regs[instr & np.uint32(0xF)]
            regs[rd] = (regs[rn] + val) & np.uint32(0xFFFFFFFF)
        elif opcode == np.uint32(0x2):  # SUB
            val = imm if imm_flag else regs[instr & np.uint32(0xF)]
            regs[rd] = (regs[rn] - val) & np.uint32(0xFFFFFFFF)

        # Branch
        if (instr & np.uint32(0x0F000000)) == np.uint32(0x0A000000):  # B
            offset = instr & np.uint32(0x00FFFFFF)
            if offset & np.uint32(0x00800000):
                offset |= np.uint32(0xFF000000)
            pc = (pc + (offset << np.uint32(2))) & np.uint32(0xFFFFFFFF)
        elif (instr & np.uint32(0x0F000000)) == np.uint32(0x0B000000):  # BL
            offset = instr & np.uint32(0x00FFFFFF)
            if offset & np.uint32(0x00800000):
                offset |= np.uint32(0xFF000000)
            regs[14] = pc  # LR
            pc = (pc + (offset << np.uint32(2))) & np.uint32(0xFFFFFFFF)

        # Load/Store simplified
        elif (instr & np.uint32(0x0C000000)) == np.uint32(0x04000000):
            rn_val = regs[rn]
            if ((instr >> np.uint32(20)) & np.uint32(1)) == np.uint32(1):  # LDR
                regs[rd] = (
                    memory[rn_val] |
                    (memory[rn_val+1] << np.uint32(8)) |
                    (memory[rn_val+2] << np.uint32(16)) |
                    (memory[rn_val+3] << np.uint32(24))
                )
            else:  # STR
                val = regs[rd]
                memory[rn_val] = val & np.uint8(0xFF)
                memory[rn_val+1] = (val >> np.uint8(8)) & np.uint8(0xFF)
                memory[rn_val+2] = (val >> np.uint8(16)) & np.uint8(0xFF)
                memory[rn_val+3] = (val >> np.uint8(24)) & np.uint8(0xFF)

    return regs, pc, cpsr
