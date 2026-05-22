

# minimal implementation, emulation of the mvm







# from https://github.com/volcano852/pyevm/blob/master/pyevm/evm.py






from typing import Dict, List, Tuple

from instructions.instruction import Instruction, UnknownInstructionException


class StackEmptyException(Exception):
    pass


class StackOverflowException(Exception):
    pass


class VirtualMachine:
    def __init__(self):
        self.instruction_set: Dict[int, Instruction] = {}
        self.gas_available: int = 0
        self.pc: int = 0
        self.memory: Dict[int, int] = {}
        self.stack: List[int] = []
        self.gas_used: int = 0
        self.operations = []
        self.storage = {}

    def execute(self, operations: bytearray):
        self.operations = operations
        while self.pc < len(self.operations):
            op_code = operations[self.pc]
            try:
                instruction = self.instruction_set[op_code]
            except KeyError as err:
                raise UnknownInstructionException(f"{instruction} does not exist in the instruction set", err)

            args = self.stack_pop(instruction.number_input)
            self.gas_used += instruction.consume_gas(args)
            ret_values = instruction.execute(args, self)
            self.stack_push(ret_values)
            self.pc += 1

    def stack_pop(self, number: int) -> Tuple[int]:
        res = []
        i = 0
        while i < number and len(self.stack) > 0:
            res.append(self.stack.pop())
            i += 1
        if i < number:
            raise StackEmptyException(f"Stack is empty. Cannot pop {number - i} remaining values")
        return tuple(res)

    def stack_push(self, values: Tuple[int]):
        vals = list(values)
        for _ in values:
            val = vals.pop()
            self.stack.append(val)
            if len(self.stack) > 1024:
                raise StackOverflowException(f"Maximum stack size reached {len(self.stack)}. Cannot push {val}")









import logging
import sys

from evm import VirtualMachine
from instructions.instruction_set import instruction_set


def main(args):
    virtual_machine = VirtualMachine()
    virtual_machine.instruction_set = instruction_set
    virtual_machine.execute(bytearray.fromhex(args[0]))
    print(virtual_machine)


if __name__ == "__main__":
    logger = logging.getLogger("pyevm")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if len(sys.argv) < 1:
        print("** pyevm: missing opcodes in hexa")
        exit(1)
    try:
        main(sys.argv[1:])
    except Exception as err:
        logger.fatal("Error!", err)
        # print(err.with_traceback())
        exit(1)








# ---------------------------------------------------









# from https://github.com/saklani/tiny-evm/blob/main/tinyevm/evm.py






from tinyevm.context import Context
from tinyevm.instruction import Instruction
from tinyevm.instructions import INSTRUCTIONS
from tinyevm.storage import Storage
from tinyevm.utils import from_hex


class EVM:
    def __init__(self) -> None:
        self.context = None
        self.storage = Storage()

    def get_instruction(self, context:  Context) -> Instruction:
        """decodes instruction from the bytecode"""

        # Return STOP if program counter reaches end of program
        if context.program_counter >= len(context.code):
            return INSTRUCTIONS[0x00]

        # Extract byte from code
        opcode = from_hex(
            context.code[context.program_counter:  context.program_counter + 2])
        context.next()

        instruction = INSTRUCTIONS[opcode]
        return instruction

    def execute(self, code: str) -> str:
        self.context = Context(code=code, storage=self.storage)
        while self.context.run:
            instruction = self.get_instruction(self.context)
            instruction.fn(self.context)
        if self.context.return_value:
            return self.context.return_value
        return ""














from tinyevm.memory import Memory
from tinyevm.stack import Stack
from tinyevm.storage import Storage


class Context:

    def __init__(self, code: str, storage: Storage) -> None:
        self.code = code
        self.memory = Memory()
        self.program_counter = 0
        self.return_value = None
        self.run = True
        self.stack = Stack()
        self.storage = storage

    def stop(self) -> None:
        self.run = False

    def jump(self, target: int) -> None:
        self.program_counter = target

    def next(self) -> None:
        self.program_counter += 2








# ------------------------------------------------








# from https://github.com/zobront/pyevm/blob/main/pyevm.py









#!/usr/bin/env python3
import json
import math
import sha3
import copy 

def evm(code, state, block, tx):
    bytestring = code.hex()
    stack = []
    memory = {}
    storage = {}
    new_storage = copy.deepcopy(storage)

    highest_accessed_memory = 0

    while len(bytestring) > 0:
        next_inst = int(bytestring[:2], 16)
        bytestring = bytestring[2:]

        if next_inst == 0:
            storage = copy.deepcopy(new_storage)
            return [{}, stack]

        elif next_inst == 1:
            result = overflower(stack[0] + stack[1])
            stack = [result] + stack[2:]
        
        elif next_inst == 2:
            result = overflower(stack[0] * stack[1])
            stack = [result] + stack[2:]
       
        elif next_inst == 3:
            result = overflower(stack[0] - stack[1])
            stack = [result] + stack[2:]
        
        elif next_inst == 4:
            if stack[1] == 0:
                result = 0
            else:
                result = overflower(stack[0] // stack[1])
            stack = [result] + stack[2:]
        
        elif next_inst == 5:
            if stack[1] == 0:
                result = 0
            else:
                result = overflower(twos_comp(stack[0]) // twos_comp(stack[1]))
            stack = [result] + stack[2:]
        
        elif next_inst == 6:
            if stack[1] == 0: result = 0
            else:
                result = stack[0] % stack[1]
            stack = [result] + stack[2:]

        elif next_inst == 7:
            if stack[1] == 0: result = 0
            else:
                result = overflower(twos_comp(stack[0]) % twos_comp(stack[1]))
            stack = [result] + stack[2:]

        elif next_inst == int("10", 16):
            if stack[0] < stack[1]: result = 1
            else: result = 0
            stack = [result] + stack[2:]

        elif next_inst == int("11", 16):
            if stack[0] > stack[1]: result = 1
            else: result = 0
            stack = [result] + stack[2:]

        elif next_inst == int("12", 16):
            if twos_comp(stack[0]) < twos_comp(stack[1]): result = 1
            else: result = 0
            stack = [result] + stack[2:]
        
        elif next_inst == int("13", 16):
            if twos_comp(stack[0]) > twos_comp(stack[1]): result = 1
            else: result = 0
            stack = [result] + stack[2:]
        
        elif next_inst == int("14", 16):
            if stack[0] == stack[1]: result = 1
            else: result = 0
            stack = [result] + stack[2:]
        
        elif next_inst == int("15", 16):
            if stack[0] == 0: result = 1
            else: result = 0
            stack = [result] + stack[1:]

        elif next_inst == int("16", 16):
            result = stack[0] & stack[1]
            stack = [result] + stack[2:]

        elif next_inst == int("17", 16):
            result = stack[0] | stack[1]
            stack = [result] + stack[2:]
        
        elif next_inst == int("18", 16):
            result = stack[0] ^ stack[1]
            stack = [result] + stack[2:]
        
        elif next_inst == int("19", 16):
            result = overflower(~stack[0])
            stack = [result] + stack[1:]
        
        elif next_inst == int("1A", 16):
            if stack[0] > 32: result = 0
            else:
                result = stack[1] >> (248 - (8 * stack[0]))
            stack = [result] + stack[2:]
        
        elif next_inst == int("20", 16):
            value = [memory.get(i, 0) for i in range(stack[0], stack[0] + stack[1])]
            if stack[0] + stack[1] > highest_accessed_memory: 
                highest_accessed_memory = math.ceil((stack[0] + stack[1]) / 32) * 32
            hex_value = "".join([hex(i)[2:].zfill(2) for i in value])
            k = sha3.keccak_256()
            k.update(bytes.fromhex(hex_value))
            stack = [int(k.hexdigest(), 16)] + stack[2:]
        
        elif next_inst == int("30", 16):
            stack = [int(tx.get("to", 0), 16)] + stack

        elif next_inst == int("31", 16):
            stack = [int(state.get(hex(stack[0]), {}).get("balance", 0))] + stack[1:]

        elif next_inst == int("32", 16):
            stack = [int(tx.get("origin", ""), 16)] + stack

        elif next_inst == int("33", 16):
            stack = [int(tx.get("from", ""), 16)] + stack

        elif next_inst == int("34", 16):
            stack = [int(tx.get("value", 0))] + stack

        elif next_inst == int("35", 16):
            stack = [int(tx.get("data", 0)[stack[0]*2:].ljust(64, "0"), 16)] + stack[1:]

        elif next_inst == int("36", 16):
            stack = [len(str(tx.get("data", 0))) // 2] + stack

        elif next_inst == int("37", 16):
            value = bytes.fromhex(tx.get("data", "")[stack[1]*2:(stack[1] + stack[2])*2])
            for i in range(stack[2]):
                memory[stack[0] + i] = value[i] 
            if stack[0] + stack[2] > highest_accessed_memory: 
                highest_accessed_memory = math.ceil(stack[0] + stack[2])
            stack = stack[3:]
        
        elif next_inst == int("38", 16):
            stack = [len(code.hex()) // 2] + stack
        
        elif next_inst == int("39", 16):
            value = bytes.fromhex(code.hex()[stack[1]*2:(stack[1] + stack[2])*2].ljust(64, "0"))
            for i in range(stack[2]): memory[stack[0] + i] = value[i] 
            if stack[0] + stack[2] > highest_accessed_memory: 
                highest_accessed_memory = math.ceil(stack[0] + stack[2])
            stack = stack[3:]
        
        elif next_inst == int("3A", 16):
            stack = [int(tx.get("gasprice", 0))] + stack
        
        elif next_inst == int("3B", 16):
            stack = [len(state.get(hex(stack[0]), {}).get("code", {}).get("bin", "")) // 2] + stack[1:]

        elif next_inst == int("3C", 16):
            [addr, mem_offset, data_offset, size] = stack[:4]
            extcode = state.get(hex(addr), {}).get("code", {}).get("bin", "")
            value = bytes.fromhex(extcode[data_offset*2:(data_offset + size)*2].ljust(64, "0"))
            for i in range(stack[3]): memory[stack[1] + i] = value[i] 
            if stack[1] + stack[3] > highest_accessed_memory: 
                highest_accessed_memory = stack[1] + stack[3]
            stack = stack[4:]

        elif next_inst == int("41", 16):
            stack = [int(block.get("coinbase", 0), 16)] + stack

        elif next_inst == int("42", 16):
            stack = [int(block.get("timestamp", 0))] + stack
        
        elif next_inst == int("43", 16):
            stack = [int(block.get("number", 0))] + stack

        elif next_inst == int("44", 16):
            stack = [int(block.get("difficulty", 0), 16)] + stack

        elif next_inst == int("45", 16):
            stack = [int(block.get("gaslimit", 0), 16)] + stack

        elif next_inst == int("46", 16):
            stack = [int(block.get("chainid", 0))] + stack

        elif next_inst == int("47", 16):
            stack = [int(state.get(hex(int(tx.get("to", 0), 16)), {}).get("balance", 0))] + stack

        elif next_inst == int("50", 16):
            stack = stack[1:]
        
        elif next_inst == int("51", 16):
            value = [memory.get(i, 0) for i in range(stack[0], stack[0] + 32)]
            if stack[0] + 32 > highest_accessed_memory: 
                highest_accessed_memory = math.ceil((stack[0] + 32) / 32) * 32
            hex_value = "".join([hex(i)[2:].zfill(2) for i in value]) + "00" * (32 - len(value))
            stack = [int(hex_value, 16)] + stack[1:]

        elif next_inst == int("52", 16):
            value = bytes.fromhex(hex(stack[1])[2:].zfill(64))
            for i in range(32):
                memory[stack[0] + i] = value[i] 
            if stack[0] + 32 > highest_accessed_memory: 
                highest_accessed_memory = math.ceil((stack[0] + 32) / 32) * 32
            stack = stack[2:]
        
        elif next_inst == int("53", 16):
            value = bytes.fromhex(hex(stack[1])[-2:])
            memory[stack[0]] = stack[1]
            if stack[0] > highest_accessed_memory: 
                highest_accessed_memory = math.ceil((stack[0]) / 32) * 32
            stack = stack[2:]
        
        elif next_inst == int("54", 16):
            stack = [new_storage.get(stack[0], 0)] + stack[1:]

        elif next_inst == int("55", 16):
            [key, value] = stack[:2]
            new_storage[key] = value
            stack = stack[2:]
        
        elif next_inst == int("56", 16):
            bytestring = code.hex()[stack[0] * 2:]
            stack = stack[1:]
            assert(bytestring[:2] == "5b")

        elif next_inst == int("57", 16):
            if stack[1] > 0: 
                bytestring = code.hex()[stack[0] * 2:]
                assert(bytestring[:2] == "5b")
            stack = stack[2:]

        elif next_inst == int("58", 16):
            counter = (len(code.hex()) - (len(bytestring) + 2)) // 2
            stack = [counter] + stack

        elif next_inst == int("59", 16):
            stack = [highest_accessed_memory] + stack
        
        elif next_inst >= int("60", 16) and next_inst <= int("7F", 16):
            bytes_to_append = next_inst - int("60", 16) + 1
            stack = [int(bytestring[0:bytes_to_append * 2], 16)] + stack
            bytestring = bytestring[bytes_to_append * 2:]
            
        elif next_inst >= int("80", 16) and next_inst <= int("8F", 16):
            position_to_duplicate = next_inst - int("80", 16)
            stack = [stack[position_to_duplicate]] + stack

        elif next_inst >= int("90", 16) and next_inst <= int("9F", 16):
            position_to_swap = next_inst - int("90", 16) + 1
            tmp_stack_0 = stack[0]
            stack[0] = stack[position_to_swap]
            stack[position_to_swap] = tmp_stack_0
        
        elif next_inst == int("F1", 16):
            [gas, addr, value, argsOffset, argsSize, retOffset, retSize] = stack[:7]
            called_code = bytes.fromhex(state.get(hex(addr), {}).get("code", {}).get("bin", ""))
            tx["from"] = tx.get("to", "")
            tx["to"] = addr
            [return_value, _] = evm(called_code, state, block, tx)
            success = return_value.get("success", False)
            result = bytes.fromhex(return_value.get("return", "").zfill(retSize * 2))

            if success == "true":
                stack = [1] + stack[7:]
            else:
                stack = [0] + stack[7:]
            
            for i in range(retSize):
                memory[retOffset + i] = result[i] 
            if retOffset + retSize > highest_accessed_memory: 
                highest_accessed_memory = retOffset + retSize

        
        elif next_inst == int("F3", 16):
            [offset, size] = stack[:2]
            value = [memory.get(i, 0) for i in range(offset, offset + size)]
            result = "".join([hex(val)[2:] for val in value])
            storage = copy.deepcopy(new_storage)
            return [{"success":"true", "return":result}, stack[2:]]
        
        elif next_inst == int("FD", 16):
            [offset, size] = stack[:2]
            value = [memory.get(i, 0) for i in range(offset, offset + size)]
            result = "".join([hex(val)[2:] for val in value])
            return [{"success":"false", "return":result}, stack[2:]]
    
    return [{}, stack]






def overflower(i, bits=256):
    return i % 2**bits





def twos_comp(val, bits=256):
    if ((val & (1 << (bits - 1))) != 0):
        val = val - (1<<bits)
    return val







   
def test():
    with open('./evm.json') as f:
        data = json.load(f)
        total = len(data)

        for i, test in enumerate(data):
            print("Test #" + str(i + 1), "of", total, test['name'])

            # Note: as the test cases get more complex, you'll need to modify this
            # to pass down more arguments to the evm function
            code = bytes.fromhex(test['code']['bin'])
            tx = test.get("tx", {})
            state = test.get("state", {})
            block = test.get("block", {})
            [returned, stack] = evm(code, state, block, tx)

            expected_stack = [int(x, 16) if x.startswith("0x") else int(x) for x in test['expect'].get('stack', [])]

            if stack != expected_stack:
                print("Stack doesn't match")
                print(" expected:", expected_stack)
                print("   actual:", stack)
                print("")
                print("Test code:")
                print(test['code']['asm'])
                print("")
                print("Progress: " + str(i) + "/" + str(len(data)))
                print("")
                break
        
            expected_return = test["expect"].get("return", "0x")
            return_value = returned.get("return", "0x")
            if return_value != expected_return:
                print("Return value doesn't match")
                print(" expected:", expected_return)
                print("   actual:", return_value)
                print("")
                print("Test code:")
                print(test['code']['asm'])
                print("")
                print("Progress: " + str(i) + "/" + str(len(data)))
                print("")
                break


if __name__ == '__main__':
    test()




