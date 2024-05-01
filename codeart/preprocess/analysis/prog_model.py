

import copy
from utils.asm_parser import ishexnumber, ispurenumber, isaddr

import networkx as nx

GLOBAL_VAR = "GLOBAL"
CONST_VAR = "CONST"
EFLAGS = "EFLAGS"


def cut(operand, chars):
    ret = ''
    while operand[0] in chars:
        ret += operand[0]
        operand = operand[1:]
        if len(operand) == 0:
            break
    return operand, ret

reg_normalize = {
    'al': 'rax', 'ah': 'rax', 'ax': 'rax', 'eax': 'rax', 'rax': 'rax',
    'bl': 'rbx', 'bh': 'rbx', 'bx': 'rbx', 'ebx': 'rbx', 'rbx': 'rbx',
    'cl': 'rcx', 'ch': 'rcx', 'cx': 'rcx', 'ecx': 'rcx', 'rcx': 'rcx',
    'dl': 'rdx', 'dh': 'rdx', 'dx': 'rdx', 'edx': 'rdx', 'rdx': 'rdx',
    'sil': 'rsi', 'si': 'rsi', 'esi': 'rsi', 'rsi': 'rsi',
    'dil': 'rdi', 'di': 'rdi', 'edi': 'rdi', 'rdi': 'rdi',
    'bpl': 'rbp', 'bp': 'rbp', 'ebp': 'rbp', 'rbp': 'rbp',
    'spl': 'rsp', 'sp': 'rsp', 'esp': 'rsp', 'rsp': 'rsp',
    'r8b': 'r8', 'r8w': 'r8', 'r8d': 'r8', 'r8': 'r8',
    'r9b': 'r9', 'r9w': 'r9', 'r9d': 'r9', 'r9': 'r9',
    'r10b': 'r10', 'r10w': 'r10', 'r10d': 'r10', 'r10': 'r10',
    'r11b': 'r11', 'r11w': 'r11', 'r11d': 'r11', 'r11': 'r11',
    'r12b': 'r12', 'r12w': 'r12', 'r12d': 'r12', 'r12': 'r12',
    'r13b': 'r13', 'r13w': 'r13', 'r13d': 'r13', 'r13': 'r13',
    'r14b': 'r14', 'r14w': 'r14', 'r14d': 'r14', 'r14': 'r14',
    'r15b': 'r15', 'r15w': 'r15', 'r15d': 'r15', 'r15': 'r15',
}


def _isregister(x):
    if x in reg_normalize:
        x = reg_normalize[x]
    registers = ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp',
                 'rsp', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']
    # registers=['rax','rbx','rcx','rdx','rsi','rdi','rbp','rsp','r8','r9','r10','r11','r12','r13','r14','r15']
    return x in registers

def _normalize_reg(x):
    if x in reg_normalize:
        x = reg_normalize[x]
    return x

def _is_mem_addr(x):
    if 'stack' in x:
        return False
    if not isaddr(x):
        return False
    if 'rbp' in x or 'rsp' in x:
        return False
    return True

def _is_important_var(x):    
    x = x.strip()
    if _isregister(x) or ispurenumber(x) or ishexnumber(x):
        return False
    if isaddr(x):
        if 'rbp' in x or 'rsp' in x:
            return False
        return True
    if x == GLOBAL_VAR:
        return True
    if x == CONST_VAR:
        return False
    if x == EFLAGS:
        return False  
    if 'stack' in x:
        return False
    return True

def _parse_operand(operator, location, operand1):
    """
    This function parses the operand of an instruction.
    Note that it does not normalize the operand for ML models.
    Instead, it keeps all the information in the operand.
    """
    operand1 = operand1.strip(' ')
    ret = ''

    if ('offset' in operand1 or 'off_' in operand1
            or 'loc_' in operand1 or 'unk_' in operand1
            or ('sub_' in operand1 and
                ('lea' in operator))):
        # operand1 = operand1.replace('offset ', '')
        return CONST_VAR
    if 'xmmword ptr' in operand1:
        # ret += 'xmmword ptr '
        operand1 = operand1.replace('xmmword ptr ', '')
    if 'dword ptr' in operand1:
        # ret += 'dword ptr '
        operand1 = operand1.replace('dword ptr ', '')
    if 'qword ptr' in operand1:
        # ret += 'qword ptr '
        operand1 = operand1.replace('qword ptr ', '')
    if 'word ptr' in operand1:
        # ret += 'word ptr '
        operand1 = operand1.replace('word ptr ', '')
    if 'byte ptr' in operand1:
        # ret += 'byte ptr '
        operand1 = operand1.replace('byte ptr ', '')
    if 'short ptr' in operand1:
        # ret += 'short ptr '
        operand1 = operand1.replace('short ptr ', '')
    if 'ptr' in operand1:
        # ret += 'ptr '
        operand1 = operand1.replace('ptr ', '')

    operand1 = operand1.replace('-', '+')

    operand1 = operand1.strip()

    if operand1[0:3] == 'cs:':
        # ret += 'cs : '
        operand1 = operand1[3:]
    if operand1[0:3] == 'ss:':
        # ret += 'ss : '
        operand1 = operand1[3:]
    if operand1[0:3] == 'fs:':
        # ret += 'fs : '
        operand1 = operand1[3:]
    if operand1[0:3] == 'ds:':
        # ret += 'ds : '
        operand1 = operand1[3:]
    if operand1[0:3] == 'es:':
        # ret += 'es : '
        operand1 = operand1[3:]
    if operand1[0:3] == 'gs:':
        # ret += 'gs : '
        operand1 = operand1[3:]

    operand1 = operand1.strip()

    if (operator[0] == 'j') and not _isregister(operand1):
        return CONST_VAR

    if (operand1[0:4] == 'loc_' or operand1[0:4] == 'off_'
        or operand1[0:4] == 'unk_' or operand1[0:4] == 'sub_'
        or operand1[0:4] == 'arg_' or operand1[0:4] == 'def_'
            or operand1[0:4] == 'var_'):
        ret += ''
        operand1 = operand1[4:]
        # skip characters 0-9
        operand1, value = cut(operand1, '0123456789ABCDEF')
        ret += value
    operand1 = operand1.strip()

    if len(operand1) == 0:
        return ret

    if operand1[0:6] == 'locret':
        # ret += 'hexvar '
        operand1 = operand1[6:]

    if operand1[0] == '(' and operand1[-1] == ')':
        return operand1[1:-1].strip()
        # ret += '( num ) '
        # return ret
    if operator == 'lea' and location == 2:
        # handle some address constants
        if not ishexnumber(operand1) and not isaddr(operand1):
            return operand1.strip()
    if operator == 'call' and location == 1:
        if len(operand1) > 3:
            return operand1.strip()
    if operator == 'extrn':
        return operand1.strip()

    operand1 = operand1.strip()
    if ishexnumber(operand1):
        # ret += operand1.strip()
        # return ret
        return CONST_VAR
    elif ispurenumber(operand1):
        # ret += operand1.strip()
        # return ret
        return CONST_VAR
    if isaddr(operand1):
        params = operand1[1:-1].split('+')
        for i in range(len(params)):
            params[i] = params[i].strip()
            if ishexnumber(params[i]):
                params[i] = params[i].strip()
            elif ispurenumber(params[i]):
                params[i] = params[i].strip()
            elif params[i][0:4] == 'var_':
                params[i] = params[i][4:].strip()
            elif params[i][0:4] == 'arg_':
                params[i] = params[i][4:].strip()
            elif not _isregister(params[i]):
                if params[i].find('*') == -1:
                    params[i] = params[i].strip()
                else:
                    # split by *
                    sub_params = params[i].split('*')
                    for j in range(len(sub_params)):
                        sub_params[j] = sub_params[j].strip()
                        if ispurenumber(sub_params[j]):
                            sub_params[j] = sub_params[j].strip()
                        elif ishexnumber(sub_params[j]):
                            sub_params[j] = sub_params[j].strip()
                    params[i] = '*'.join(sub_params)
            elif _isregister(params[i]):
                params[i] = _normalize_reg(params[i])
        s1 = '+'
        ret += '['+s1.join(params)+']'
        return ret

    if not _isregister(operand1) and len(operand1) > 4:
        ret += operand1
    elif not _isregister(operand1):
        ret += operand1
    if _isregister(operand1):
        ret += _normalize_reg(operand1)

    return ret

def get_use_reg_from_addr(addr:str):
    if 'rax' in addr:
        return 'rax'
    if 'rbx' in addr:
        return 'rbx'
    if 'rcx' in addr:
        return 'rcx'
    if 'rdx' in addr:
        return 'rdx'
    if 'rsi' in addr:
        return 'rsi'
    if 'rdi' in addr:
        return 'rdi'
    if 'r8' in addr:
        return 'r8'
    if 'r9' in addr:
        return 'r9'
    if 'r10' in addr:
        return 'r10'
    if 'r11' in addr:
        return 'r11'
    if 'r12' in addr:
        return 'r12'
    if 'r13' in addr:
        return 'r13'
    if 'r14' in addr:
        return 'r14'
    if 'r15' in addr:
        return 'r15'
    return None



class Instruction:

    def _tokenize_instr(self, code: str):
        annotation = None
        operator, operand = None, None
        operand1, operand2, operand3 = None, None, None
        if code.find(';') != -1:
            id = code.find(';')
            annotation = code[id+1:]
            code = code[0:id]
        if code.find(' ') != -1:
            id = code.find(' ')
            operand = code[id+1:]
            operator = code[0:id]
        else:
            operator = code
        if operand != None:
            if operand.find(',') != -1:
                strs = operand.split(',')
                if len(strs) == 2:
                    operand1, operand2 = strs[0], strs[1]
                else:
                    operand1, operand2, operand3 = strs[0], strs[1], strs[2]
            else:
                operand1 = operand
                operand2 = None

        if operand1 != None:
            operand1 = _parse_operand(operator, 1, operand1)
        if operand2 != None:
            operand2 = _parse_operand(operator, 2, operand2)
        if operand3 != None:
            operand3 = _parse_operand(operator, 3, operand3)
        return operator, operand1, operand2, operand3, annotation
    
    # define equal operator
    def __eq__(self, other):
        if isinstance(other, Instruction):
            return self.id == other.id and self.basic_block == other.basic_block
        return False
    
    # define hash function
    def __hash__(self):
        return self.id

    def __init__(self, code: str, id: int):
        self.id = id
        # properties
        self.is_important = False
        self.affected_vars = set()
        self.code = code
        self.hash_value = hash(code)
        self.basic_block = None
        operator, operand1, operand2, operand3, annotation = self._tokenize_instr(
            code)            
        self.operator = operator
        self.defs = []
        self.uses = []        
        if operand1 != None and isaddr(operand1):
            use_reg = get_use_reg_from_addr(operand1)
            if use_reg != None:
                self.uses.append(use_reg)            
        if operand2 != None and isaddr(operand2):
            use_reg = get_use_reg_from_addr(operand2)
            if use_reg != None:
                self.uses.append(use_reg)
        if operand3 != None and isaddr(operand3):
            use_reg = get_use_reg_from_addr(operand3)
            if use_reg != None:
                self.uses.append(use_reg)
        if 'ret' in operator:
            self.uses.append('rax')
        # insturctions that define EFLAGS
        if operator in set(['test', 'add', 'sub', 'and', 'or', 'xor', 'imul', 'idiv', 'imul', 'cmp', 'shl', 'shr', 'sar', 'shld', 'shrd', 'rol', 'ror', 'rcl', 'rcr', 'sal', 'sar', 'sbb', 'adc', 'neg', 'not', 'inc', 'dec', 'mul', 'div', 'imul', 'idiv']):
            self.defs.append(EFLAGS)
        # instructions that use EFLAGS
        if operator in set(['jz', 'jnz', 'jbe', 'ja', 'jle', 'jg', 'jnb', 'jb',
                            'jnl', 'jl', 'jno', 'jo', 'jnp', 'jp', 'jns', 'js', 'jnc', 'jc', 'jnge', 'jge',
                            'jng', 'jg', 'jne', 'je', 'jna', 'ja', 'jnae', 'jae', 'jnbe', 'jbe', 'jcxz', 'jecxz',
                            'loop', 'loope', 'loopne', 'setnz', 'setz', 'setnb', 'setb', 'setnl', 'setl', 'setno',
                            'seto', 'setnp', 'setp', 'setns', 'sets', 'setnc', 'setc', 'setnge', 'setge', 'setng',
                            'setg', 'setne', 'sete', 'setna', 'seta', 'setnae', 'setae', 'setnbe', 'setbe', 'cmovnz',
                            'cmovz', 'cmovnb', 'cmovb', 'cmovnl', 'cmovl', 'cmovno', 'cmovo', 'cmovnp', 'cmovp',
                            'cmovns', 'cmovs', 'cmovnc', 'cmovc', 'cmovnge', 'cmovge', 'cmovng', 'cmovg', 'cmovne',
                            'cmove', 'cmovna', 'cmova', 'cmovnae', 'cmovae', 'cmovnbe', 'cmovbe', 'adc']):
            self.uses.append(EFLAGS)
        if operator in set(['push']):
            self.defs.append("stack_%s" % operand1)
            self.uses.append(operand1)
        elif operator in set(['pop']):
            self.uses.append("stack_%s" % operand1)
            self.defs.append(operand1)
        elif operand1 != None and operand2 is None:
            if 'j' not in operator and 'call' not in operator and 'ret' not in operator:
                self.defs.append(operand1)
            if 'call' in operator:
                self.defs.append('rax')
            # only one operand
            self.uses.append(operand1)
            if _is_mem_addr(operand1):
                self.uses.append(GLOBAL_VAR)
            
        elif operand1 != None and operand2 != None:
            # two operands
            if 'test' in operator or 'cmp' in operator:
                # no define, only use (note that we have already added EFLAGS to uses)
                self.uses.append(operand1)
                if _is_mem_addr(operand1):
                    self.uses.append(GLOBAL_VAR)                
                self.uses.append(operand2)
                if _is_mem_addr(operand2):
                    self.uses.append(GLOBAL_VAR)                
            if 'mov' in operator:
                self.defs.append(operand1)      
                if _is_mem_addr(operand1):
                    self.defs.append(GLOBAL_VAR)
                self.uses.append(operand2)
                if _is_mem_addr(operand2):
                    self.uses.append(GLOBAL_VAR)                
            elif 'xchg' in operator and operand1 == operand2:
                # no define and no use
                # self.uses.append(operand1)
                pass
            elif 'xor' in operator and operand1 == operand2:
                # no use and define a constant value
                self.defs.append(operand1)                
            elif 'cmp' in operator:
                # all uses and no define
                self.uses.append(operand1)
                if _is_mem_addr(operand1):
                    self.uses.append(GLOBAL_VAR)                
                self.uses.append(operand2)      
                if _is_mem_addr(operand2):
                    self.uses.append(GLOBAL_VAR)
            else:
                # first define and all uses
                self.defs.append(operand1)
                if _is_mem_addr(operand1):
                    self.defs.append(GLOBAL_VAR)                    
                self.uses.append(operand1)
                if _is_mem_addr(operand1):
                    self.uses.append(GLOBAL_VAR)
                self.uses.append(operand2)
                if _is_mem_addr(operand2):
                    self.uses.append(GLOBAL_VAR)
        elif operand1 != None and operand2 != None and operand3 != None:
            # three operands
            self.defs.append(operand1)
            if _is_mem_addr(operand1):
                self.defs.append(GLOBAL_VAR)
            self.uses.append(operand2)
            if _is_mem_addr(operand2):
                self.uses.append(GLOBAL_VAR)                
            self.uses.append(operand3)
            if _is_mem_addr(operand3):
                self.uses.append(GLOBAL_VAR)

    def __str__(self):
        return self.code + ';\t\t defs:' + str(self.defs) + ',\t\t uses:' + str(self.uses)

    def __repr__(self) -> str:
        return self.__str__()

class BasicBlock:
    
    def __init__(self, addr, asms, id_begin):
        self.addr = addr
        self.instrs = []
        current_id = id_begin
        for asm in asms:
            self.instrs.append(Instruction(code=asm, id=current_id))
            current_id += 1
        for instr in self.instrs:
            instr.basic_block = self
        # self.use = set()
        # self.defs = set()

        # for instr in self.instrs:
        #     for current_use in instr.uses:
        #         # If I use a var that is defined in this BB
        #         # It is not a use for this BB
        #         if current_use not in self.defs:
        #             self.use.add(current_use)
        #     self.defs.update(instr.defs)


class Function:

    def __init__(self, cfg:nx.classes.digraph.DiGraph):
        self.cfg = cfg
        self.addr2our_bb = {}        
        instr_cnt = 0        
        for n in sorted(self.cfg.nodes):
            if instr_cnt > 2000:
                print("Too many instructions!")
                break
            self.addr2our_bb[n] = BasicBlock(n, self.cfg.nodes[n]['asm'], id_begin=instr_cnt)
            instr_cnt += len(self.cfg.nodes[n]['asm'])
        
        
class ReachDefinitionAnalysis:

    def __init__(self, func:Function):
        self.func = func
        self.cfg = func.cfg
        self.addr2our_bb = func.addr2our_bb
        self.bb_in = {} # bb -> dict [var -> set(Instruction)]
        self.bb_out = {} # bb -> dict [var -> set(Instruction)]
        # initialize
        for n in self.cfg.nodes:
            self.bb_in[n] = {}
            self.bb_out[n] = {}
        

    def merge(self, bb:BasicBlock):
        predecessors = sorted(self.cfg.predecessors(bb.addr))        
        previous_bb_in = copy.copy(self.bb_in[bb.addr])
        for pred in predecessors:                
            for var in self.bb_out[pred]:
                if var not in self.bb_in[bb.addr]:
                    self.bb_in[bb.addr][var] = set()
                self.bb_in[bb.addr][var] = self.bb_in[bb.addr][var].union(self.bb_out[pred][var])                

        if previous_bb_in != self.bb_in[bb.addr]:
            return True

        return False

    def propagate(self, bb:BasicBlock):
        self.bb_out[bb.addr] = copy.copy(self.bb_in[bb.addr])
        for instr in bb.instrs:
            for var in instr.defs:                
                if var != CONST_VAR and var != GLOBAL_VAR:
                    # kill other definitions
                    self.bb_out[bb.addr][var] = set([instr])
                elif var == GLOBAL_VAR:
                    # define, but not kill
                    if GLOBAL_VAR not in self.bb_out[bb.addr]:
                        self.bb_out[bb.addr][GLOBAL_VAR] = set()
                    self.bb_out[bb.addr][GLOBAL_VAR].add(instr)

            
    def run(self):
        done = False        
        cnt = 0
        while not done:
            # print("iteration %d" % cnt)
            cnt += 1
            done = True
            for bb_addr in sorted(self.addr2our_bb.keys()):
                bb = self.addr2our_bb[bb_addr]
                if self.merge(bb):
                    done = False
                self.propagate(bb)        
            
class PostDominatorAnalysis:

    def __init__(self, func:Function):
        self.func = func
        self.cfg = func.cfg
        self.addr2our_bb = func.addr2our_bb        
        self.post_dominator = {}
        for n in sorted(self.cfg.nodes):
            self.post_dominator[n] = set()            
            if self.cfg.out_degree(n) == 0:
                self.post_dominator[n] = set([n])
        self.back_edges = set()        
        # use stack-based DFS to find back edges
        stack = []
        visited = set()
        for n in sorted(self.cfg.nodes):
            if n in visited:
                continue
            stack.append(n)
            while len(stack) > 0:
                current = stack[-1]
                visited.add(current)
                for succ in sorted(self.cfg.successors(current)):
                    if succ in stack:
                        self.back_edges.add((current, succ))
                    elif succ in visited:
                        pass
                    else:
                        stack.append(succ)
                        break
                else:
                    stack.pop()
        self.back_edge_counters = {}
        for back_edge in self.back_edges:
            self.back_edge_counters[back_edge] = 0
        self.universal_set = set(self.cfg.nodes)
        
    def merge(self, bb_addr):
        successors = sorted(self.cfg.successors(bb_addr))        
        if len(successors) == 0:
            self.post_dominator[bb_addr] = set([bb_addr])
            return False        
        previous_post_dominator = copy.copy(self.post_dominator[bb_addr])
        current_post_dominators = set(self.universal_set)
        for succ in successors:
            if (bb_addr, succ) in self.back_edges and self.back_edge_counters[(bb_addr, succ)] == 0:
                self.back_edge_counters[(bb_addr, succ)] = 1
                current_post_dominators = current_post_dominators.intersection(self.universal_set)
            else:
                current_post_dominators = current_post_dominators.intersection(self.post_dominator[succ])
        self.post_dominator[bb_addr] = current_post_dominators.union(set([bb_addr]))
        if previous_post_dominator != self.post_dominator[bb_addr]:
            return True
            
        return False


    def run(self):
        done = False
        cnt = 0
        while not done:
            # print("iteration %d" % cnt)
            cnt += 1
            done = True
            for bb_addr in reversed(sorted(self.cfg.nodes())):
                # bb = self.addr2our_bb[bb_addr]
                if self.merge(bb_addr):
                    done = False
                

        
class ControlDependenceAnalysis:
    
    def __init__(self, func, post_dominator:PostDominatorAnalysis):
        self.func = func
        self.cfg = func.cfg
        self.addr2our_bb = func.addr2our_bb
        self.post_dominator = post_dominator
        self.addr2pdom_by = {} # addr -> set(addr)
        for n in sorted(self.cfg.nodes):
            self.addr2pdom_by[n] = set()
        
        for bb_addr in sorted(self.cfg.nodes):
            for pdom in self.post_dominator.post_dominator[bb_addr]:
                self.addr2pdom_by[pdom].add(bb_addr)
        
        self.control_dependence = {} # addr -> set(addr)
        for n in sorted(self.cfg.nodes):
            self.control_dependence[n] = set()

    def run(self):
        for bb_addr in sorted(self.cfg.nodes):
            for pdom_by in self.addr2pdom_by[bb_addr]:
                for pred in sorted(self.cfg.predecessors(pdom_by)):
                    if pred not in self.addr2pdom_by[bb_addr]:
                        self.control_dependence[bb_addr].add(pred)

        

class InstructionImportanceAnalysis:

    def _mark_important_vars(self, bb:BasicBlock):        
        for instr in bb.instrs:
            for var in instr.defs:                
                if _is_important_var(var):
                    instr.is_important = True
            if 'call' in instr.operator:
                instr.is_important = True
            if 'ret' in instr.operator:
                instr.is_important = True

    def __init__(self, cfg:nx.classes.DiGraph):
        self.func = Function(cfg)
        self.reach_def = ReachDefinitionAnalysis(self.func)
        self.reach_def.run()
        self.post_dominator = PostDominatorAnalysis(self.func)
        self.post_dominator.run()
        self.control_dependence = ControlDependenceAnalysis(self.func, self.post_dominator)
        self.control_dependence.run()
        self.addr2our_bb = self.func.addr2our_bb
        self.max_importance = 1e-5
        for bb_addr in sorted(self.addr2our_bb.keys()):
            bb = self.addr2our_bb[bb_addr]
            self._mark_important_vars(bb)
        for bb_addr in sorted(self.addr2our_bb.keys()):
            bb = self.addr2our_bb[bb_addr]
            for instr in bb.instrs:
                if instr.is_important:
                    self.max_importance += 1
                    instr.affected_vars = instr.affected_vars.union(set([instr.id]))
                    self._propagate_importance(instr)
        
    def get_instr_list(self):
        instr_list = []
        for bb_addr in sorted(self.addr2our_bb.keys()):
            bb = self.addr2our_bb[bb_addr]
            for instr in bb.instrs:
                instr.importance = len(instr.affected_vars)/self.max_importance
                instr_list.append(instr)
            
        return instr_list

    def _propagate_importance(self, instr:Instruction):
        instr_stack = [instr]
        visited = set()
        while len(instr_stack) > 0:
            current_instr = instr_stack.pop()
            current_uses = set(current_instr.uses)
            if 'call' in current_instr.operator:
                current_uses = current_uses.union(set(['rsi', 'rdi', 'rcx', 'rdx', 'r8', 'r9']))
            # first, see if the current block defines any of the uses
            for instr_in_current_bb in reversed(current_instr.basic_block.instrs):
                if instr_in_current_bb.id >= current_instr.id:
                    continue
                for var in instr_in_current_bb.defs:
                    if var in current_uses:
                        current_uses.remove(var)
                        if instr_in_current_bb not in visited:
                            instr_stack.append(instr_in_current_bb)                        
                            instr_in_current_bb.affected_vars.add(instr.id)
                            visited.add(instr_in_current_bb)
            # for all the uses that are not defined in the current block,
            # check the reach definition
            current_bb_reach_def = self.reach_def.bb_in[current_instr.basic_block.addr]
            for var in current_uses:
                if var in current_bb_reach_def:
                    # push each def into the stack
                    for def_instr in current_bb_reach_def[var]:
                        if def_instr not in visited:
                            instr_stack.append(def_instr)
                            def_instr.affected_vars.add(instr.id)
                            visited.add(def_instr)
            # add the last instruction of control dependence blocks
            control_deps = self.control_dependence.control_dependence[current_instr.basic_block.addr]
            for control_dep in control_deps:
                if control_dep in self.addr2our_bb:
                    control_dep_instr = self.addr2our_bb[control_dep].instrs[-1]
                    if control_dep_instr not in visited:
                        instr_stack.append(control_dep_instr)
                        control_dep_instr.affected_vars.add(instr.id)
                        visited.add(control_dep_instr)
