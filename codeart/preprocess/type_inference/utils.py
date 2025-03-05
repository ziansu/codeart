import json
import os
import re

class TypeInfoEntry:
    def __init__(self, raw_entry):
        self.type = raw_entry['type']
        self.low_pc = int(raw_entry['low_pc'], 16)
        self.high_pc = int(raw_entry['high_pc'], 16)
        self.loc = self._normalize_loc(raw_entry['loc'])        

    def _normalize_loc(self, loc):
        if loc[0].startswith('fbreg'):
            # fbreg(rbp) --> RBP
            # fbreg(rsp) --> RSP
            normalized_reg_str = loc[0].replace('fbreg', '').replace('(', '').replace(')', '')
            normalized_reg_str = normalized_reg_str.upper()
            return "[%s%x]" % (normalized_reg_str, loc[1])
        elif loc[0].startswith('reg'):
            reg_name = loc[0].split('_')[1]
            return reg_name
        elif loc[0].startswith('breg'):
            reg_name = loc[0].split('_')[1]
            return "[%s%x]" % (reg_name, loc[1])
        else:
            return None
        
    def __repr__(self) -> str:
        return "TIE-[%x, %x)@%s: %s"%(
            self.low_pc, self.high_pc, self.loc, self.type
        )

class DebugInfoIntervalTreeNode:
    def __init__(self, med, parent):
        self.med = med
        self.parent = parent        
        self.left = None
        self.right = None

    def __repr__(self) -> str:
        return "Node-%x" % (self.med)
        
def _build_tree(sorted_points, ties, parent):
    if not sorted_points:
        return None
    
    mid = len(sorted_points) // 2
    median = sorted_points[mid]
    # intervals that are on the left
    left_ties = [tie for tie in ties if tie.high_pc < median]
    # intervals that are on the right
    right_ties = [tie for tie in ties if tie.low_pc > median]
    # intervals that contain the median
    median_ties = [tie for tie in ties if tie.low_pc <= median and tie.high_pc >= median]
    node = DebugInfoIntervalTreeNode(median, parent)
    if len(left_ties) > 0:
        node.left = _build_tree(sorted_points[:mid], left_ties, node)
    if len(right_ties) > 0:
        node.right = _build_tree(sorted_points[mid:], right_ties, node)
    my_tie_sorted_by_low = sorted(median_ties, key=lambda x: x.low_pc)
    node.ties_sorted_by_low = my_tie_sorted_by_low
    my_tie_sorted_by_high = sorted(median_ties, key=lambda x: x.high_pc, reverse=True)
    node.ties_sorted_by_high = my_tie_sorted_by_high
    return node


def _precompute_loc_info(node):
    if not node:
        return
    my_locs = set()
    for tie in node.ties_sorted_by_low:
        my_locs.add(tie.loc)
    node.my_locs = my_locs
    my_children_locs = set()
    if node.left:
        _precompute_loc_info(node.left)
        my_children_locs.update(node.left.locs_all)
    if node.right:
        _precompute_loc_info(node.right)
        my_children_locs.update(node.right.locs_all)

    node.locs_all = my_locs | my_children_locs


def _query_tree(addr, loc, node):
    match_all = loc == '**'
    if loc not in node.locs_all and not match_all:
        return None
    related_ties = []
    if addr <= node.med:
        if loc in node.my_locs or match_all:
            for tie in node.ties_sorted_by_low:
                if (tie.loc == loc or match_all) and tie.low_pc <= addr:
                    related_ties.append(tie)
                if tie.low_pc > addr:
                    break
        if node.left:
            left_ret = _query_tree(addr, loc, node.left)
            if left_ret:
              related_ties.extend(left_ret)
    else:
        if loc in node.my_locs or match_all:
            for tie in node.ties_sorted_by_high:
                if (tie.loc == loc or match_all) and tie.high_pc > addr:
                    related_ties.append(tie)
                if tie.high_pc <= addr:
                    break
        if node.right:
            right_ret = _query_tree(addr, loc, node.right)
            if right_ret:
              related_ties.extend(right_ret)
    if len(related_ties) == 0:
        return None
    return related_ties

class BinaryDebugInfoSearcher:
    
    def __init__(self, file_name):
        self.file_name = file_name
        self.dbg_info_list_raw = []
        if os.path.exists(file_name):
            self._load()
        self.interval_entries = []
        for entry in self.dbg_info_list_raw:
            typeinfo_entry = TypeInfoEntry(entry)
            if typeinfo_entry.loc is not None:
                self.interval_entries.append(typeinfo_entry)            
        interval_points = set()
        for entry in self.interval_entries:
            interval_points.add(entry.low_pc)
            interval_points.add(entry.high_pc)
        interval_points_sorted = sorted(list(interval_points))
        # create interval tree
        self.root = _build_tree(interval_points_sorted, self.interval_entries, None)
        # also pre-compute the location information for each tree node
        _precompute_loc_info(self.root)

    def _load(self):        
        with open(self.file_name, 'r') as fin:
            for line in fin:
                data = json.loads(line)
                self.dbg_info_list_raw.append(data)

    def query(self, addr, loc):
        if len(self.interval_entries) == 0:
            return None
        return _query_tree(addr, loc, self.root)
        



reg_normalize = {
    'al': 'RAX', 'ah': 'RAX', 'ax': 'RAX', 'eax': 'RAX', 'rax': 'RAX',
    'bl': 'RBX', 'bh': 'RBX', 'bx': 'RBX', 'ebx': 'RBX', 'rbx': 'RBX',
    'cl': 'RCX', 'ch': 'RCX', 'cx': 'RCX', 'ecx': 'RCX', 'rcx': 'RCX',
    'dl': 'RDX', 'dh': 'RDX', 'dx': 'RDX', 'edx': 'RDX', 'rdx': 'RDX',
    'sil': 'RSI', 'si': 'RSI', 'esi': 'RSI', 'rsi': 'RSI',
    'dil': 'RDI', 'di': 'RDI', 'edi': 'RDI', 'rdi': 'RDI',
    'bpl': 'RBP', 'bp': 'RBP', 'ebp': 'RBP', 'rbp': 'RBP',
    'spl': 'RSP', 'sp': 'RSP', 'esp': 'RSP', 'rsp': 'RSP',
    'r8b': 'R8', 'r8w': 'R8', 'r8d': 'R8', 'r8': 'R8',
    'r9b': 'R9', 'r9w': 'R9', 'r9d': 'R9', 'r9': 'R9',
    'r10b': 'R10', 'r10w': 'R10', 'r10d': 'R10', 'r10': 'R10',
    'r11b': 'R11', 'r11w': 'R11', 'r11d': 'R11', 'r11': 'R11',
    'r12b': 'R12', 'r12w': 'R12', 'r12d': 'R12', 'r12': 'R12',
    'r13b': 'R13', 'r13w': 'R13', 'r13d': 'R13', 'r13': 'R13',
    'r14b': 'R14', 'r14w': 'R14', 'r14d': 'R14', 'r14': 'R14',
    'r15b': 'R15', 'r15w': 'R15', 'r15d': 'R15', 'r15': 'R15',
}

NUM_PAT = re.compile(r'[0-9]+')

def _eval_addr_str(addr_str):
    if '+' in addr_str:
        plus_idx = addr_str.find('+')
        base_str = addr_str[:plus_idx]
        offset_str = addr_str[plus_idx+1:]
        base_expr = _eval_addr_str(base_str)
        offset_expr = _eval_addr_str(offset_str)
        if type(base_expr) == int and type(offset_expr) == int:
            return base_expr + offset_expr
        elif type(offset_expr) == int and offset_expr > 0:
            return "%s+%x" % (base_expr, offset_expr)
        elif type(offset_expr) == int and offset_expr < 0:
            return "%s-%x" % (base_expr, -offset_expr)        
        elif type(offset_expr) == int and offset_expr == 0:            
            return base_expr        
    elif '-' in addr_str:
        minus_idx = addr_str.find('-')
        base_str = addr_str[:minus_idx]
        offset_str = addr_str[minus_idx+1:]
        base_expr = _eval_addr_str(base_str)
        offset_expr = _eval_addr_str(offset_str)
        if type(base_expr) == int and type(offset_expr) == int:
            return base_expr - offset_expr
        if type(base_expr) == int and type(offset_expr) == int:
            return base_expr - offset_expr
        elif type(offset_expr) == int and offset_expr > 0:
            return "%s-%s" % (base_expr, offset_expr)
        elif type(offset_expr) == int and offset_expr < 0:
            return "%s+%x" % (base_expr, -offset_expr)        
        elif type(offset_expr) == int and offset_expr == 0:            
            return base_expr
    elif '*' in addr_str:
        mul_idx = addr_str.find('*')
        base_str = addr_str[:mul_idx]
        offset_str = addr_str[mul_idx+1:]
        base_expr = _eval_addr_str(base_str)
        offset_expr = _eval_addr_str(offset_str)
        if type(base_expr) == int and type(offset_expr) == int:
            return base_expr * offset_expr
        elif type(offset_expr) == int and offset_expr > 0:
            return "%s*%s" % (base_expr, offset_expr)
        elif type(offset_expr) == int and offset_expr < 0:
            return "%s*%x" % (base_expr, -offset_expr)
        elif type(offset_expr) == int and offset_expr == 0:
            return 0            
    elif addr_str.strip() in reg_normalize:
        return reg_normalize[addr_str.strip()]
    elif addr_str.startswith('var_'):
        # try to parse it as a hex number
        hex_num = addr_str[4:]
        try:
            return -int(hex_num, 16)
        except:
            pass
        return addr_str
    elif addr_str.endswith('h'):
        hex_num = addr_str[:-1]
        try:
            return int(hex_num, 16)
        except:
            pass
        return addr_str
    elif NUM_PAT.match(addr_str):
        return int(addr_str)
    else:
        return addr_str
        
        

ADDR_PAT = re.compile(r'\[.*\]')
def _normalize_op(op_str):
    # if it's a register, normalize it
    if op_str in reg_normalize:
        return reg_normalize[op_str]
    # if it's [reg+xxxh], normalize it
    if '[' in op_str and ']' in op_str:
        addr_str = ADDR_PAT.findall(op_str)[0]
        # remove "[", "]"
        addr_str_expr = addr_str[1:-1]
        evaluated_expr = _eval_addr_str(addr_str_expr)
        if type(evaluated_expr) == int:
            return "[%x]" % evaluated_expr
        else:
            return "[%s]" % evaluated_expr
    return op_str

    
    

def parse_insn_possible_op(insn):
    operator = None
    operand = None
    comments = None
    operand_strs = []
    operands = []
    insn_ori = insn
    insn = insn.strip()
    if insn.find(';') != -1:
        comments = insn[insn.find(';')+1:]
        insn = insn[:insn.find(';')]
    insn = insn.strip()
    if insn.find(' ') != -1:
        operator = insn[:insn.find(' ')]
        operand = insn[insn.find(' ')+1:]
    else:
        operator = insn
    
    if operand is not None:
        operand_strs = operand.split(',')  
        for op in operand_strs:
            op = op.strip()
            op_parsed = _normalize_op(op)
            operands.append(op_parsed)
    
    ret = []
    ret.append(operator)
    for op, op_str in zip(operands, operand_strs):
        ret.append((op_str, op))    
    return ret

                 
    
    pass