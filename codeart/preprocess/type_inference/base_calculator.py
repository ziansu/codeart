from elftools.elf.elffile import ELFFile
import die_globals
from die_globals import resolve_ofs

skip_cnt = {}


def solve_type(die):
    tag = die.tag
    if tag == 'DW_TAG_base_type':
        base_type_attr = die.attributes['DW_AT_name'].value.decode('utf-8')
        return "base(%s)" % base_type_attr
    elif tag == 'DW_TAG_const_type':
        if 'DW_AT_type' not in die.attributes:
            return "void"
        base = solve_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
        return base
    elif tag == 'DW_TAG_packed_type':
        base = solve_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
        return base
    elif tag == 'DW_TAG_pointer_type':
        if 'DW_AT_type' not in die.attributes:
            return "void*"
        base = solve_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
        return "%s*" % base
    elif tag == 'DW_TAG_reference_type':
        raise NotImplementedError
        # return "pointer"
    elif tag == 'DW_TAG_restrict_type':
        base = solve_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
        return base
    elif tag == 'DW_TAG_rvalue_reference_type':
        base = solve_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
        return base
    elif tag == 'DW_TAG_shared_type':
        base = solve_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
        return base
    elif tag == 'DW_TAG_volatile_type':
        base = solve_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
        return base
    elif tag == 'DW_TAG_typedef':
        if 'DW_AT_type' in die.attributes:
            base = solve_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
            return base
        else:
            if 'type-unk-typedef' not in skip_cnt:
                skip_cnt['type-unk-typedef'] = 0
            skip_cnt['type-unk-typedef'] += 1
            return None
    elif tag == 'DW_TAG_array_type':
        return "array"
    elif tag == 'DW_TAG_structure_type':
        return "struct"
    elif tag == 'DW_TAG_union_type':
        return "union"
    elif tag == 'DW_TAG_class_type':
        return "class"
    elif tag == 'DW_TAG_enumeration_type':
        return "enum"
    elif tag == 'DW_TAG_subroutine_type':
        return "subroutine"
    elif tag == 'DW_TAG_ptr_to_member_type':
        return "ptr_to_member"
    elif tag == 'DW_TAG_unspecified_type':
        return "unspecified"
    else:
        print(tag)
        raise NotImplementedError


def _complex_data_type(die):
    def type_err(err_tag):
        print("Type Error in _complex_data_type")
        print(err_tag)
        raise 233333

    tag = die.tag
    if tag == 'DW_TAG_base_type':
        return False
    elif tag == 'DW_TAG_const_type':
        if 'DW_AT_type' not in die.attributes:
            return False
        base = _complex_data_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
        return base
    elif tag == 'DW_TAG_packed_type':
        type_err(tag)
    elif tag == 'DW_TAG_pointer_type':
        return False
        # if 'DW_AT_type' not in die.attributes:
        #     return False
        # base = _complex_data_type(all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
        # return base
    elif tag == 'DW_TAG_reference_type':
        base = _complex_data_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
        return base
    elif tag == 'DW_TAG_restrict_type':
        base = _complex_data_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
        return base
    elif tag == 'DW_TAG_rvalue_reference_type':
        base = _complex_data_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
        return base
    elif tag == 'DW_TAG_shared_type':
        base = _complex_data_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
        return base
    elif tag == 'DW_TAG_volatile_type':
        base = _complex_data_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
        return base
    elif tag == 'DW_TAG_typedef':
        if 'DW_AT_type' in die.attributes:
            base = _complex_data_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])
            return base
        else:
            type_err(tag)
    elif tag == 'DW_TAG_array_type':
        return True
    elif tag == 'DW_TAG_structure_type':
        return True
    elif tag == 'DW_TAG_union_type':
        return True
    elif tag == 'DW_TAG_class_type':
        # TODO
        # return True
        type_err(tag)
    elif tag == 'DW_TAG_enumeration_type':
        return False
    elif tag == 'DW_TAG_subroutine_type':
        return False
    elif tag == 'DW_TAG_ptr_to_member_type':
        type_err(tag)
    elif tag == 'DW_TAG_unspecified_type':
        type_err(tag)

    else:
        type_err(tag)


def complex_data_type(die):
    if 'DW_AT_type' not in die.attributes:
        return False
    return _complex_data_type(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)])


def _calculate_base_addr(die, offset, base_list):
    def type_err(err_tag):
        print("Type Error in _calculate_base_addr")
        print(err_tag)
        raise 233333

    tag = die.tag
    if tag == 'DW_TAG_base_type':
        return base_list
    elif tag == 'DW_TAG_const_type':
        if 'DW_AT_type' not in die.attributes:
            return base_list
        base = _calculate_base_addr(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)],
                                    offset, base_list)
        return base
    elif tag == 'DW_TAG_packed_type':
        type_err(tag)
    elif tag == 'DW_TAG_pointer_type':
        # TODO
        return base_list
    elif tag == 'DW_TAG_reference_type':
        return base_list
    elif tag == 'DW_TAG_restrict_type':
        base = _calculate_base_addr(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)],
                                    offset, base_list)
        return base
    elif tag == 'DW_TAG_rvalue_reference_type':
        return base_list
    elif tag == 'DW_TAG_shared_type':
        type_err(tag)
    elif tag == 'DW_TAG_volatile_type':
        base = _calculate_base_addr(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)],
                                    offset, base_list)
        return base
    elif tag == 'DW_TAG_typedef':
        if 'DW_AT_type' in die.attributes:
            base = _calculate_base_addr(die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)],
                                        offset, base_list)
            return base
        else:
            type_err(tag)
    elif tag == 'DW_TAG_array_type':
        base_list.append(offset)
        return base_list
    elif tag == 'DW_TAG_structure_type':
        base_list.append(offset)
        for child in die.iter_children():
            if child.tag == 'DW_TAG_member':
                location = child.attributes['DW_AT_data_member_location'].value
                type_tag = die_globals.all_dies[resolve_ofs(child.attributes['DW_AT_type'].value)]
                base_list = _calculate_base_addr(type_tag, offset + location, base_list)
        return base_list
    elif tag == 'DW_TAG_union_type':
        for child in die.iter_children():
            if child.tag == 'DW_TAG_member':
                type_tag = die_globals.all_dies[resolve_ofs(child.attributes['DW_AT_type'].value)]
                base_list.append(offset)
                base_list = _calculate_base_addr(type_tag, offset, base_list)
        return base_list
    elif tag == 'DW_TAG_class_type':
        # TODO
        # return True
        type_err(tag)
    elif tag == 'DW_TAG_enumeration_type':
        return base_list
    elif tag == 'DW_TAG_subroutine_type':
        type_err(tag)
    elif tag == 'DW_TAG_ptr_to_member_type':
        type_err(tag)
    elif tag == 'DW_TAG_unspecified_type':
        type_err(tag)

    else:
        type_err(tag)


def parse_LEB128(encoding, sign):
    length = 0
    for byte in encoding:
        length += 1
        if (byte & 0x80) == 0:
            break        
    i = 0
    current = encoding[i]
    MSB = encoding[length - 1]
    negative = (MSB & 0x40) != 0
    negative = negative and sign
    if negative:
        ret = -1
    else:
        ret = 0

    while True:
        mask = 0x7f << (i * 7)
        mask = ~mask
        ret &= mask
        has_next = (current & 0x80) != 0
        current = current & 0x7f
        current <<= i * 7
        ret = ret | current
        if not has_next:
            break
        i += 1
        current = encoding[i]

    return (ret, length)


def parse_SLEB128(encoding):
    return parse_LEB128(encoding, True)


def parse_ULEB128(encoding):
    return parse_LEB128(encoding, False)


reg_pretty_name = {
    0: 'RAX',
    1: 'RDX',
    2: 'RCX',
    3: 'RBX',
    4: 'RSI',
    5: 'RDI',
    6: 'RBP',
    7: 'RSP',
    8: 'R8',
    9: 'R9',
    10: 'R10',
    11: 'R11',
    12: 'R12',
    13: 'R13',
    14: 'R14',
    15: 'R15',
    16: 'RIP',
    17: 'XMM0',
    18: 'XMM1',
    19: 'XMM2',
    20: 'XMM3',
    21: 'XMM4',
    22: 'XMM5',
    23: 'XMM6',
    24: 'XMM7',    
}

def parse_exprloc(encoding):
    return _parse_location(encoding)

DW_LOC_EXPR_OPCODES = {
    'DW_OP_piece': 0x93,
    'DW_OP_stack_value': 0x9f,
    'DW_OP_call_frame_cfa': 0x9c,
    'DW_OP_deref': 0x6,
    'DW_OP_regx': 0x90,
}

regx_pretty_name = {
    33: 'ST0',
    34: 'ST1',
    35: 'ST2',
    36: 'ST3',
    37: 'ST4',
    38: 'ST5',
    39: 'ST6',
    40: 'ST7',
}

def _parse_location(location_encoding):
    opcode = location_encoding[0]
    if opcode == 0x91:
        offset, length = parse_SLEB128(location_encoding[1:])
        if length + 1 < len(location_encoding):
            if location_encoding[-1] == DW_LOC_EXPR_OPCODES['DW_OP_stack_value']:
                # DW_OP_stack_value
                return None
            elif location_encoding[length + 1] == DW_LOC_EXPR_OPCODES['DW_OP_piece']:
                # DW_OP_piece
                return None
            elif location_encoding[length + 1] == DW_LOC_EXPR_OPCODES['DW_OP_deref']:
                # DW_OP_deref
                return None            
            else:
                skip_reason = 'fbreg-unknown'
                if skip_reason not in skip_cnt:
                    skip_cnt[skip_reason] = 0
                skip_cnt[skip_reason] += 1
                return None
        return 'fbreg', offset
    elif opcode <= 0x11:
        # constants
        return 'constant', 0
    elif 0x30 <= opcode <= 0x4f:
        # literal
        return 'literal', 0
    elif 0x50 <= opcode <= 0x6f:        
        reg_num = opcode - 0x50
        reg_num_pretty = reg_pretty_name[reg_num]
        if len(location_encoding) > 1:    
            if location_encoding[1] == DW_LOC_EXPR_OPCODES['DW_OP_piece']:
                if 'reg-piece' not in skip_cnt:
                    skip_cnt['reg-piece'] = 0
                skip_cnt['reg-piece'] += 1
                # DW_OP_piece
                return None    
            raise TypeError("Unimplemented")
        return 'reg%d_%s'%(reg_num, reg_num_pretty), 0

    elif 0x70 <= opcode <= 0x8f:
        reg_num = opcode - 0x70
        reg_num_pretty = reg_pretty_name[reg_num]
        offset, length = parse_SLEB128(location_encoding[1:])
        if length + 1 < len(location_encoding):
            if location_encoding[-1] == DW_LOC_EXPR_OPCODES['DW_OP_stack_value']:
                # DW_OP_stack_value
                if 'breg-stack-value' not in skip_cnt:
                    skip_cnt['breg-stack-value'] = 0
                skip_cnt['breg-stack-value'] += 1
                return None
            elif location_encoding[length + 1] == DW_LOC_EXPR_OPCODES['DW_OP_piece']:
                # DW_OP_piece
                if 'breg-piece' not in skip_cnt:
                    skip_cnt['breg-piece'] = 0
                skip_cnt['breg-piece'] += 1                
                return None
            else:
                if 'breg-complex' not in skip_cnt:
                    skip_cnt['breg-complex'] = 0
                skip_cnt['breg-complex'] += 1
                return None
        return 'breg%d_%s'%(reg_num, reg_num_pretty), offset
    # elif opcode == 0x92:
    #     # bregx
    #     pass
    elif 0xe0 <= opcode <= 0xff:
        return None
    elif opcode in [0x9e, 0x93]:
        # implicit value, pieces
        return None    
    elif opcode == DW_LOC_EXPR_OPCODES['DW_OP_call_frame_cfa']:
        return 'cfa', 0
    elif opcode == DW_LOC_EXPR_OPCODES['DW_OP_regx']:
        reg_num, length = parse_ULEB128(location_encoding[1:])
        reg_num_pretty = regx_pretty_name[reg_num]
        if len(location_encoding) > length + 1:   
            if location_encoding[length + 1] == DW_LOC_EXPR_OPCODES['DW_OP_piece']:
                # DW_OP_piece
                return None                     
            raise TypeError("Unimplemented")
        return 'reg%d_%s'%(reg_num, reg_num_pretty), 0
    else:
        raise TypeError("Unimplemented")


def add_offset(location_tuple, ofs):
    loc_base = location_tuple[0]
    if loc_base == 'fbreg':
        return location_tuple[0], location_tuple[1] + ofs
    elif loc_base == 'breg':
        return location_tuple[0], location_tuple[1] + ofs
    else:
        
        return location_tuple[0], location_tuple[1]


def calculate_base_addr(die, sub_prog_range_list):
    if 'DW_AT_location' not in die.attributes:
        return []
    
    if 'DW_AT_abstract_origin' in die.attributes:
        origin_ofs_raw = die.attributes['DW_AT_abstract_origin'].value
        origin_ofs = die_globals.resolve_ofs(origin_ofs_raw)
        original_def = die_globals.all_dies[origin_ofs]
        var_name = original_def.attributes['DW_AT_name'].value.decode('utf-8')
        type_entry = die_globals.all_dies[resolve_ofs(original_def.attributes['DW_AT_type'].value)]
        type_info = solve_type(type_entry)
    elif 'DW_AT_name' not in die.attributes:
        if 'name-unknown' not in skip_cnt:
            skip_cnt['name-unknown'] = 0
        skip_cnt['name-unknown'] += 1
        return []
    else:
        var_name = die.attributes['DW_AT_name'].value.decode('utf-8')
        type_entry = die_globals.all_dies[resolve_ofs(die.attributes['DW_AT_type'].value)]
        type_info = solve_type(type_entry)    
    base_location = die.attributes['DW_AT_location']
    locations = []
    if base_location.form == 'DW_FORM_sec_offset':
        locs = base_location.value
        locs = die_globals.location_list.get_location_list_at_offset(locs)
        for loc in locs:
            loc_expr = _parse_location(loc.loc_expr)
            if loc_expr is None:
                continue
            addr_start = die_globals.resolve_code_ofs(loc.begin_offset)
            addr_end = die_globals.resolve_code_ofs(loc.end_offset)
            locations.append((addr_start, addr_end, loc_expr))
    else:
        loc_expr = _parse_location(base_location.value)
        if loc_expr is None:
            return []
        for addr_start, addr_end in sub_prog_range_list:
            locations.append((addr_start, addr_end, loc_expr))        
    ret = []
    for addr_start, addr_end, location in locations:
        ret.append((var_name, type_info, addr_start, addr_end, location))
    return ret
