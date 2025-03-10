all_dies = {}
gof = 0
range_list = None
location_list = None

import elftools


def resolve_ofs(ofs):
    return ofs + gof


cu_code_base_addr = 0


def resolve_code_ofs(ofs):
    return ofs + cu_code_base_addr


_dbg_cfi_entries = {}
addr2cfi_entries = {}


def parse_cfi_entires(cfi_entries):
    for entry in cfi_entries:
        if "header" not in dir(entry):
            continue
        if "initial_location" not in entry.header:
            continue
        addr = entry.header["initial_location"]
        cfi_table = entry._decode_CFI_table()
        _dbg_cfi_entries[addr] = cfi_table
        if len(cfi_table) != 2:
            raise Exception("CFI table length is not 2")
        current_cfa_entries = cfi_table[0]
        # the register ids mentioned in the CFI table
        # cols = cfi_table[1]
        has_rbp = False
        for cfa_entry in current_cfa_entries:
            cfa_info = cfa_entry["cfa"]
            if type(cfa_info) != elftools.dwarf.callframe.CFARule:
                raise Exception("CFI table is not CFARule")
            if cfa_info.reg == 6:
                has_rbp = True
                break

        if has_rbp:
            addr2cfi_entries[addr] = "rbp"
        else:
            addr2cfi_entries[addr] = "rsp"

        # heuristics to identify whether stack frame is rsp or rbp based
