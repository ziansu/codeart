import argparse
from elftools.elf.elffile import ELFFile
import sys
from tqdm import tqdm
import die_globals
import base_calculator
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fin", type=str, default="type-inference/diffutils-3.4-O2/diff"
    )
    parser.add_argument("--fout", type=str, default="")

    args = parser.parse_args()
    return args


def collect_dies(die):
    die_globals.all_dies[die.offset] = die
    for child in die.iter_children():
        collect_dies(child)


def parse_die(die):
    if (
        die.tag == "DW_TAG_subprogram"
        or die.tag == "DW_TAG_inlined_subroutine"
        or die.tag == "DW_TAG_lexical_block"
    ):
        sub_prog_range_list = []
        if "DW_AT_low_pc" in die.attributes:
            low_pc = die.attributes["DW_AT_low_pc"].value
            high_pc = die.attributes["DW_AT_high_pc"].value
            if die.attributes["DW_AT_high_pc"].form.startswith("DW_FORM_addr"):
                pass
            elif die.attributes["DW_AT_high_pc"].form.startswith("DW_FORM_data"):
                high_pc += low_pc
            else:
                raise Exception("Invalid DW_AT_high_pc form")
            sub_prog_range_list.append((low_pc, high_pc))
        elif "DW_AT_ranges" in die.attributes:
            offset = die.attributes["DW_AT_ranges"].value
            ranges = die_globals.range_list.get_range_list_at_offset(offset)
            for entry in ranges:
                low_pc = die_globals.resolve_code_ofs(entry.begin_offset)
                high_pc = die_globals.resolve_code_ofs(entry.end_offset)
                sub_prog_range_list.append((low_pc, high_pc))
        if len(sub_prog_range_list) > 0:
            # for entry in sub_prog_range_list:
            #     print("[%x, %x)" % (entry[0], entry[1]))
            # all_structure_list = []
            all_type_list = []
            to_visit = [(die, sub_prog_range_list)]
            while to_visit:
                entry = to_visit.pop()
                current_die = entry[0]
                my_range_list = entry[1]
                for child in current_die.iter_children():
                    if (
                        child.tag == "DW_TAG_variable"
                        or child.tag == "DW_TAG_formal_parameter"
                    ):
                        # if base_calculator.complex_data_type(child):
                        base_list = base_calculator.calculate_base_addr(
                            child, my_range_list
                        )
                        all_type_list.extend(base_list)
                    elif (
                        child.tag == "DW_TAG_subprogram"
                        or child.tag == "DW_TAG_inlined_subroutine"
                        or child.tag == "DW_TAG_lexical_block"
                    ):
                        types_in_child = parse_die(child)
                        all_type_list.extend(types_in_child)
                    else:
                        to_visit.append((child, my_range_list))

            all_type_list_pretty = []
            if "DW_AT_frame_base" in die.attributes:
                frame_base = die.attributes["DW_AT_frame_base"].value
                fbase = base_calculator.parse_exprloc(frame_base)
                if fbase[0] == "cfa":
                    # resolve cfa
                    my_addr = sub_prog_range_list[0][0]
                    if my_addr not in die_globals.addr2cfi_entries:
                        raise Exception("CFI entry not found")
                    fbase_str = die_globals.addr2cfi_entries[my_addr]
                else:
                    fbase_str = fbase[0]

                for entry in all_type_list:
                    if "fbreg" in entry[4][0]:
                        pretty_fbreg = entry[4][0].replace(
                            "fbreg", "fbreg(%s)" % fbase_str
                        )
                        pretty_loc = (pretty_fbreg, entry[4][1])
                        all_type_list_pretty.append(
                            (entry[0], entry[1], entry[2], entry[3], pretty_loc)
                        )
                    else:
                        all_type_list_pretty.append(entry)
            else:
                all_type_list_pretty = all_type_list

            
            # for entry in all_type_list_pretty:
            #     print(
            #         "%s, %s, [%x, %x), %s"
            #         % (entry[0], entry[1], entry[2], entry[3], entry[4])
            #     )
            return all_type_list_pretty
        else:
            # print("No range list for %s" % die)
            # function declarations do not have range list
            return []
    else:
        ret = []
        if die.has_children:
            for child in die.iter_children():
                types_in_child = parse_die(child)
                ret.extend(types_in_child)
        return ret


def main():
    args = parse_args()
    fin = open(args.fin, "rb")
    elf = ELFFile(fin)
    dwarf = elf.get_dwarf_info()
    die_globals.range_list = dwarf.range_lists()
    die_globals.location_list = dwarf.location_lists()
    eh_cfi = dwarf.EH_CFI_entries()
    die_globals.parse_cfi_entires(eh_cfi)

    types_all = []
    types_all_pretty = []
    for cu in dwarf.iter_CUs():
        die_globals.gof = cu.cu_offset
        die = cu.get_top_DIE()
        if "DW_AT_low_pc" not in die.attributes:
            continue
        die_globals.cu_code_base_addr = die.attributes["DW_AT_low_pc"].value
        collect_dies(die)
        type_info = parse_die(die)
        type_info_pretty = []
        for info in type_info:
            pretty_info = (info[0], info[1], "%x" % info[2], "%x" % info[3], info[4])
            type_info_pretty.append(pretty_info)
        types_all.extend(type_info)
        types_all_pretty.extend(type_info_pretty)

    if args.fout == '':
        args.fout = args.fin + ".type_info.jsonl"
    with open(args.fout, "w") as fout:
        for info in types_all_pretty:
            if info[1] is None:
                continue
            entry = {
                'varname': info[0],
                'type': info[1],
                'low_pc': info[2],
                'high_pc': info[3],
                'loc': info[4]
            }
            fout.write(json.dumps(entry) + "\n")

    print(base_calculator.skip_cnt)


if __name__ == "__main__":
    main()
