import pickle
import os
from tqdm import tqdm

import argparse

ADDR_IDX = 0
ASM_IDX = 1
RAW_IDX = 2
CFG_IDX = 3


def get_args():
    parser = argparse.ArgumentParser(description="Collect the preprocess results")
    parser.add_argument(
        "--binary_list_file",
        type=str,
        default="",
        help="This file contains the file names to be loaded",
    )
    parser.add_argument(
        "--fout", type=str, default="", help="The output file name, in pickle format"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    MAX_LEN = 512
    args = get_args()
    print(args)
    # load picked binary
    fin = open(args.binary_list_file, "r")
    binary_list = fin.readlines()
    fin.close()
    binfolder_binary_entries = []
    all_binary_len = len(binary_list)
    print("Loading binaries ...")

    for b in tqdm(binary_list):
        project_name = os.path.basename(os.path.dirname(b))
        bin_fin = open(b.strip(), "rb")
        binary = pickle.load(bin_fin)
        bin_fin.close()
        addr2function = {}
        for name, entry in binary.items():
            addr2function[entry[0]] = entry

        for name, entry in binary.items():
            my_cfg = entry[CFG_IDX]
            # this is to support legacy code
            # there is only one version of CFG now
            new_cfg = my_cfg
            func_addr = entry[ADDR_IDX]
            new_cfg.nodes[func_addr]["num"] = -1
            logical_order = [(n, my_cfg.nodes[n]) for n in my_cfg.nodes()]

            binfolder_binary_entries.append(
                {
                    "project_name": project_name,
                    "funcname": name,
                    "binname": os.path.basename(b),
                    "funcaddr": entry[ADDR_IDX],
                    "cfg": new_cfg,
                    "dbg_logical_order": logical_order,
                }
            )

    fout = open(args.fout, "wb")
    pickle.dump(binfolder_binary_entries, fout)
    fout.close()
    exit(0)
