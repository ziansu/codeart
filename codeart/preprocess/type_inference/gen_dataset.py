import argparse
import os
import pickle
import json
import utils
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl-data-in', type=str, help='path to codeart jsonl data')
    parser.add_argument('--raw-data-in', type=str, help='path to the .pkl file corresponding to the jsonl data')
    parser.add_argument('--bin-dir-root', type=str, help='root dir containing the binaries')
    parser.add_argument('--fout', type=str, default='')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    raw_data_in = pickle.load(open(args.raw_data_in, 'rb'))
    jsonl_data_in = open(args.jsonl_data_in, 'r')
    name2raw = {}
    binname2debug_info = {}
    for function in raw_data_in:
        proj_name = function['project_name']
        binname = function['binname']
        func_addr = function['funcaddr']
        name2raw[(proj_name, binname, func_addr)] = function
    
    del proj_name
    del binname
    del func_addr
    del function
    if args.fout == '':
        args.fout = args.jsonl_data_in + '.type_data.jsonl'
    fout = open(args.fout, 'w')
    for line in tqdm(jsonl_data_in):
        data = json.loads(line)
        project_name = data['metadata']['project_name']        
        function_addr = data['metadata']['function_addr']
        binary_name = data['metadata']['binary_name']        
        if (project_name, binary_name, function_addr) not in name2raw:
            raise Exception("Function %s not found" % str((project_name, binary_name, function_addr)))
        
        normalized_bin_name = binary_name.strip().replace('.elf_extract.pkl', '')
        if (project_name, normalized_bin_name) not in binname2debug_info:
            # binname2debug_info[(proj_name, normalized_bin_name)] = None
            bin_path = os.path.join(args.bin_dir_root, project_name, normalized_bin_name + ".type_info.jsonl")            
            dbg_info_searcher = utils.BinaryDebugInfoSearcher(bin_path)
            binname2debug_info[(project_name, normalized_bin_name)] = dbg_info_searcher
        
        current_dbg_info_searcher = binname2debug_info[(project_name, normalized_bin_name)]
        marinda_insns = data['code']
        function_raw = name2raw[(project_name, binary_name, function_addr)]
        cfg = function_raw['cfg']
        sorted_node_ids = sorted(cfg.nodes)
        nodes = [cfg.nodes[node_id] for node_id in sorted_node_ids]
        raw_insn_list = []
        current_insns = 0
        for node in nodes:
            if len(node['asm']) != len(node['addr_list']):
                raise Exception("Length of asm and addr_list not equal")
            for asm, addr in zip(node['asm'], node['addr_list']):
                if current_insns == len(marinda_insns):
                    break
                raw_insn_list.append((current_insns, asm, addr))
                current_insns += 1

        instrs_w_type = []
        for i, marinda_insn in enumerate(marinda_insns):
            raw_insn = raw_insn_list[i]
            raw_insn_str = raw_insn[1]
            insn_addr = raw_insn[2]
            parse_ret = utils.parse_insn_possible_op(raw_insn_str)
            current_insn_w_type = []
            for item in parse_ret:
                if item is None:
                    continue
                if type(item) == str:
                    current_insn_w_type.append((item, None))
                else:
                    ties = current_dbg_info_searcher.query(insn_addr, item[1])
                    if ties is None:
                        current_insn_w_type.append((item[0], None))                    
                    else:
                        current_insn_w_type.append((item[0], ties[-1].type))
            
            instrs_w_type.append((marinda_insn, current_insn_w_type))
        to_print = {
            'metadata': data['metadata'],
            'code': marinda_insns,
            'data_dep': data['data_dep'],
            'code_w_type': [i[1] for i in instrs_w_type]
        }
        fout.write(json.dumps(to_print))
        fout.write('\n')
        

    fout.close()
    


if __name__ == '__main__':
    main()