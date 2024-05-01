import pickle
import os
import argparse
import json
import glob
from utils.data_utils import dump_cfg, parse_cfg
from analysis import prog_model
import networkx as nx
from analysis.expr_lang_analyzer import ExprLangAnalyzer
from tqdm import tqdm
import signal

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', type=str, default='')    
    parser.add_argument('--fout', type=str, default='')    
    args = parser.parse_args()
    return args


def timeout_handler(signum, frame):
    raise Exception("timeout")

def main():
    args = parse_args()
    
    data_in = pickle.load(open(args.data_in, 'rb'))
    fout = open(args.fout, 'w')
    signal.signal(signal.SIGALRM, timeout_handler)
    for function in tqdm(data_in):
        meta = {
            'project_name': function['project_name'],
            'function_name': function['funcname'],
            'function_addr': function['funcaddr'],
            "binary_name": function['binname'],
        }
        # maximum analysis time is 10 seconds
        signal.alarm(10)
        try:
          expr_lang_analyzer = ExprLangAnalyzer(function['cfg'])
          signal.alarm(10)
          expr_lang_analyzer.print_func_to_jsonl(fout, metadata=meta)
        except Exception as e:
          # if e is not time out
          if str(e) != "timeout":
            print("Error in function: ")
            print(meta)          
            # raise e
          else:  
            print(meta)          
            print(e)

    fout.close()        
    print(args)



if __name__ == '__main__':
    main()
