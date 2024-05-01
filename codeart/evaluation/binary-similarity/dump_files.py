import faiss
import json
import numpy as np
import os
import torch
from tqdm import tqdm
from typing import List, Union, Tuple, Optional, Dict

from datasets import load_dataset

def dump_files_jsonl(
    source_dataset_path, 
    target_dataset_path,
    output_prefix,
    swap=False,
):
    func2id = {}

    with open(source_dataset_path, 'r') as f:
        source_dataset = [json.loads(line.strip()) for line in f.readlines()]
        for i, js in enumerate(source_dataset):
            func2id[(js['metadata']['project_name'], js['metadata']['function_name'])] = i

    with open(target_dataset_path, 'r') as f:
        target_dataset = [json.loads(line.strip()) for line in f.readlines()]

    query_ids, queries = [], []
    pool_ids, pool = [], []
    success = 0
    for example in target_dataset:
        func_id = (example['metadata']['project_name'], example['metadata']['function_name'])
        try:
            source_id = func2id[func_id]
            query_ids.append(func_id)
            queries.append(source_dataset[source_id])
            pool_ids.append(func_id)
            pool.append(example)
            success += 1
        except:
            pass
    print(f"success/source/target: {success}/{len(source_dataset)}/{len(target_dataset)}")

    if swap:
        query_ids, pool_ids = pool_ids, query_ids
        queries, pool = pool, queries

    # dump queries and pool
    oracle_file = {}    # NOTE: not necessay right now, modify under strict evaluation
    with open("%s-query.id"%output_prefix, 'w') as f:        
        for query_id in query_ids:
            f.write(json.dumps(query_id) + '\n')
    with open("%s-query.jsonl"%output_prefix, 'w') as f:
        for query in queries:
            f.write(json.dumps(query) + '\n')
    with open("%s-pool.id"%output_prefix, 'w') as f:
        for pool_id in pool_ids:
            f.write(json.dumps(pool_id) + '\n')
    with open("%s-pool.jsonl"%output_prefix, 'w') as f:
        for pool_binary in pool:
            f.write(json.dumps(pool_binary) + '\n')


if __name__ == '__main__':
    import sys
    sys.path.append('../../code/')
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_dataset_path', type=str, default='binsim-dataset/coreutilsh-O0.jsonl',
        help='source dataset path'
    )
    parser.add_argument(
        '--tgt_dataset_path', type=str, default='binsim-dataset/coreutilsh-O3.jsonl',
        help='target dataset path'
    )
    parser.add_argument(
        '--output_prefix', type=str, default='cache/binary_clone_detection/coreutilsh',
        help='output prefix'
    )

    args = parser.parse_args()

    dump_files_jsonl(
        source_dataset_path=args.src_dataset_path,
        target_dataset_path=args.tgt_dataset_path,
        output_prefix=args.output_prefix        
    )