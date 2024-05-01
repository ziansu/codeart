import faiss
import json
import numpy as np
import os
import torch
from tqdm import tqdm
from typing import List, Union, Tuple, Optional, Dict
from eval import eval_from_dict, TOP_K
from datasets import load_dataset


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return str(self.avg)


class BinaryRetriever(object):

    @staticmethod
    def retrieve_from_file_random(
        source_embed_file, 
        target_embed_file,
        source_id_file,
        target_id_file,
        pool_size,
        top_k,
        seed=42,
        round_num=5
    ):
        with open(source_id_file, 'r') as f:
            source_id_map = {}
            source_func_id2idx = {}
            for idx, line in enumerate(f.readlines()):
                source_id_map[idx] = line.strip()
                source_func_id2idx[line.strip()] = idx

        with open(target_id_file, 'r') as f:
            target_id_map = {}
            target_func_id2idx = {}
            for idx, line in enumerate(f.readlines()):
                target_id_map[idx] = line.strip()
                target_func_id2idx[line.strip()] = idx
        
        overlapped_funcs = set(source_func_id2idx.keys()) & set(target_func_id2idx.keys())
        print(f'Number of overlapped functions: {len(overlapped_funcs)}')
        source_embed_all = np.load(source_embed_file + '.npy')
        target_embed_all = np.load(target_embed_file + '.npy')
        # random sample
        np.random.seed(seed)
        pool_size = min(pool_size, len(overlapped_funcs))
        
        overlapped_funcs_list = sorted(list(overlapped_funcs))

        results_all = {
            'recall': {k: AverageMeter() for k in TOP_K},
            'mrr': AverageMeter()
        }
        for rnd in range(round_num):
            selected_overlapped_funcs = np.random.choice(overlapped_funcs_list, pool_size, replace=False)
            print(f'Number of selected overlapped functions: {len(selected_overlapped_funcs)}')
            selected_indices = [source_func_id2idx[x] for x in selected_overlapped_funcs]


            source_embed = source_embed_all[selected_indices]
            target_embed = target_embed_all[selected_indices]

            print(f'source embedding shape: {source_embed.shape}, target embedding shape: {target_embed.shape}')
            indexer = faiss.IndexFlatIP(target_embed.shape[1])
            indexer.add(target_embed)
            D, I = indexer.search(source_embed, top_k)

            results = {}
            for source_idx, (dist, retrieved_index) in enumerate(zip(D, I)):
                source_id = source_id_map[source_idx]
                results[source_id] = {}
                retrieved_target_id = [target_id_map[x] for x in retrieved_index]
                results[source_id]['retrieved'] = retrieved_target_id
                results[source_id]['score'] = dist.tolist()
            ret = eval_from_dict(results)
            results_all['mrr'].update(ret['mrr'])
            for k in TOP_K:
                results_all['recall'][k].update(ret['recall'][k])
        
        return results_all

        

    @staticmethod
    def retrieve_from_file(
        source_embed_file, 
        target_embed_file,
        source_id_file,
        target_id_file,
        pool_size,
        top_k, 
        save_file,
    ):
        with open(source_id_file, 'r') as f:
            source_id_map = {}
            for idx, line in enumerate(f.readlines()[:pool_size]):
                source_id_map[idx] = line.strip()

        with open(target_id_file, 'r') as f:
            target_id_map = {}
            for idx, line in enumerate(f.readlines()[:pool_size]):
                target_id_map[idx] = line.strip()
        
        source_embed = np.load(source_embed_file + '.npy')
        target_embed = np.load(target_embed_file + '.npy')
        assert (len(source_id_map) == source_embed.shape[0])
        assert (len(target_id_map) == target_embed.shape[0])
        indexer = faiss.IndexFlatIP(target_embed.shape[1])
        indexer.add(target_embed)
        print(f'source embedding shape: {source_embed.shape}, target embedding shape: {target_embed.shape}')
        D, I = indexer.search(source_embed, top_k)

        results = {}
        for source_idx, (dist, retrieved_index) in enumerate(zip(D, I)):
            source_id = source_id_map[source_idx]
            results[source_id] = {}
            retrieved_target_id = [target_id_map[x] for x in retrieved_index]
            results[source_id]['retrieved'] = retrieved_target_id
            results[source_id]['score'] = dist.tolist()

        with open(save_file, 'w+') as f:
            json.dump(results, f, indent=2)
        
        return results


if __name__ == '__main__':
    import sys
    sys.path.append('../../code/')
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_file', type=str, default='output/src_embedding',
        help='source embedding file'
    )
    parser.add_argument(
        '--target_file', type=str, default='output/tgt_embedding',
        help='target embedding file'    
    )
    parser.add_argument(
        '--source_id_file', type=str, default='cache/binary_clone_detection/query.id',
        help='source id file'
    )
    parser.add_argument(
        '--target_id_file', type=str, default='cache/binary_clone_detection/pool.id',
        help='target id file'
    )
    parser.add_argument(
        '--pool_size', type=int, default=100,
        help='pool size'
    )
    parser.add_argument(
        '--top_k', type=int, default=10,
        help='top k'
    )
    parser.add_argument(
        '--save_file', type=str, default='output/retrieval_results.json',
        help='save file'
    )

    args = parser.parse_args()

    # dump_files(
    #     'sheepy928/binkit-O0-raw',
    #     '../../data/binkit-O3.jsonl',
    #     '../../save/.cache',
    #     swap=True
    # )
    # dump_files_jsonl(
    #     '../../data/coreutils.gcc.O3.jsonl',
    #     '../../data/coreutils.clang.O0.jsonl',
    #     cache_dir='../../save/.cache',
    #     swap=False
    # )
    ret = BinaryRetriever.retrieve_from_file_random(
        source_embed_file=args.source_file,
        target_embed_file=args.target_file,
        source_id_file=args.source_id_file,
        target_id_file=args.target_id_file,
        pool_size=args.pool_size,
        top_k=args.top_k        
    )
    
    print(ret)
    print("Final-PR@1: ", ret['recall'][1])
    print("Final-MRR: ", ret['mrr'])

