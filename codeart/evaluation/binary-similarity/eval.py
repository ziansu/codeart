import argparse
import copy
from collections import OrderedDict
import json
import numpy as np
from typing import Dict


TOP_K = [1, 3, 5, 10]


def mrr(gold, pred):
    ranks = []
    for g, p in zip(gold, pred):
        try:
            r = p['retrieved'].index(g)
            ranks.append(1 / (r + 1))
        except ValueError:
            ranks.append(0)
    
    return np.mean(ranks)


def recall(gold, pred, top_k=None):
    top_k = TOP_K if top_k is None else top_k
    recall_n = {x: 0 for x in top_k}

    for g, p in zip(gold, pred):

        for k in top_k:
            candidates = p['retrieved'][: k]
            recall_n[k] += 1 if g in candidates else 0

    recall_n = {k: v / len(pred) for k, v in recall_n.items()}

    return recall_n


def eval_from_dict(results: Dict):
    
    gold, pred = [], []

    for k, v in results.items():
        gold.append(k)
        pred.append(v)
    
    metrics = {
        'recall': recall(gold, pred),
        'mrr': mrr(gold, pred)
    }

    return metrics


def eval_from_file(result_file):
    gold, pred = [], []

    with open(result_file, 'r') as f:
        results = json.load(f)

    for k, v in results.items():
        gold.append(k)
        pred.append(v)
    
    metrics = {
        'recall': recall(gold, pred),
        'mrr': mrr(gold, pred)
    }

    print(metrics)


if __name__ == '__main__':
    eval_from_file('../../save/.cache/binary_clone_detection/retrieval_results.json')