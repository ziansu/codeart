import argparse
import json

import pandas as pd
from datasets import Dataset, concatenate_datasets, Features
from tqdm import tqdm
import numpy as np
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--opt", type=str, default="O3")

args = parser.parse_args()

shards = glob.glob("dataset/*.type_data.jsonl")
OPT = args.opt
out_ds_name = "type-inference-all-%s" % OPT

print("OPT: %s" % OPT)

feature_dict = {
    "metadata": {
        "binary_name": {"dtype": "string", "_type": "Value"},
        "function_addr": {"dtype": "int64", "_type": "Value"},
        "function_name": {"dtype": "string", "_type": "Value"},
        "project_name": {"dtype": "string", "_type": "Value"},
    },
    "code_w_type": {"dtype": "string", "_type": "Value"},
    "code": {"dtype": "string", "_type": "Value"},
    "data_dep": {"dtype": "string", "_type": "Value"},
}
features = Features.from_dict(feature_dict)

valid_projs = set(
    [
        "coreutils-5.93-O0",
        "coreutils-5.93-O1",
        "coreutils-5.93-O2",
        "coreutils-5.93-O3",
        "coreutils-6.4-O0",
        "coreutils-6.4-O1",
        "coreutils-6.4-O2",
        "coreutils-6.4-O3",
        "coreutils-7.6-O0",
        "coreutils-7.6-O1",
        "coreutils-7.6-O2",
        "coreutils-7.6-O3",
        "coreutils-8.1-O0",
        "coreutils-8.1-O1",
        "coreutils-8.1-O2",
        "coreutils-8.1-O3",
        "coreutils-8.30-O0",
        "coreutils-8.30-O1",
        "coreutils-8.30-O2",
        "coreutils-8.30-O3",
        "diffutils-2.8-O0",
        "diffutils-2.8-O1",
        "diffutils-2.8-O2",
        "diffutils-2.8-O3",
        "diffutils-3.1-O0",
        "diffutils-3.1-O1",
        "diffutils-3.1-O2",
        "diffutils-3.1-O3",
        "diffutils-3.3-O0",
        "diffutils-3.3-O1",
        "diffutils-3.3-O2",
        "diffutils-3.3-O3",
        "diffutils-3.4-O0",
        "diffutils-3.4-O1",
        "diffutils-3.4-O2",
        "diffutils-3.4-O3",
        "findutils-4.233-O0",
        "findutils-4.233-O1",
        "findutils-4.233-O2",
        "findutils-4.233-O3",
        "findutils-4.41-O0",
        "findutils-4.41-O1",
        "findutils-4.41-O2",
        "findutils-4.41-O3",
        "findutils-4.6-O0",
        "findutils-4.6-O1",
        "findutils-4.6-O2",
        "findutils-4.6-O3",
    ]
)

valid_projs = set([x for x in valid_projs if x.endswith(OPT)])


def gen(shards):
    for shard in shards:
        with open(shard, "r") as f:
            for line in f:
                data = json.loads(line)
                proj_name = data["metadata"]["project_name"]
                # if proj_name not in valid_projs:
                #     continue
                if not proj_name.endswith(OPT):
                    continue
                code_w_type = []
                for types in data["code_w_type"]:
                    code_w_type.append([i[1] if i[1] else "noacc" for i in types])
                out_data_entry = {
                    "metadata": data["metadata"],
                    "code_w_type": str(code_w_type),
                    "code": str(data["code"]),
                    "data_dep": str(data["data_dep"]),
                }
                yield out_data_entry


ds = Dataset.from_generator(
    gen,
    features=features,
    gen_kwargs={"shards": shards},
    num_proc=8,
    cache_dir="./ds-cache",
)

# split train/valid/test
dataset = ds.shuffle(seed=42)
ret_data_dict = dataset.train_test_split(test_size=0.1, seed=42)
ret_data_dict.push_to_hub(out_ds_name + "-shuffle", private=True)

print()
