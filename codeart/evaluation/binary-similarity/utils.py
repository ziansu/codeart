import faiss
import json
import numpy as np
import os
import torch
from tqdm import tqdm
from typing import List, Union, Tuple, Optional, Dict

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

    def __init__(
        self,
        tokenizer,
        encoder,
        pooler='cls',
    ):
        self.tokenizer = tokenizer
        self.model = encoder
        self.pooler = pooler
        self.model = self.model.to('cuda').eval()

    def encode(
        self,
        binaries: Union[Dict, List[Dict]],
        device: str = None,
        batch_size=64,
        max_length=512,
        max_transitions=None,
        normalize=False,
        keepdim=False
    ) -> Union[np.ndarray, torch.Tensor]:

        target_device = self.device if device is None else device
        # self.model = self.model.to(target_device)
        # self.model.eval()

        single_binary = False
        if isinstance(binaries, Dict):
            binaries = [binaries]
            single_binary = True

        embedding_list = []
        with torch.no_grad():
            total_batch = len(binaries) // batch_size + \
                (1 if len(binaries) % batch_size > 0 else 0)

            for batch_id in tqdm(range(total_batch)):
                batch = binaries[batch_id *
                                 batch_size: (batch_id + 1) * batch_size]
                if "Rabert" in self.tokenizer.__class__.__name__:  # NOTE: Rabert
                    inputs = self.tokenizer.batch_inst_encode(
                        batch)
                else:
                    inputs = self.tokenizer.batch_inst_encode(
                        batch, max_transitions=max_transitions)
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                # pooling
                if self.pooler == 'cls':
                    embeddings = outputs.last_hidden_state[:, 0, :]
                elif self.pooler == 'pooler':
                    embeddings = outputs.pooler_output
                elif self.pooler == 'cls_before_pooler':
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                # normalize
                if normalize:
                    embeddings = embeddings / \
                        embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings)
        embeddings = torch.cat(embedding_list, dim=0).cpu()

        if single_binary and not keepdim:
            embeddings = embeddings[0]

        return embeddings.numpy()

    def encode_file(
        self,
        data_file: str,
        save_file: str,
        normalize_embed: bool,
        max_transitions: Optional[int] = None,
        pool_size: int = None
    ):
        # normalize_embed = kwargs.get("normalize_embed")

        with open(data_file, 'r') as f:
            dataset = [json.loads(line.strip()) for line in f.readlines()]
            if pool_size is not None:
                dataset = dataset[:pool_size]

        print(f'Number of binaries in {data_file}: {len(dataset)}')

        embeddings = self.encode(
            dataset,
            device='cuda',
            batch_size=128,
            max_length=512,
            max_transitions=max_transitions,
            normalize=normalize_embed,
        )

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        np.save(save_file, embeddings)

    @staticmethod
    def retrieve(
        source_embed,
        target_embed,
        source_id_map,
        target_id_map,
        top_k
    ):

        indexer = faiss.IndexFlatIP(target_embed.shape[1])
        indexer.add(target_embed)
        # print(f'source embedding shape: {source_embed.shape}, target embedding shape: {target_embed.shape}')
        D, I = indexer.search(source_embed, top_k)

        results = {}
        for source_idx, (dist, retrieved_index) in enumerate(zip(D, I)):
            source_id = source_id_map[source_idx]
            results[source_id] = {}
            retrieved_target_id = [target_id_map[x] for x in retrieved_index]
            results[source_id]['retrieved'] = retrieved_target_id
            results[source_id]['score'] = dist.tolist()

        return results

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
        print(
            f'source embedding shape: {source_embed.shape}, target embedding shape: {target_embed.shape}')
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


class BinaryRetrieverForGCBLike(BinaryRetriever):
    def encode(
        self,
        binaries: Union[Dict, List[Dict]],
        device: str = None,
        batch_size=64,
        max_length=512,
        max_transitions=None,
        normalize=False,
        keepdim=False
    ) -> Union[np.ndarray, torch.Tensor]:

        target_device = self.device if device is None else device
        # self.model = self.model.to(target_device)
        # self.model.eval()

        single_binary = False
        if isinstance(binaries, Dict):
            binaries = [binaries]
            single_binary = True

        # def preprocess_function(example):
        #     # example_entry = json.loads(example)

        #     example_entry = example
        #     # funcstr = " ".join(example_entry['code'])
        #     # join example_entry['code'][i][1], prepend <INST>
        #     funcstr = " ".join(
        #         [f"<INST> {x[1]}" for x in example_entry['code']])

        #     # Tokenize the texts
        #     tokenizer_ret = self.tokenizer(
        #         funcstr, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        #     # squeeze
        #     for k, v in tokenizer_ret.items():
        #         tokenizer_ret[k] = v.squeeze(0)
        #     return tokenizer_ret

        embedding_list = []
        with torch.no_grad():
            total_batch = len(binaries) // batch_size + \
                (1 if len(binaries) % batch_size > 0 else 0)

            for batch_id in tqdm(range(total_batch)):
                batch = binaries[batch_id *
                                 batch_size: (batch_id + 1) * batch_size]
                inputs = self.tokenizer.batch_inst_encode(batch)

                # instead of batch_inst_encode, we use preprocess_function
                # inputs = {}.fromkeys(
                #     ['input_ids', 'attention_mask', 'token_type_ids'])
                # for k in inputs.keys():
                #     inputs[k] = torch.stack(
                #         [preprocess_function(example)[k] for example in batch])

                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                # pooling
                if self.pooler == 'cls':
                    embeddings = outputs.last_hidden_state[:, 0, :]
                elif self.pooler == 'pooler':
                    embeddings = outputs.pooler_output
                elif self.pooler == 'cls_before_pooler':
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                # normalize
                if normalize:
                    embeddings = embeddings / \
                        embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings)
        embeddings = torch.cat(embedding_list, dim=0).cpu()

        if single_binary and not keepdim:
            embeddings = embeddings[0]
        return embeddings.numpy()


def dump_files(
    source_dataset_path,
    target_dataset_path,
    cache_dir,
    swap=False
):
    # load O0 dataset
    source_dataset = load_dataset(
        path=source_dataset_path,
        name=None,
        cache_dir=cache_dir,
        use_auth_token=True,
        streaming=False
    )
    # get test set
    # load O3 dataset (build func -> id mapping)
    func2id = {}
    with open(target_dataset_path, 'r') as f:
        target_dataset = [json.loads(line.strip()) for line in f.readlines()]
        for i, js in enumerate(target_dataset):
            func2id[(js['metadata']['project_name'],
                     js['metadata']['function_name'])] = i
    # get same test set (aligned by id in queries and pool)
    query_ids, queries = [], []
    pool_ids, pool = [], []
    success = 0
    for test_example in tqdm(source_dataset['test']):
        # format conversion
        test_example = {k: eval(v) for k, v in test_example.items()}
        try:
            func_id = (test_example['metadata']['project_name'],
                       test_example['metadata']['function_name'])
            target_id = func2id[func_id]
            query_ids.append(func_id)
            queries.append(target_dataset[target_id])
            pool_ids.append(func_id)
            pool.append(test_example)
            success += 1
        # FIXME: some functions are not found (sucess/total: 35979/48277)
        except KeyError:
            pass
    print(f"sucess/total: {success}/{len(source_dataset['test'])}")

    if swap:
        query_ids, pool_ids = pool_ids, query_ids
        queries, pool = pool, queries

        # print(query_ids[0])
        # print(pool_ids[0])
        # print(queries[0])
        # print(pool[0])

    # dump queries and pool
    oracle_file = {}    # NOTE: not necessay right now, modify under strict evaluation
    with open(os.path.join(cache_dir, 'binary_clone_detection', 'query.id'), 'w') as f:
        for query_id in query_ids:
            f.write(json.dumps(query_id) + '\n')
    with open(os.path.join(cache_dir, 'binary_clone_detection', 'query.jsonl'), 'w') as f:
        for query in queries:
            f.write(json.dumps(query) + '\n')
    with open(os.path.join(cache_dir, 'binary_clone_detection', 'pool.id'), 'w') as f:
        for pool_id in pool_ids:
            f.write(json.dumps(pool_id) + '\n')
    with open(os.path.join(cache_dir, 'binary_clone_detection', 'pool.jsonl'), 'w') as f:
        for pool_binary in pool:
            f.write(json.dumps(pool_binary) + '\n')


def dump_files_jsonl(
    source_dataset_path,
    target_dataset_path,
    cache_dir,
    swap=False,
):
    func2id = {}

    with open(source_dataset_path, 'r') as f:
        source_dataset = [json.loads(line.strip()) for line in f.readlines()]
        for i, js in enumerate(source_dataset):
            func2id[(js['metadata']['project_name'],
                     js['metadata']['function_name'])] = i

    with open(target_dataset_path, 'r') as f:
        target_dataset = [json.loads(line.strip()) for line in f.readlines()]

    query_ids, queries = [], []
    pool_ids, pool = [], []
    success = 0
    for example in target_dataset:
        func_id = (example['metadata']['project_name'],
                   example['metadata']['function_name'])
        try:
            source_id = func2id[func_id]
            query_ids.append(func_id)
            queries.append(source_dataset[source_id])
            pool_ids.append(func_id)
            pool.append(example)
            success += 1
        except:
            pass
    print(
        f"success/source/target: {success}/{len(source_dataset)}/{len(target_dataset)}")

    if swap:
        query_ids, pool_ids = pool_ids, query_ids
        queries, pool = pool, queries

    # dump queries and pool
    oracle_file = {}    # NOTE: not necessay right now, modify under strict evaluation
    with open(os.path.join(cache_dir, 'binary_clone_detection', 'query.id'), 'w') as f:
        for query_id in query_ids:
            f.write(json.dumps(query_id) + '\n')
    with open(os.path.join(cache_dir, 'binary_clone_detection', 'query.jsonl'), 'w') as f:
        for query in queries:
            f.write(json.dumps(query) + '\n')
    with open(os.path.join(cache_dir, 'binary_clone_detection', 'pool.id'), 'w') as f:
        for pool_id in pool_ids:
            f.write(json.dumps(pool_id) + '\n')
    with open(os.path.join(cache_dir, 'binary_clone_detection', 'pool.jsonl'), 'w') as f:
        for pool_binary in pool:
            f.write(json.dumps(pool_binary) + '\n')


if __name__ == '__main__':
    import sys
    sys.path.append('../../code/')
    # dump_files(
    #     'sheepy928/binkit-O0-raw',
    #     '../../data/binkit-O3.jsonl',
    #     '../../save/.cache',
    #     swap=True
    # )
    dump_files_jsonl(
        '../../data/coreutils.gcc.O3.jsonl',
        '../../data/coreutils.clang.O0.jsonl',
        cache_dir='../../save/.cache',
        swap=False
    )
