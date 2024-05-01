import json
import math
import numpy as np
import random
import re
import torch
from tqdm import tqdm
from typing import Any, Dict, List, Tuple


def custom_preprocessing(text):
    text = re.sub(r'(qword|dword|sub|byte|unk|sub|loc|byte)_(\w+)', r'\1 \2', text)
    return text


def my_load_dataset(path, limit=10):
    "assume jsonl of normalized dataset"
    with open(path, 'r') as f:
        dataset = [json.loads(line.strip()) for line in f.readlines()[:limit]]
    return dataset


class DataCollatorForMLMWithEdgePred(object):
    
    def __init__(
        self,
        tokenizer,
        mlm_probability: float,
        ep_probability: float,
        max_length: int,
        pad_to_multiple_of: int = None,
        max_transitions=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.ep_probability = ep_probability
        self.max_length = max_length
        self.max_transitions = max_transitions
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(
        self,
        examples,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
            'ep_bce_weights': [],
            'ep_labels': []
        }

        batch_size = len(examples)

        node_pos_all = []
       
        for example in examples:

            # need to eval when using `load_dataset` to load from hub
            example['code'] = eval(example['code'])
            example['data_dep'] = eval(example['data_dep'])

            encoded = self.tokenizer.inst_encode(
                example['code'],
                example['data_dep'],
                return_extra_info=True,
                max_transitions=self.max_transitions
            )
            
            input_ids, labels = self.mask_tokens(
                encoded['input_ids'], 
                encoded['special_tokens_mask']
            )
            ep_attention_mask, ep_bce_weights, ep_labels = \
                self.mask_graph_edges(
                    encoded['attention_mask'],
                    encoded['instruction_node_positions']
                )

            # add to batch
            batch['input_ids'].append(input_ids)
            batch['labels'].append(labels)
            batch['attention_mask'].append(ep_attention_mask)
            batch['ep_bce_weights'].append(ep_bce_weights)
            batch['ep_labels'].append(ep_labels)

            node_pos_all.append(encoded['instruction_node_positions'])

        return {
            # 'node_positions': node_pos_all,   # debug
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask']),
            'labels': torch.stack(batch['labels']),
            'ep_bce_weights': torch.stack(batch['ep_bce_weights']).float(),
            'ep_labels': torch.stack(batch['ep_labels']).float()
        }
    
    def mask_tokens(self, inputs, special_tokens_mask):
        """
        generate input_ids and labels for randomly masked input
        special tokens that should not be masked here are [CLS], [SEP], [PAD], and <INST>
        
        https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/data/data_collator.py#L751"
        """
        
        labels = inputs.clone()
         # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def mask_graph_edges(self, attention_mask, node_positions: List[int]):
        """
        Following GraphCodeBERT's edge prediction task:
        1. random sample nodes in the graph G = (V, E)
        2. mask direct edges connecting these sampled nodes by adding an infinitely 
           negative value in the mask matrix
        3. balance positive and negative predictions

        NOTE: 
        1. current edge masking will mask out node's self-attention
        2. `node_to_position`alreadly considered [CLS] at the front

        version 1:
            - dense computation
            - use weight 0 to mask out some edges
              https://discuss.pytorch.org/t/masking-binary-cross-entropy-loss/61065
        """
        
        # sample `ep_probability` of nodes
        sampled_nodes = random.sample(
                            node_positions, 
                            k=math.floor(self.ep_probability * len(node_positions)))
        
        if len(sampled_nodes) == 0:  # NOTE by zian: edge cases, e.g., there's only one instruction
            return attention_mask, \
                torch.zeros(attention_mask.shape, dtype=torch.bool), \
                torch.zeros(attention_mask.shape, dtype=torch.bool)

        # list edges

        ### cannot do non-successive indexesing in two dimensions
        # to_edges = attention_mask[sampled_nodes: node_positions]
        # from_edges = attention_mask[node_positions: sampled_nodes]

        ### iteratively processing with V_C
        # "self-edge" are not true edges but self-attention, remove
        adjacency_matrix = attention_mask.clone()
        torch.diagonal(adjacency_matrix, 0).zero_()

        to_edges = []
        from_edges = []
        for sid in sampled_nodes:
            to_edges.append(adjacency_matrix[sid, node_positions])
            from_edges.append(adjacency_matrix[node_positions, sid])
        to_edges = torch.stack(to_edges)
        from_edges = torch.stack(from_edges)

        # balanced sampling (in total 2 * V_C * V)
        # from and to edges will overlap in V_C * V_C
        n_positives = torch.sum(to_edges) + torch.sum(from_edges)
        n_negatives = 2 * len(sampled_nodes) * len(node_positions) - n_positives

        if n_positives > n_negatives:
            bernouli_p = n_negatives / n_positives
            to_edges_probability_matrix = torch.zeros(size=to_edges.shape, dtype=float)
            to_edges_probability_matrix.masked_fill_(to_edges, bernouli_p)
            to_edges_weight_matrix = torch.bernoulli(to_edges_probability_matrix).bool()
            to_edges_weight_matrix = torch.logical_or(to_edges_weight_matrix, ~to_edges)

            from_edges_probability_matrix = torch.zeros(size=from_edges.shape, dtype=float)
            from_edges_probability_matrix.masked_fill_(from_edges, bernouli_p)
            from_edges_weight_matrix = torch.bernoulli(from_edges_probability_matrix).bool()
            from_edges_weight_matrix = torch.logical_or(from_edges_weight_matrix, ~from_edges)

        else:
            bernouli_p = n_positives / n_negatives
            to_edges_probability_matrix = torch.zeros(size=to_edges.shape, dtype=float)
            to_edges_probability_matrix.masked_fill_(~to_edges, bernouli_p)
            to_edges_weight_matrix = torch.bernoulli(to_edges_probability_matrix).bool()
            to_edges_weight_matrix = torch.logical_or(to_edges_weight_matrix, to_edges)

            from_edges_probability_matrix = torch.zeros(size=from_edges.shape, dtype=float)
            from_edges_probability_matrix.masked_fill_(~from_edges, bernouli_p)
            from_edges_weight_matrix = torch.bernoulli(from_edges_probability_matrix).bool()
            from_edges_weight_matrix = torch.logical_or(from_edges_weight_matrix, from_edges)
            
        # print(f"bernouli probability: {bernouli_p}")

        ep_input_attention_mask = None
        ep_weights_for_BCE_loss = torch.zeros(attention_mask.shape, dtype=torch.bool)
        ep_labels = attention_mask
        for i, vc_id in enumerate(sampled_nodes):
            for j, v_id in enumerate(node_positions):
                ep_weights_for_BCE_loss[vc_id, v_id] = to_edges_weight_matrix[i, j]
                ep_weights_for_BCE_loss[v_id, vc_id] = from_edges_weight_matrix[i, j]   # to be tested
        ep_input_attention_mask = torch.logical_and(attention_mask, ~ep_weights_for_BCE_loss)
        ep_labels = torch.logical_and(attention_mask, ep_weights_for_BCE_loss)

        return ep_input_attention_mask, ep_weights_for_BCE_loss, ep_labels


class DataCollatorForCodeArt(DataCollatorForMLMWithEdgePred):
    
    def __init__(
        self,
        tokenizer,
        mlm_probability: float,
        ep_probability: float,
        max_length: int,
        pad_to_multiple_of: int = None,
        max_transitions=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            ep_probability=ep_probability,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            max_transitions=max_transitions
        )

    def __call__(
        self,
        examples,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'relative_position_matrix': [],
            'labels': [],
            'ep_bce_weights': [],
            'ep_labels': []
        }

        batch_size = len(examples)

        node_pos_all = []
       
        for example in examples:

            # need to eval when using `load_dataset` to load from hub
            example['code'] = eval(example['code'])
            example['data_dep'] = eval(example['data_dep'])

            encoded = self.tokenizer.inst_encode(
                example['code'],
                example['data_dep'],
                return_extra_info=True,
                max_transitions=self.max_transitions
            )
            
            input_ids, labels = self.mask_tokens(
                encoded['input_ids'], 
                encoded['special_tokens_mask']
            )
            ep_attention_mask, ep_bce_weights, ep_labels = \
                self.mask_graph_edges(
                    encoded['attention_mask'],
                    encoded['instruction_node_positions']
                )

            # add to batch
            batch['input_ids'].append(input_ids)
            batch['relative_position_matrix'].append(encoded['relative_position_matrix'])
            batch['labels'].append(labels)
            batch['attention_mask'].append(ep_attention_mask)
            batch['ep_bce_weights'].append(ep_bce_weights)
            batch['ep_labels'].append(ep_labels)

            node_pos_all.append(encoded['instruction_node_positions'])

        return {
            # 'node_positions': node_pos_all,   # debug
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask']),
            'relative_position_matrix': torch.stack(batch['relative_position_matrix']),
            'labels': torch.stack(batch['labels']),
            'ep_bce_weights': torch.stack(batch['ep_bce_weights']).float(),
            'ep_labels': torch.stack(batch['ep_labels']).float()
        }


class DataCollatorForMLMWithEPFast(object):

    def __init__(
        self,
        mlm_probability,
        ep_probability,
    ):
        self.mlm_probability = mlm_probability
        self.ep_probability = ep_probability

    def __call__(
        self,
        examples,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
            'ep_bce_weights': [],
            'ep_labels': []
        }

        batch_size = len(examples)

        node_pos_all = []
       
        for example in examples:
            
            input_ids, labels = self.mask_tokens(
                example['input_ids'], 
                example['special_tokens_mask']
            )
            ep_attention_mask, ep_bce_weights, ep_labels = \
                self.mask_graph_edges(
                    example['attention_mask'],
                    example['instruction_node_positions']
                )

            # add to batch
            batch['input_ids'].append(input_ids)
            batch['labels'].append(labels)
            batch['attention_mask'].append(ep_attention_mask)
            batch['ep_bce_weights'].append(ep_bce_weights)
            batch['ep_labels'].append(ep_labels)

            node_pos_all.append(example['instruction_node_positions'])

        return {
            # 'node_positions': node_pos_all,   # debug
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask']),
            'labels': torch.stack(batch['labels']),
            'ep_bce_weights': torch.stack(batch['ep_bce_weights']).float(),
            'ep_labels': torch.stack(batch['ep_labels']).float()
        }
    
    def mask_tokens(self, inputs, special_tokens_mask):
        """
        generate input_ids and labels for randomly masked input
        special tokens that should not be masked here are [CLS], [SEP], [PAD], and <INST>
        
        https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/data/data_collator.py#L751"
        """
        
        labels = inputs.clone()
         # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def mask_graph_edges(self, attention_mask, node_positions: List[int]):
        """
        Following GraphCodeBERT's edge prediction task:
        1. random sample nodes in the graph G = (V, E)
        2. mask direct edges connecting these sampled nodes by adding an infinitely 
           negative value in the mask matrix
        3. balance positive and negative predictions

        NOTE: 
        1. current edge masking will mask out node's self-attention
        2. `node_to_position`alreadly considered [CLS] at the front

        version 1:
            - dense computation
            - use weight 0 to mask out some edges
              https://discuss.pytorch.org/t/masking-binary-cross-entropy-loss/61065
        """
        
        # sample `ep_probability` of nodes
        sampled_nodes = random.sample(
                            node_positions, 
                            k=math.floor(self.ep_probability * len(node_positions)))
        
        if len(sampled_nodes) == 0:  # NOTE by zian: edge cases, e.g., there's only one instruction
            return attention_mask, \
                torch.zeros(attention_mask.shape, dtype=torch.bool), \
                torch.zeros(attention_mask.shape, dtype=torch.bool)

        # list edges

        ### cannot do non-successive indexesing in two dimensions
        # to_edges = attention_mask[sampled_nodes: node_positions]
        # from_edges = attention_mask[node_positions: sampled_nodes]

        ### iteratively processing with V_C
        # "self-edge" are not true edges but self-attention, remove
        adjacency_matrix = attention_mask.clone()
        torch.diagonal(adjacency_matrix, 0).zero_()

        to_edges = []
        from_edges = []
        for sid in sampled_nodes:
            to_edges.append(adjacency_matrix[sid, node_positions])
            from_edges.append(adjacency_matrix[node_positions, sid])
        to_edges = torch.stack(to_edges)
        from_edges = torch.stack(from_edges)

        # balanced sampling (in total 2 * V_C * V)
        # from and to edges will overlap in V_C * V_C
        n_positives = torch.sum(to_edges) + torch.sum(from_edges)
        n_negatives = 2 * len(sampled_nodes) * len(node_positions) - n_positives

        if n_positives > n_negatives:
            bernouli_p = n_negatives / n_positives
            to_edges_probability_matrix = torch.zeros(size=to_edges.shape, dtype=float)
            to_edges_probability_matrix.masked_fill_(to_edges, bernouli_p)
            to_edges_weight_matrix = torch.bernoulli(to_edges_probability_matrix).bool()
            to_edges_weight_matrix = torch.logical_or(to_edges_weight_matrix, ~to_edges)

            from_edges_probability_matrix = torch.zeros(size=from_edges.shape, dtype=float)
            from_edges_probability_matrix.masked_fill_(from_edges, bernouli_p)
            from_edges_weight_matrix = torch.bernoulli(from_edges_probability_matrix).bool()
            from_edges_weight_matrix = torch.logical_or(from_edges_weight_matrix, ~from_edges)

        else:
            bernouli_p = n_positives / n_negatives
            to_edges_probability_matrix = torch.zeros(size=to_edges.shape, dtype=float)
            to_edges_probability_matrix.masked_fill_(~to_edges, bernouli_p)
            to_edges_weight_matrix = torch.bernoulli(to_edges_probability_matrix).bool()
            to_edges_weight_matrix = torch.logical_or(to_edges_weight_matrix, to_edges)

            from_edges_probability_matrix = torch.zeros(size=from_edges.shape, dtype=float)
            from_edges_probability_matrix.masked_fill_(~from_edges, bernouli_p)
            from_edges_weight_matrix = torch.bernoulli(from_edges_probability_matrix).bool()
            from_edges_weight_matrix = torch.logical_or(from_edges_weight_matrix, from_edges)
            
        # print(f"bernouli probability: {bernouli_p}")

        ep_input_attention_mask = None
        ep_weights_for_BCE_loss = torch.zeros(attention_mask.shape, dtype=torch.bool)
        ep_labels = attention_mask
        for i, vc_id in enumerate(sampled_nodes):
            for j, v_id in enumerate(node_positions):
                ep_weights_for_BCE_loss[vc_id, v_id] = to_edges_weight_matrix[i, j]
                ep_weights_for_BCE_loss[v_id, vc_id] = from_edges_weight_matrix[i, j]   # to be tested
        ep_input_attention_mask = torch.logical_and(attention_mask, ~ep_weights_for_BCE_loss)
        ep_labels = torch.logical_and(attention_mask, ep_weights_for_BCE_loss)

        return ep_input_attention_mask, ep_weights_for_BCE_loss, ep_labels


class DataCollatorForCodeArtFast(DataCollatorForMLMWithEPFast):

    def __init__(self, mlm_probability, ep_probability):
        super().__init__(mlm_probability, ep_probability)

    def __call__(
        self,
        examples,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'relative_position_matrix': [],
            'labels': [],
            'ep_bce_weights': [],
            'ep_labels': []
        }

        batch_size = len(examples)

        node_pos_all = []
       
        for example in examples:

            input_ids, labels = self.mask_tokens(
                example['input_ids'], 
                example['special_tokens_mask']
            )
            ep_attention_mask, ep_bce_weights, ep_labels = \
                self.mask_graph_edges(
                    example['attention_mask'],
                    example['instruction_node_positions']
                )

            # add to batch
            batch['input_ids'].append(input_ids)
            batch['relative_position_matrix'].append(example['relative_position_matrix'])
            batch['labels'].append(labels)
            batch['attention_mask'].append(ep_attention_mask)
            batch['ep_bce_weights'].append(ep_bce_weights)
            batch['ep_labels'].append(ep_labels)

            node_pos_all.append(example['instruction_node_positions'])

        return {
            # 'node_positions': node_pos_all,   # debug
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask']),
            'relative_position_matrix': torch.stack(batch['relative_position_matrix']),
            'labels': torch.stack(batch['labels']),
            'ep_bce_weights': torch.stack(batch['ep_bce_weights']).float(),
            'ep_labels': torch.stack(batch['ep_labels']).float()
        }


class DataCollatorForGCB(DataCollatorForMLMWithEdgePred):

    def mask_graph_edges(self, attention_mask, node_positions: List[int]):
                # sample `ep_probability` of nodes
        sampled_nodes = random.sample(
                            node_positions, 
                            k=math.floor(self.ep_probability * len(node_positions)))
        
        if len(sampled_nodes) == 0:  # NOTE by zian: edge cases, e.g., there's only one instruction
            return attention_mask, \
                torch.zeros(attention_mask.shape, dtype=torch.bool), \
                torch.zeros(attention_mask.shape, dtype=torch.bool)

        # list edges

        ### cannot do non-successive indexesing in two dimensions
        # to_edges = attention_mask[sampled_nodes: node_positions]
        # from_edges = attention_mask[node_positions: sampled_nodes]

        ### iteratively processing with V_C
        # "self-edge" are not true edges but self-attention, remove
        adjacency_matrix = attention_mask.clone()
        torch.diagonal(adjacency_matrix, 0).zero_()

        to_edges = []
        from_edges = []
        for sid in sampled_nodes:
            to_edges.append(adjacency_matrix[sid, 1:-1])
            from_edges.append(adjacency_matrix[1:-1, sid])
        to_edges = torch.stack(to_edges)
        from_edges = torch.stack(from_edges)

        # balanced sampling (in total 2 * V_C * V)
        # from and to edges will overlap in V_C * V_C
        n_positives = torch.sum(to_edges) + torch.sum(from_edges)
        n_negatives = 2 * len(sampled_nodes) * 510 - n_positives

        if n_positives > n_negatives:
            bernouli_p = n_negatives / n_positives
            to_edges_probability_matrix = torch.zeros(size=to_edges.shape, dtype=float)
            to_edges_probability_matrix.masked_fill_(to_edges, bernouli_p)
            to_edges_weight_matrix = torch.bernoulli(to_edges_probability_matrix).bool()
            to_edges_weight_matrix = torch.logical_or(to_edges_weight_matrix, ~to_edges)

            from_edges_probability_matrix = torch.zeros(size=from_edges.shape, dtype=float)
            from_edges_probability_matrix.masked_fill_(from_edges, bernouli_p)
            from_edges_weight_matrix = torch.bernoulli(from_edges_probability_matrix).bool()
            from_edges_weight_matrix = torch.logical_or(from_edges_weight_matrix, ~from_edges)

        else:
            bernouli_p = n_positives / n_negatives
            to_edges_probability_matrix = torch.zeros(size=to_edges.shape, dtype=float)
            to_edges_probability_matrix.masked_fill_(~to_edges, bernouli_p)
            to_edges_weight_matrix = torch.bernoulli(to_edges_probability_matrix).bool()
            to_edges_weight_matrix = torch.logical_or(to_edges_weight_matrix, to_edges)

            from_edges_probability_matrix = torch.zeros(size=from_edges.shape, dtype=float)
            from_edges_probability_matrix.masked_fill_(~from_edges, bernouli_p)
            from_edges_weight_matrix = torch.bernoulli(from_edges_probability_matrix).bool()
            from_edges_weight_matrix = torch.logical_or(from_edges_weight_matrix, from_edges)
            
        # print(f"bernouli probability: {bernouli_p}")

        ep_input_attention_mask = None
        ep_weights_for_BCE_loss = torch.zeros(attention_mask.shape, dtype=torch.bool)
        ep_labels = attention_mask
        for i, vc_id in enumerate(sampled_nodes):
            for j, v_id in enumerate(range(1, 511)):
                ep_weights_for_BCE_loss[vc_id, v_id] = to_edges_weight_matrix[i, j]
                ep_weights_for_BCE_loss[v_id, vc_id] = from_edges_weight_matrix[i, j]   # to be tested
        ep_input_attention_mask = torch.logical_and(attention_mask, ~ep_weights_for_BCE_loss)
        ep_labels = torch.logical_and(attention_mask, ep_weights_for_BCE_loss)

        return ep_input_attention_mask, ep_weights_for_BCE_loss, ep_labels


# `load_dataset` helpers (add load_dataset scripts later)

def hide_recursion(binkit_js):
    "to support huggingface `load_dataset` from json"
    non_recursive_js = {}
    for k, v in binkit_js:
        non_recursive_js[k] = json.dumps(v)
    return non_recursive_js

def recover_recursion(non_recursive_js):
    binkit_js = {}
    for k, v in non_recursive_js:
        binkit_js[k] = json.loads(v)
    return binkit_js


if __name__ == '__main__':

    # preprocess the dataset in advance
    with open('../data/binkit-O0.jsonl', 'r') as f_in, \
        open('../data/binkit-O0-re-nr.jsonl', 'w') as f_out:
        for line in tqdm(f_in.readlines()):
            js = json.loads(line.strip())
            normalized_code = []
            for inst_id, raw_code_inst in js['code']:
                normalized_code.append((inst_id, custom_preprocessing(raw_code_inst)))
            js['code'] = normalized_code
            f_out.write(json.dumps(js) + '\n')
            
            # non_recursive_js = hide_recursion(js)
            # f_out.write(json.dumps(non_recursive_js) + '\n')