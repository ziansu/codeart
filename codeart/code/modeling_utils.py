import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths import floyd_warshall_numpy
import torch
from typing import Any, Dict, List, Tuple


def create_attention_mask_aggressive(
        input_length: int, 
        instruction_node_positions: List[int], 
        data_dep: List[Tuple[int, int]]):
    
    # 0. [CLS] (attend to all) and [SEP] (attended by all) tokens
    # 1. regular token local pattern: letting local tokens only attend to bridge tokens
    # 2. bridge (<INST>) token's local and global patterns
    # 3. bridge dependency pattern
    #       NOTE: directed graph can be easier for later processing
    
    attention_mask = torch.zeros(size=(input_length, input_length), dtype=torch.bool)

    # [CLS]
    attention_mask[0, :] = 1
    attention_mask[:, 0] = 1    # all tokens can somehow get some global information from [CLS] token

    # [SEP]: currently this is not meaningful as rabert has no NSP

    for inst_id, position_id in enumerate(instruction_node_positions):
        next_position_id = instruction_node_positions[inst_id + 1] \
            if inst_id + 1 < len(instruction_node_positions) else input_length - 1
        
        # attention_mask[position_id + 1: next_position_id, instruction_node_positions] = 1
        attention_mask[position_id, position_id: next_position_id] = 1
        for token_position in range(position_id + 1, next_position_id):
            # local attend to local and direct bridge
            attention_mask[token_position, position_id: next_position_id] = 1
            # local to attend to all other bridges
            attention_mask[token_position, instruction_node_positions] = 1
            
    for source_inst_id, target_inst_id in data_dep:
        if source_inst_id >= inst_id or target_inst_id >= inst_id: # support truncation
            continue
        source_position_id = instruction_node_positions[source_inst_id]
        target_position_id = instruction_node_positions[target_inst_id]
        attention_mask[source_position_id, target_position_id] = 1
        attention_mask[target_position_id, source_position_id] = 1

    return attention_mask


def create_attention_mask_conservative(
    input_length,
    instruction_node_positions,
    data_dep
):
    """
    The difference between `aggressive` and `conservative` mask creation is that,
    `aggressive` only allows <INST> tokens as information source between dependent
    instructions, whereas `conservative` allows all tokens in dependent instruction
    context to attend to each other.

    Only `aggressive` version can lead to sparse solutions.
    """
    
    attention_mask = torch.zeros(size=(input_length, input_length), dtype=torch.bool)

    # [CLS]
    attention_mask[0, :] = 1
    attention_mask[:, 0] = 1    # all tokens can somehow get some global information from [CLS] token

    # [SEP]: currently this is not meaningful as rabert has no NSP

    for inst_id, position_id in enumerate(instruction_node_positions):
        next_position_id = instruction_node_positions[inst_id + 1] \
            if inst_id + 1 < len(instruction_node_positions) else input_length - 1
        
        attention_mask[position_id, position_id: next_position_id] = 1
        for token_position in range(position_id + 1, next_position_id):
            # local attend to local and direct bridge
            attention_mask[token_position, position_id: next_position_id] = 1
            # local to attend to all other bridges
            attention_mask[token_position, instruction_node_positions] = 1
            
    for source_inst_id, target_inst_id in data_dep:
        if source_inst_id >= inst_id or target_inst_id >= inst_id: # support truncation
            continue
        source_position_id = instruction_node_positions[source_inst_id]
        target_position_id = instruction_node_positions[target_inst_id]
        attention_mask[source_position_id, target_position_id] = 1
        attention_mask[target_position_id, source_position_id] = 1

    return attention_mask


def create_attention_mask_gcb(
    input_length,
    instruction_node_positions,
    data_dep
):
    attention_mask = torch.ones(size=(input_length, input_length), dtype=torch.bool)

    # local patterns
    for inst_id, position_id in enumerate(instruction_node_positions):
        next_position_id = instruction_node_positions[inst_id + 1] \
            if inst_id + 1 < len(instruction_node_positions) else input_length - 1
        
        # remove all node related attention first
        attention_mask[position_id, :] = 0
        attention_mask[:, position_id] = 0
        
        if position_id + 2 <= next_position_id:
            attention_mask[position_id, position_id + 2] = 1
            attention_mask[position_id + 2, position_id] = 1

    # graph patterns
    for source_inst_id, target_inst_id in data_dep:
        if source_inst_id >= inst_id or target_inst_id >= inst_id: # support truncation
            continue
        source_position_id = instruction_node_positions[source_inst_id]
        target_position_id = instruction_node_positions[target_inst_id]
        attention_mask[source_position_id, target_position_id] = 1
        attention_mask[target_position_id, source_position_id] = 1

    return attention_mask



class MaskBuilder(object):

    def __init__(
        self,
        preset=None,
        enable_global_memory_patterns=True,
        enable_bridge_patterns=True,
        enable_graph_patterns=True,
        device='cpu'
    ):
        self.preset = preset
        self.enable_global_memory_patterns = enable_global_memory_patterns
        self.enable_bridge_patterns = enable_bridge_patterns
        self.enable_graph_patterns = enable_graph_patterns
        self.device = device
        
    def create_attention_mask(
        self,
        input_length,
        instruction_node_positions,
        data_dep=None,
    ):
        if self.preset == 'graphcodebert':
            return create_attention_mask_gcb(input_length, instruction_node_positions, data_dep)
        elif self.preset is None:
            pass
        else:
            raise NotImplementedError

        attention_mask = torch.zeros(size=(input_length, input_length), dtype=torch.bool)

        # [CLS] token
        attention_mask[0, :] = 1

        # global memory
        if self.enable_global_memory_patterns:
            attention_mask[:, 0] = 1    # all tokens can somehow get some global information from [CLS] token

        # local patterns
        for inst_id, position_id in enumerate(instruction_node_positions):
            next_position_id = instruction_node_positions[inst_id + 1] \
                if inst_id + 1 < len(instruction_node_positions) else input_length - 1
            
            attention_mask[position_id, position_id: next_position_id] = 1
            for token_position in range(position_id + 1, next_position_id):
                # local attend to local and direct bridge
                attention_mask[token_position, position_id: next_position_id] = 1
                # local to attend to all other bridges
                if self.enable_bridge_patterns:
                    attention_mask[token_position, instruction_node_positions] = 1

        # graph patterns
        if self.enable_graph_patterns:
            for source_inst_id, target_inst_id in data_dep:
                if source_inst_id >= inst_id or target_inst_id >= inst_id: # support truncation
                    continue
                source_position_id = instruction_node_positions[source_inst_id]
                target_position_id = instruction_node_positions[target_inst_id]
                attention_mask[source_position_id, target_position_id] = 1
                attention_mask[target_position_id, source_position_id] = 1

        return attention_mask
    
    def create_attention_mask_no_local(
        self,
        input_length,
        instruction_node_positions,
        data_dep=None,
    ):

        attention_mask = torch.ones(size=(input_length, input_length), dtype=torch.bool)

        # [CLS] token and global memory already there

        # local patterns
        for inst_id, position_id in enumerate(instruction_node_positions):
            next_position_id = instruction_node_positions[inst_id + 1] \
                if inst_id + 1 < len(instruction_node_positions) else input_length - 1


        for i in instruction_node_positions:
            attention_mask[i, instruction_node_positions] = 0


        # graph patterns
        if self.enable_graph_patterns:
            for source_inst_id, target_inst_id in data_dep:
                if source_inst_id >= inst_id or target_inst_id >= inst_id: # support truncation
                    continue
                source_position_id = instruction_node_positions[source_inst_id]
                target_position_id = instruction_node_positions[target_inst_id]
                attention_mask[source_position_id, target_position_id] = 1
                attention_mask[target_position_id, source_position_id] = 1


        return attention_mask

    def create_attention_mask_and_relative_position_matrix(
        self,
        input_length,
        instruction_node_positions,
        data_dep,
        max_transitions=None,
    ):
        # 1. data_dep mask option
        # 2. bridge token option
        # 3. global memory token option

        attention_mask = torch.zeros(size=(input_length, input_length), dtype=torch.bool, device=self.device)

        # [CLS] token
        attention_mask[0, :] = 1

        # global memory
        if self.enable_global_memory_patterns:    # TODO: enable larger memory
            attention_mask[:, 0] = 1    # all tokens can somehow get some global information from [CLS] token
            # attention_mask[:, -1] = 1 # all tokens can somehow get some global information from [SEP] token

        # local patterns
        for inst_id, position_id in enumerate(instruction_node_positions):
            next_position_id = instruction_node_positions[inst_id + 1] \
                if inst_id + 1 < len(instruction_node_positions) else input_length - 1
            
            attention_mask[position_id, position_id: next_position_id] = 1
            for token_position in range(position_id + 1, next_position_id):
                # local attend to local and direct bridge
                attention_mask[token_position, position_id: next_position_id] = 1
                # local to attend to all other bridges
                if self.enable_bridge_patterns:
                    attention_mask[token_position, instruction_node_positions] = 1

        # filter out out-of-range dependencies
        remaining_data_dep = []
        if self.enable_graph_patterns:
            for source_inst_id, target_inst_id in data_dep:
                if source_inst_id >= inst_id or target_inst_id >= inst_id: # support truncation
                    continue
                else:
                    remaining_data_dep.append((source_inst_id, target_inst_id))
   
        # graph construction
        graph = nx.Graph()
        graph.add_nodes_from(range(len(instruction_node_positions)))
        graph.add_edges_from(remaining_data_dep)

        # dense graph all pairs shortest path length
        spl_matrix = floyd_warshall_numpy(graph)
        if max_transitions is not None:
            spl_matrix[spl_matrix > max_transitions] = -1
        spl_matrix[spl_matrix == np.inf] = -1
        spl_matrix = torch.tensor(spl_matrix, dtype=torch.long, device=self.device)

        # recover full matrix
        rel_pos_matrix = torch.full(size=(input_length, input_length), fill_value=-1, dtype=torch.long, device=self.device)
        for i, nid in enumerate(instruction_node_positions):
            rel_pos_matrix[nid, instruction_node_positions] = spl_matrix[i, :]


        # merge rel_pos_matrix into attention_mask
        attention_mask = torch.logical_or(attention_mask, rel_pos_matrix >= 0)

        return attention_mask, rel_pos_matrix
        
    def create_attention_mask_and_relative_position_matrix_no_local(
        self,
        input_length,
        instruction_node_positions,
        data_dep,
        max_transitions=None,
    ):
        # 1. data_dep mask option
        # 2. bridge token option
        # 3. global memory token option


        attention_mask = torch.ones(size=(input_length, input_length), dtype=torch.bool, device=self.device)


        # local patterns
        for inst_id, position_id in enumerate(instruction_node_positions):
            next_position_id = instruction_node_positions[inst_id + 1] \
                if inst_id + 1 < len(instruction_node_positions) else input_length - 1


        # filter out out-of-range dependencies
        remaining_data_dep = []
        if self.enable_graph_patterns:
            for source_inst_id, target_inst_id in data_dep:
                if source_inst_id >= inst_id or target_inst_id >= inst_id: # support truncation
                    continue
                else:
                    remaining_data_dep.append((source_inst_id, target_inst_id))
       
        # graph construction
        graph = nx.Graph()
        graph.add_nodes_from(range(len(instruction_node_positions)))
        graph.add_edges_from(remaining_data_dep)


        # dense graph all pairs shortest path length
        spl_matrix = floyd_warshall_numpy(graph)
        if max_transitions is not None:
            spl_matrix[spl_matrix > max_transitions] = -1
        spl_matrix[spl_matrix == np.inf] = -1
        spl_matrix = torch.tensor(spl_matrix, dtype=torch.long, device=self.device)


        # recover full matrix
        rel_pos_matrix = torch.full(size=(input_length, input_length), fill_value=-1, dtype=torch.long, device=self.device)
        for i, nid in enumerate(instruction_node_positions):
            rel_pos_matrix[nid, instruction_node_positions] = spl_matrix[i, :]
       
        for i in instruction_node_positions:
            attention_mask[i, instruction_node_positions] = 0


        # merge rel_pos_matrix into attention_mask
        attention_mask = torch.logical_or(attention_mask, rel_pos_matrix >= 0)


        return attention_mask, rel_pos_matrix