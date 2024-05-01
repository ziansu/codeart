
import torch
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class CodeArtTokenizer(PreTrainedTokenizerFast):

    inst_token = '<INST>'
    maskbuilder = None
    local_patterns=True

    def inst_encode(
        self,
        code: List[Tuple[int, str]],
        data_dep: List[Tuple[int, int]],
        max_transitions=None,
        return_extra_info=False,
    ):
        # tokenization & locate <INST> tokens
        tokens, instruction_node_positions = [], []
        special_tokens_mask = []    # for MLM
        tokens.append(self.cls_token)
        special_tokens_mask.append(1)

        for inst_id, instruction in code:
            if len(tokens) >= self.model_max_length:
                break
            instruction_tokens = self.tokenize(instruction)
            instruction_node_positions.append(len(tokens))
            tokens.append(self.inst_token)
            special_tokens_mask.append(1)
            tokens += instruction_tokens
            special_tokens_mask += [0] * len(instruction_tokens)
            assert(len(tokens) == len(special_tokens_mask))  # debug
            
        # truncation & padding & [SEP]
        tokens = tokens[:self.model_max_length - 1]
        special_tokens_mask = special_tokens_mask[:self.model_max_length - 1]
        assert(len(tokens) == len(special_tokens_mask))  # debug
        tokens.append(self.sep_token)
        special_tokens_mask.append(1)
        if len(tokens) < self.model_max_length:
            tokens += [self.pad_token] * (self.model_max_length - len(tokens))
            special_tokens_mask += [1] * (self.model_max_length - len(special_tokens_mask))
        
        # print(len(tokens), len(special_tokens_mask))
        assert(len(tokens) == len(special_tokens_mask))  # debug

        # convert tokens to ids
        input_ids = self.convert_tokens_to_ids(tokens)

        assert self.maskbuilder

        if not self.local_patterns:
            attention_mask, relative_position_matrix = \
                self.maskbuilder.create_attention_mask_and_relative_position_matrix_no_local(
                    self.model_max_length,
                    instruction_node_positions,
                    data_dep,
                    max_transitions=max_transitions
                )
        else:
            attention_mask, relative_position_matrix = \
                self.maskbuilder.create_attention_mask_and_relative_position_matrix(
                    self.model_max_length,
                    instruction_node_positions,
                    data_dep,
                    max_transitions=max_transitions
                )

        if return_extra_info:
            return {    # NOTE: `attention_mask` needs to be bool to support collator's edge sampling
                'input_ids': torch.tensor(input_ids, dtype=torch.long), \
                'attention_mask': attention_mask, \
                'special_tokens_mask': torch.tensor(special_tokens_mask, dtype=torch.long),
                'relative_position_matrix': relative_position_matrix, \
                'instruction_node_positions': instruction_node_positions
            }
        else:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long), \
                'attention_mask': attention_mask, \
                'relative_position_matrix': relative_position_matrix
            }

    def batch_inst_encode(
        self,
        examples,
        max_transitions=None,
    ):
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'relative_position_matrix': []
        }
        
        for example in examples:
            # encoded = self.inst_encode(eval(example['code']), eval(example['data_dep']))
            encoded = self.inst_encode(example['code'], example['data_dep'], max_transitions=max_transitions)
            batch['input_ids'].append(encoded['input_ids'])
            batch['attention_mask'].append(encoded['attention_mask'])
            batch['relative_position_matrix'].append(encoded['relative_position_matrix'])
        return {
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask']),
            'relative_position_matrix': torch.stack(batch['relative_position_matrix'])
        }