

import torch
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class RabertTokenizer(PreTrainedTokenizerFast):

    inst_token = '<INST>'
    maskbuilder = None
    local_patterns=True

    def inst_encode(
        self,
        code: List[Tuple[int, str]],
        data_dep: List[Tuple[int, int]],
        return_extra_info=False,
        **kwargs
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

        # rabert mask (NOTE: currently paddings will all be masked out)
        if self.maskbuilder is None:
            from modeling_utils import create_attention_mask_aggressive
            attention_mask = create_attention_mask_aggressive(
                self.model_max_length,
                instruction_node_positions,
                data_dep
            )
        else:
            if not self.local_patterns:
                attention_mask = self.maskbuilder.create_attention_mask_no_local(
                    self.model_max_length,
                    instruction_node_positions,
                    data_dep
                )
            else:
                attention_mask = self.maskbuilder.create_attention_mask(
                    self.model_max_length,
                    instruction_node_positions,
                    data_dep
                )

        if return_extra_info:
            return {    # NOTE: `attention_mask` needs to be bool to support collator's edge sampling
                'input_ids': torch.tensor(input_ids, dtype=torch.long), \
                'attention_mask': attention_mask, \
                'special_tokens_mask': torch.tensor(special_tokens_mask, dtype=torch.long),
                'instruction_node_positions': instruction_node_positions
            }
        else:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long), \
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            }

    def batch_inst_encode(
        self,
        examples
    ):
        batch = {
            'input_ids': [],
            'attention_mask': []
        }
        
        for example in examples:
            encoded = self.inst_encode(example['code'], example['data_dep'])
            batch['input_ids'].append(encoded['input_ids'])
            batch['attention_mask'].append(encoded['attention_mask'])
        
        return {
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask'])
        }
    


class GCBLikeTokenizer(PreTrainedTokenizerFast):

    inst_token = '<INST>'
    maskbuilder = None
    local_patterns=True

    def inst_encode(
        self,
        code: List[Tuple[int, str]],
        data_dep: List[Tuple[int, int]],
        return_extra_info=False,
        **kwargs
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
            if len(instruction_tokens) > 1:  # maybe with operand
                tokens[instruction_node_positions[-1]] = instruction_tokens[1]  # overwrite <INST> with the operand
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

        # gcb-like mask
        assert self.maskbuilder
        attention_mask = self.maskbuilder.create_attention_mask(
            self.model_max_length,
            instruction_node_positions,
            data_dep
        )

        if return_extra_info:
            return {    # NOTE: `attention_mask` needs to be bool to support collator's edge sampling
                'input_ids': torch.tensor(input_ids, dtype=torch.long), \
                'attention_mask': attention_mask, \
                'special_tokens_mask': torch.tensor(special_tokens_mask, dtype=torch.long),
                'instruction_node_positions': instruction_node_positions
            }
        else:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long), \
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            }

    def batch_inst_encode(
        self,
        examples
    ):
        batch = {
            'input_ids': [],
            'attention_mask': []
        }
        
        for example in examples:
            encoded = self.inst_encode(example['code'], example['data_dep'])
            batch['input_ids'].append(encoded['input_ids'])
            batch['attention_mask'].append(encoded['attention_mask'])
        
        return {
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask'])
        }