import random
import torch


class DataCollatorForRabert(object):

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(
        self,
        examples
    ):
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        for example in examples:
            encoded = self.tokenizer.inst_encode(eval(example['code']), eval(example['data_dep']))
            batch['input_ids'].append(encoded['input_ids'])
            batch['attention_mask'].append(encoded['attention_mask'])            
        return {
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask'])            
        }


class DataCollatorForCodeArt(object):

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer        

    def __call__(
        self,
        examples
    ):
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'relative_position_matrix': []        
        }
        
        for example in examples:
            num_functions = 0
            sequence_mask = []
            current_ids = []
            current_attention_mask = []
            current_relative_position_matrix = []
            for function in example['functions']:
                encoded = self.tokenizer.inst_encode(eval(function['code']), eval(function['data_dep']))
                current_ids.append(encoded['input_ids'])
                current_attention_mask.append(encoded['attention_mask'])
                current_relative_position_matrix.append(encoded['relative_position_matrix'])
                num_functions += 1
                sequence_mask.append(1)
            batch['input_ids'].append(torch.stack(current_ids))
            batch['attention_mask'].append(torch.stack(current_attention_mask))
            batch['relative_position_matrix'].append(torch.stack(current_relative_position_matrix))
        return {
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask']),
            'relative_position_matrix': torch.stack(batch['relative_position_matrix']),            
        }