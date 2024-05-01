import torch


class DataCollatorForCodeArt(object):

    def __init__(self, tokenizer, label2id) -> None:
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.comma_id = tokenizer.convert_tokens_to_ids(',')
        self.ignore = [
            tokenizer.convert_tokens_to_ids('u'),
            tokenizer.convert_tokens_to_ids('cpu'),
            tokenizer.convert_tokens_to_ids('m'),
            tokenizer.convert_tokens_to_ids('x')
        ]

    def __call__(
        self,
        examples
    ):
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'relative_position_matrix': [],
            'labels': []
        }
        
        for example in examples:
            encoded = self.tokenizer.inst_encode(
                eval(example['code']), 
                eval(example['data_dep']), 
                return_extra_info=True
            )
            batch['input_ids'].append(encoded['input_ids'])
            batch['attention_mask'].append(encoded['attention_mask'])
            batch['relative_position_matrix'].append(encoded['relative_position_matrix'])

            instruction_node_positions = encoded['instruction_node_positions']
            instruction_labels = eval(example['code_w_type'])
            token_labels = [-100]   # ignore cls
            for i, inst_id in enumerate(instruction_node_positions):
                if 'base(char)' in instruction_labels[i]:
                    print(eval(example['code'])[i])
                start = inst_id
                if i + 1 == len(instruction_node_positions):
                    end = encoded['input_ids'].tolist().index(self.tokenizer.sep_token_id)
                else:
                    end = instruction_node_positions[i + 1]

                # ignore <INST>
                token_labels.append(-100)   # NOTE: may surpass max_length
                # get operator type
                token_labels.append(self.label2id[instruction_labels[i][0]])    # NOTE: may surpass max_length
                # get operands by `,`
                cur_id = start + 2
                if start < 511 and encoded['input_ids'][start + 1] in self.ignore:
                    cur_type_id = 0
                else:
                    cur_type_id = 1
                while cur_id < end:
                    if encoded['input_ids'][cur_id] == self.comma_id:
                        token_labels.append(-100)
                        cur_type_id += 1
                    else:
                        try:
                            token_labels.append(self.label2id[instruction_labels[i][cur_type_id]])
                        except IndexError:
                            print(example['metadata'])
                            print(self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][start: end]))
                            print(end, cur_type_id, self.tokenizer.sep_token_id, self.tokenizer.convert_ids_to_tokens(self.tokenizer.sep_token_id))
                            # raise IndexError

                            # set `cur_type_id` to 0
                            cur_type_id = 0
                            token_labels.append(self.label2id[instruction_labels[i][cur_type_id]])
                    cur_id += 1
            
            # brute force
            if len(token_labels) < self.tokenizer.model_max_length:
                token_labels += [-100] * (self.tokenizer.model_max_length - len(token_labels))

            if len(token_labels) > self.tokenizer.model_max_length:
                # print([(t, l) for l, t in zip(token_labels, self.tokenizer.convert_ids_to_tokens(encoded['input_ids']))])
                # print(token_labels[-1])
                token_labels = token_labels[:self.tokenizer.model_max_length - 1] + [-100]  # force [SEP] at end
            assert len(token_labels) == self.tokenizer.model_max_length

            batch["labels"].append(token_labels)

        return {
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask']),
            'relative_position_matrix': torch.stack(batch['relative_position_matrix']),
            'labels': torch.tensor(batch['labels'], dtype=torch.long)
        }