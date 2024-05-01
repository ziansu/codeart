# CodeArt: Better Code Models by Attention Regularization When Symbols Are Lacking

This repo contains code for the paper *CodeArt: Better Code Models by Attention Regularization When Symbols Are Lacking*.

## Environment

- torch==2.0.1
- transformers==4.30.2
- datasets==2.14.4
- networkx==3.1
- scikit-learn=1.3.0

## Quick Tour

To use CodeArt, you can follow this general pipeline of encoding instructions and dependences:

```python
import sys
sys.path.append('path_to_/code/')

from models import (
    CodeArtConfig,
    CodeArtTokenizer,
    CodeArtModel
)
from modeling_utils import MaskBuilder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = CodeArtTokenizer.from_pretrained("path_to_pretrained_checkpoint")
maskbuilder = MaskBuilder(
    preset=None,
    enable_global_memory_patterns=True,
    enable_bridge_patterns=False,
    enable_graph_patterns=True
)
tokenizer.maskbuilder = maskbuilder

model = CodeArtModel.from_pretrained("path_to_pretrained_checkpoint")
model.to(device)

# encode one example
instructions = [[0, 'push   r15'], [1, 'push    r14'], [2, 'mov r14d, r9d'], ...]
dependences = [[16, 12], [17, 16], [21, 10], [25, 13], ...]
encoded = tokenizer.inst_encode(instructions, dependences)
embeddings = model(
    input_ids=encoded['input_ids'],
    attention_mask=encoded['attention_mask'],
    relative_position_matrix=encoded['relative_position_matrix']
)
```

## Dependence Analysis and Preprocessing

To convert a binary program to an input to CodeArt, we need to
first use IDA Pro to disassemble the binary program and then
perform a conservative dependence analysis to extract the program dependence.

Please refer to `preprocess/README.md` for details.

## Datasets

The preprocessed datasets we use for training and evaluation are on HuggingFace Hub and you can refer to the configuration files to check them.

> Due to safety concerns, we will not directly release the raw binaries of the malware dataset.
> Instead, after publication, we will provide the raw binaries upon request to insterested researchers.
> For now, we release the sha256 hashes of the samples in the malware dataset in `evaluation/malware-family-classification/id2family.jsonl`.

## Pretraining

To replicate the pretraining, you can navigate to `scripts/`, and run `train_config.sh config/default.json`.

## Evaluation

The evaluation code of CodeArt is under the directory `evaluation/`.

### Binary Similarity Analysis

Please refer to `evaluation/binary-similarity/README.md` for details.

### Malware Family Classification

Navigate to `evaluation/malware-family-classification/`. To finetune CodeArt on this task, run `run_config.sh config/train-2f-100c.json`. To evaluate the finetuned model, run `eval_config.sh config/eval-2f-100c.json`. Note that you need to specify the correct `model_name_or_path` in the configurations.

### Type Inference

Navigate to `evaluation/type-inference/`. To finetune CodeArt on this task, run `run_config.sh config/train-O0.json` (you can modify the `dataset_name` in the configuration to finetune for O1, O2, and O3). To evaluate the finetuned model, run `eval_config.sh config/eval-O0.json`. Note that you need to specify the correct `model_name_or_path` in the configurations.

## Checkpoints

We release checkpoints of pretraining and downstream tasks in [this link](https://drive.google.com/drive/folders/1PwNLmWmjXYH8ZYYtD7HmOtMybvpTBMRp?usp=sharing). You can download these checkpoints and extract them to `checkpoints/`.