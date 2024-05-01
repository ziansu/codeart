# Binary Similarity

This file illustrates how we evaluate CodeArt on the binary similarity task.
Please first download the checkpoint for CodeArt as specified in the `README.md` in the root directory of this repository.

## Download Other Files

Please download the preprocessed test data from [this google drive link](https://drive.google.com/file/d/1KSJMVtSgoI5bBMx9xhQtvBadUU3rehER/view?usp=share_link) to `evaluation/binary-similarity/cache` and unzip it.

Your directory should look like this:

```
evaluation/binary-similarity
|--cache
   |--binary_clone_detection
      |--binutilsh-pool.id
      |--...
```

## Finetuned Model

To provide a quick validation of CodeArt, we provide a finetuned model on the binary similarity task.
Please download the finetuned model from [this google drive link](https://drive.google.com/file/d/1FF1BS4kXkkB6561CV63GwruumPsGvgF6/view?usp=share_link) to `evaluation/save/codeart-binsim`.

Your directory should look like this:

```
evaluation/save
|--codeart-binsim
   |--checkpoint-4000
      |--pytorch_model.bin
      |--...
```

## Evaluation

The evaluation has three steps.
The first step is to encode the test data to their embeddings.
In the second step, the script constructs candidate function pools with different sizes,
and randomly picks functions to query the candidate pools.
In the third step, the script reports the results averaged over multiple runs.

For the encoding step, please run `encode.sh`.

For the evaluation step, please run `sample_and_report.sh`.

The raw results are stored in `report_ckpt-4k-<project name>-<pool size>.txt`.

For the report step, please use `pretty_print_all.py` to generate a human-readable report.
Specifically, please run `python3 pretty_print_all.py report_ckpt-4k`.

## Finetuning

Interested readers can finetune CodeArt on the binary similarity task by running `./run_config.sh config/train.json`.
Please fill in your wandb information in `run_config.sh` before running the script.
