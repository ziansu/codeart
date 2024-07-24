# Preprocess

Our preprocess pipeline is derived from the code of [jTrans](https://github.com/vul337/jTrans).
Our preprocess has three steps:

1. We use IDA Pro to disassemble the binary program. This script is largely based on the preprocessing script of jTrans.
2. We collect the preprocessed results from individual binary programs and merge them to a single file.
3. We perform a conservative dependence analysis to extract the program dependence.

This file introduces the two key components of our preprocess pipeline.

## Disassemble

Given an input binary program, we use IDA Pro to disassemble it.
Specifically, we use IDA Pro to obtain the control flow graph (CFG) of the binary program,
and further use IDA Pro to decode binary code to assembly instructions for each basic block.

The IDA script we use is `disassemble.py`.
It assumes the following directory structure:

```
example-project
|--unstrip
|  |--example-binary0.elf
|  |--...
|--extracted-bins
|  |--(empty)
|--example-binary0.elf
|--...
```

`example-project` is a directory denoting the project name (e.g., Coreutils).
Suppose that the example project contains a list of binary programs (e.g., example-bianry0.elf, example-binary1.elf, ...).
The directory `unstrip` contains the unstripped binary programs.
They are used to obtain the function names of the binary programs.

**The names are used exclusively for generating ground truth labels for the binary
similarity task and is not used in CodeArt.**

The directory `extracted-bins` is empty at the beginning. The IDA script will
store the intermediate results in this directory in the format of `.pickle` files.

The file `example-project/example-binary0.elf` is the binary program we want to disassemble.
It is stripped.

The IDA script runs as follow:

```shell
$PATH_TO_IDA/idat64 -A -S"$PWD/disassemble.py" example-project/example-binary0.elf
```

The script will generate the following file(s):

```
example-project
|--extracted-bins
|  |--example-binary0.elf_extract.pkl
   |--...
```

## Collect Preprocessed Results

This step aims to merge the preprocessed results from individual binary programs to a single file. The script is `collect.py`.
It takes as input a file that contains paths to the preprocessed results of individual binary programs and outputs a single pickle file that contains all the preprocessed results.
For example, the input file is similar to the following:

```
> cat example-list.txt
example-project/extracted-bins/example-binary0.elf_extract.pkl
example-project/extracted-bins/example-binary1.elf_extract.pkl
...
```

## Dependence Analysis

Then we perform a conservative dependence analysis to extract the program dependence.
The entry point of our analysis is in `analyze.py`.
It takes as input a pickle file obtained from the previous "collect" step, iterates over all functions in the disassembled binary program,
and outputs the input to CodeArt in a `.jsonl` file.

The main logic of our analysis is in `ExprLangAnalyzer` of `preprocess/analysis/expr_lang_analyzer.py`.

To use the resulting `.jsonl` file for evaluation, please refer to the file `codeart/evaluation/binary-similarity/dump_files.py`
This file takes as input three arguments: the paths to the query `.jsonl` and the pool `.jsonl` files, and the output path. The `README.md` file under `codeart/evaluation/binary-similarity` provides a link to Google Drive, containing the preprocessed test data used by CodeArt paper. The results of `codeart/evaluation/binary-similarity/dump_files.py` are expected to have the same format as our example data on Google Drive.

For more details on the binary similarity task evaluation, please refer to the `README.md` under `codeart/evaluation/binary-similarity`.
