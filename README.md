# Overview

This file contains the instructions for running CodeArt and reproducing the results in the paper.
Reviewers will need to install Docker and pull the Docker image from [Docker Hub](https://hub.docker.com/repository/docker/sheepy928/codeart).
Then, they can run CodeArt in the Docker image.



# Setup and Run a Quick Example

## Pull the Docker Image

Please use the following command to pull the Docker image.
The size of the image is about 3.8GB, which contains the CodeArt repository and a full PyTorch environment with CUDA installed. Please note that it may take a few minutes to download the image.

```bash
docker pull sheepy928/codeart:v1-release
```

After pulling the image, please run the container with the following command.
`-it` means the container is interactive.
```bash
docker run -it -p 47906:47906 codeart
```
The following message will be shown if the container is successfully started. Note that the public URL will be different each time, and each is valid for 72 hours or until the container is stopped. You can access our demo through either the local URL or the public URL.
```
Running on local URL:  http://127.0.0.1:47906
Running on public URL: https://<some_random_character>.gradio.live
```

The CodeArt repository is located at `/workspace/codeart` in the container. The model checkpoints and datasets are hosted on Hugging Face, but the links are already hardcoded in the demo.

If you want to access the models and datasets, you can find them [here](https://huggingface.co/collections/PurCL/codeart-artifacts-662e73dcd9b837e4b970c9be).

## A Quick Example

In this example, we will reproduce one experiment reported Table 1 in the paper.

1. Open a web browser and visit the URL `https://<some_random_character>.gradio.live` as shown in the message or `http://127.0.0.1:47906` if you are running the container locally.

2. Click on the `Table 1` tab on the top of the page.

3. Select the model as `PurCL/codeart-26m` and the dataset as `binutilsh`, which corresponds to the `Binutils` experiment in Table 1. Then, select the pool size as `50`. You will notice that `Run Alias` is automatically populated, and you do not need to change it.

4. Click on the `Run` button. The model will be downloaded from Hugging Face, and it may take a few minutes to download depending on your network speed. After the model is downloaded, the model will automatically be evaluated on the dataset, and the results will be displayed on the right side of the page. Please note that on a single A6000 GPU, the evaluation may take around 10 minutes.

5. You may see the actual command executed in the terminal.

## Notes

If the reviewer notices “Error” in the output block, please refresh the webpage. If the error persists, please restart the server program.

Please run one experiment at a time.

## Figure 8

Figure 8 evaluates the performance of CodeArt on the binary similarity downstream task.
The testing process of a binary similarity model is as follows: The model takes as input two binary files that are compiled from the same source code but with different compiler flags. For each function in the binary files, the model encodes the function into a vector (i.e., the embedding of the function).

Then for each function in the binary file compiled with the `-O0` flag, we compute the cosine similarity between the embedding of the function and the embeddings of candidate functions in the binary file compiled with the `-O3` flag. Then we rank the candidate functions according to the cosine similarity, and record the rank of the function that corresponds to the same source code function as the function being queried.

In Fig 8, the x-axis denotes pool sizes and y-axis the performance. It is worth noting that a larger pool size implies a more challenging setup. Therefore, while our interface supports all the experiments, the performance of CodeArt can be validated by only running experiments with a pool size of 500.

### Key Experiments

This section describes how to run CodeArt with a pool size of 500. It takes around 1 hour in total. For the baseline models, we use the numbers reported by DiEmph[1] (for JTrans) and PEM[2] (for GNNs).

First, please click the tab “Figure8” in the UI. In the “Select Model” field, please select “PurCL/codeart-binsim”. In the “Select Dataset” field, please select a project corresponding to a subfigure in Figure 8.  For the “Pool Size” field, please select 500. Then click the button “run”. The results will be available within around 10 minutes. (We conduct the test with a single Nvidia A6000(48GB) GPU. The time may vary depending on GPUs.)

For example, for the following experiment:
{Select Model: PurCL/codeart-binsim, Select Dataset: libcurlh, Pool Size: 500}
the expected output looks like:
```shell
Number of overlapped functions: 666
……
Number of selected overlapped functions: 500
source embedding shape: (500, 768), target embedding shape: (500, 768)
{'recall': {1: 0.6948000000000001, 3: 0.8308, 5: 0.8704000000000001, 10: 0.9128000000000001}, 'mrr': 0.7722353968253969}
Final-PR@1:  0.6948000000000001
Final-MRR:  0.7722353968253969
```
The value in ‘recall’-1 (0.6948) is corresponding to the point (500, 0.694) in the subfigure for Curl.

Note that due to the random essence of the sampling pool of functions, the results may have variances no more than 5%.

### Other Experiments

Please change “Pool Size” to other values to obtain the results for the other pool sizes.

## Table 1

Table 1 is similar to Figure 8. It evaluates the performance of CodeArt in a zero-shot setup. Please refer to the previous section of Figure 8 for backgrounds about evaluating models on the binary similarity task.

It is sufficient to evaluate CodeArt on the most challenging setup (with a pool size of 500). Our interface supports all the experiments though.

### Key Experiments

This section describes how to run CodeArt with a pool size of 500. It takes around 1 hour in total.

Please set the “Select Model” to “PurCL/codeart-26m”. In the “Select Dataset” field, please select a project corresponding to a subfigure in Table 1.  For the “Pool Size” field, please select 500. Then click the button “run”. The results will be available within around 10 minutes.

For example, for the curl dataset, the expected output looks like
```shell
Number of overlapped functions: 666
Number of selected overlapped functions: 500
source embedding shape: (500, 768), target embedding shape: (500, 768)
…
{'recall': {1: 0.47639999999999993, 3: 0.63, 5: 0.6799999999999999, 10: 0.7464}, 'mrr': 0.5643303174603174}
Final-PR@1:  0.47639999999999993
Final-MRR:  0.5643303174603174
```
The ‘recall’-1 is 0.476, corresponding to the column Pool-size 500-CodeArt (the last column) and row “Curl” in Table 1, which has the value 0.47.
Note that the results may have variances no more than 5%.

### Other Experiments

Please set “Pool Size” to other values for experiments with a different pool sizes.

## Table 2

Table 2 evaluates the performance of CodeArt on the malware family classification downstream task. It takes as input N binary functions from a malware sample, and outputs a label denoting the family of the malware. The “N-Funcs” denotes how many functions are taken as input for the model to classify the whole binary to some malware families. Our interface supports results from baseline JTrans and CodeArt with different “N-Funcs” options.

### Key Experiments

First, please click the tab “Table 2” in the UI. In the “Select Model” field, please select “2Funcs-CodeArt”. Then, click the button “Run”. The results will be available in 10 minutes.

The expected output looks like:

```shell

[some loggings]

auc: 0.9248027229306747, lrap: 0.5966341229736315, lrl: 0.0864753141245597
```


## Table 3
Table 3 evaluates the zero-shot binary-similarity performance of CodeArt variants pretrained on BinCorp-3m for ablation study. The test set is coreutils and pool size is 100. Our interface supports all variants in the table. Specifically, the mapping between the name in the table and options in “Select Model” is:
“w/o local mask” -> “PurCL/codeart-3m-wo_local_mask”
“w/o trans-closure” -> “PurCL/codeart-3m-wo_trans_closure”
“max-trans-closure 4” -> “PurCL/codeart-3m-max_trans_closure_4”
“max-trans-closure 6” -> “PurCL/codeart-3m-max_trans_closure_6”
“w/o rel-pos-bias” -> “PurCL/codeart-3m-wo_rel_pos_bias”

### Key Experiments

First, please click the tab “Table 3” in the UI. In the “Select Model” field, please select “PurCL/codeart-3m_wo_local_mask”. In the “Select Dataset” field, please select “coreutilsh”. In the “Pool Size” field, please select 100. Then, click the button “Run”. The time cost and output format are similar to previous binary similarity experiments.

Note that results in this table also have some randomness as other binary-similarity results, but the relative performance gap is consistent.


## Figure 9

Figure 9 evaluates the performance of CodeArt on the type inference downstream task. Our interface supports CodeArt for type inference with different optimization levels (O0, O1, O2, and O3).

### Key Experiments

First, please click the tab “Figure 9” in the UI. In the “Select Optimization Level” field, please select “O1”. Then, click the button “Run”. The results will be available in 10 minutes.

The expected output looks like:

```shell

[some loggings]

***** predict metrics *****
  predict_f1             	= 	0.9447
  predict_loss           	= 	0.0175
  predict_precision      	= 	0.9447
  predict_recall         	= 	0.9447
  predict_runtime        	= 0:00:48.47
  predict_samples        	=   	4124
  predict_samples_per_second = 	85.081
  predict_steps_per_second   =  	0.681
```
