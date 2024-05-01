# Overview

This file contains the instructions for running CodeArt and reproducing the results in the paper.
Reviewers will need to install Docker and pull the Docker image from [Docker Hub TODO]().
Then, they can run CodeArt in the Docker image.

# Setup and Run a Quick Example

## Pull the Docker Image

Please use the following command to pull the Docker image.
The size of the image is about 3GB. It may take a few minutes to download the image.

```bash
docker pull sheepy928/codeart:v1-release
```

After pulling the image, please run the container with the following command.
`-it` means the container is interactive.
```bash
docker run -it -p 47906:47906 codeart
```
The following message will be shown if the container is successfully started. Note that the public URL will be different each time. You can access our demo through either the local URL or the public URL.
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