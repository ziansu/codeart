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
Running on public URL: https://24e27529fc2d7d47ae.gradio.live
```

The CodeArt repository is located at `/workspace/codeart` in the container. The model checkpoints and datasets are hosted on Hugging Face, but the links are already hardcoded in the demo.

If you want to access the models and datasets, you can find them [here](https://huggingface.co/collections/PurCL/codeart-artifacts-662e73dcd9b837e4b970c9be).

## A Quick Example

In this example, we will reproduce

