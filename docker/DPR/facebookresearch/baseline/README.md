# Introduction
This project is for building a baseline container image of PyTorch(Nvidia Accelerated) for our team.

PyTorch-Nvidia guide: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch

And everyone could fork this repo or create a new branch or build a new folder to build your own container image.

# Directory structure
```shell
> tree
.
├── Dockerfile
├── conf
│   ├── README.md
│   ├── biencoder_train_cfg.yaml
│   ├── ctx_sources
│   │   └── default_sources.yaml
│   ├── datasets
│   │   ├── encoder_train_default.yaml
│   │   └── retriever_default.yaml
│   ├── dense_retriever.yaml
│   ├── encoder
│   │   └── hf_bert.yaml
│   ├── extractive_reader_train_cfg.yaml
│   ├── gen_embs.yaml
│   └── train
│       ├── biencoder_default.yaml
│       ├── biencoder_local.yaml
│       ├── biencoder_nq.yaml
│       └── extractive_reader_default.yaml
├── en_core_web_sm-2.3.0.tar.gz
├── run.py
└── run.sh
```

The typical directory structure looks like above.
* Dockerfile: describe how to build the image
* conf: saves the configuration needed by facebookresearch/DPR project. You can edit as needed and during building stage 'docker build' commands will copy them into the image
* en_core_web_sm-2.3.0.tar.gz: python package to install additionally.
* run.py: a python program to automatically run the facebooksearch/DPR project, including training, embedding, evaluation stages. And output the result.
* run.sh: an easy shell script used to start run.py in docker file.

# Steps
## Customize your own configuration.
facebookresearch/DPR provide a set of configuration to affect both the training & embedding procedure.
If there's any requirement, please edit the configuration in conf folder before build the container image.

## Build container image
```shell
> docker build -t ${IMAGE_NAME}:${TAG} .
```
For instance: **docker build -t pytorch.nvidia:v1 .**

## Run the docker
```shell
> docker run -d -v ${HOST_DIRECTORY_ABSOLUTE_PATH}:/usr/uow/nlp/DPR/result --rm ${IMAGE_NAME}  
```
For instance: **docker run -v /tmp/result:/usr/uow/nlp/DPR/result --rm pytorch.nvidia:v1**

**Make sure the ${HOST_DIRECTORY_ABSOLUTE_PATH} exists before running.**

## Retrieve result
Now data is saved in the host directory specified in docker run command. e.g.: /tmp/result/result.log
The result.log content looks like below.
```shell
> cat /tmp/result/result.log
Best Model:/usr/uow/nlp/DPR/model/dpr_biencoder.1
Final Accuracy:0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10
Accuracy Top 20: 0.10
Accuracy Top 100: 0.10
```