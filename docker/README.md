# Introduction
This project is for building a baseline container image of PyTorch(Nvidia Accelerated) for our team.

PyTorch-Nvidia guide: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch

Main branch only contains essential pre-requisites, for instance, python libraries (numpy, etc.)

And everyone could fork this repo or create a new branch or build a new folder to build your own container image.

# Directory structure
```shell
> tree
.
├── Dockerfile
├── code
│   └── echo.py
├── data
│   └── result.txt
└── script
    └── exec.sh
```

The typical directory structure looks like above.
* Dockerfile: describe how to build the image
* code: saves the code need to run, commands in Dockerfile will copy them into the image
* data: saves input & output data, you need to mount this folder to a host directory
* script: saves script to run, commands in Dockerfile will copy them into the image

# Steps
## Build your own Dockerfile
* By fork this repo
* By create a new branch
  * `DO REMEMBER: do not merge into main branch`
* By create a new folder
  * `Please use your name as folder name`

## Prepare your code, data & script
* Make sure code runs well & copy it to code folder on host machine
  * `Remember to output your result in data folder in your code`
* Copy your data to data folder on host machine
  * In case the dataset is too large, so we need to mount it to the image while running
* Modify ./script/exec.sh to run your own python command

## Build container image
```shell
> docker build -t ${IMAGE_NAME}:${TAG} .
```
For instance: **docker build -t pytorch.nvidia:v1 .**

## Run the docker
```shell
> docker run -d -v ${HOST_DIRECTORY_ABSOLUTE_PATH}:/usr/uow/app/data --rm ${IMAGE_NAME}  
```
For instance: **docker run -v /data:/usr/uow/app/data --rm pytorch.nvidia:v1**

## Retrieve result
And now data is saved in the host directory specified in docker run command