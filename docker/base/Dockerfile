# base image from Nvidia official site
# tags ref: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:22.07-py3

# set up default working directory
# directory definition, you should organize your directories as follow structure
#   .
#   ├── code
#   ├── data
#   └── script
# code: saves your code to run
# data: saves input & output data
# script: saves shell script to run
RUN mkdir code && mkdir data && mkdir script
WORKDIR "/usr/uow/app"

# copy data
COPY code/ code/
COPY data/ data/
COPY script/ script/

# install pre-requisite libraries
#   if installation is complicated,
#   it is suggestted that compose them in a shell file and run the scrpit in the following 'RUN' command
RUN pip install numpy \
    && pip install pandas

# mount volumes if needed
VOLUME data

# run the command
RUN chmod +x ./script/exec.sh
CMD ["sh", "-c", "./script/exec.sh"]
