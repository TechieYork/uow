# base image from Nvidia official site
# tags ref: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:22.07-py3

# set up default working directory
WORKDIR "/usr/uow/nlp"

# install facebook dpr & dependices
RUN git clone https://github.com/facebookresearch/DPR
RUN pip install ./DPR

# copy run.py & run.sh to nlp directory
COPY conf/ ./DPR/conf/
COPY run.py run.sh en_core_web_sm-2.3.0.tar.gz ./
RUN pip install ./en_core_web_sm-2.3.0.tar.gz -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN mkdir ./DPR/result && chmod +x run.sh

# mount volumes if needed
VOLUME /usr/uow/nlp/DPR/result

# run the command
CMD ["sh", "-c", "./run.sh"]
