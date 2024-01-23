#FROM nvcr.io/nvidia/tensorrt:23.12-py3
FROM ubuntu:latest

WORKDIR /app

# install conda and activate base environment
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN ~/miniconda3/bin/conda init bash
RUN source/.bashrc

RUN conda create -n GraphDRP
RUN conda activate GraphDRP
RUN conda install pip

# install graph drp requirements
RUN sh conda_env_py37.sh

WORKDIR /
# install improve library
RUN git clone https://github.com/JDACS4C-IMPROVE/IMPROVE

WORKDIR /app/GraphDRP
