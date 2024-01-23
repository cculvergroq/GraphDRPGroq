#FROM nvcr.io/nvidia/tensorrt:23.12-py3
FROM ubuntu:latest

WORKDIR /app/GraphDRP

RUN apt-get update -y \
    && apt-get install -y wget git \
    && apt-get clean

COPY . .

# install conda and activate base environment
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

ENV PATH ~/miniconda3/bin:$PATH


#RUN echo $PATH
#RUN ls ~/miniconda3/bin
#RUN ~/miniconda3/bin/conda --version
# these don't work even though conda is on path...
#RUN conda update conda \ 
#    && conda env create -f env_config.yml

RUN ~/miniconda3/bin/conda update conda \ 
    && ~/miniconda3/bin/conda env create -f env_config_cpu.yml 

ENV CONDA_DEFAULT_ENV GraphDRP

RUN ~/miniconda3/bin/conda init bash
RUN echo "conda activate GraphDRP" >> ~/.bashrc

WORKDIR /
# install improve library
RUN git clone https://github.com/JDACS4C-IMPROVE/IMPROVE

WORKDIR /app/GraphDRP

RUN sh download_csa.sh