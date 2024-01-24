FROM nvcr.io/nvidia/tensorrt:22.12-py3
#FROM ubuntu:latest

WORKDIR /app/GraphDRP

RUN apt-get update -y \
    && apt-get install -y wget git \
    && apt-get clean

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


RUN ~/miniconda3/bin/conda update conda

COPY . .

RUN ~/miniconda3/bin/conda env create -f env_config.yml 

ENV CONDA_DEFAULT_ENV GraphDRP

RUN ~/miniconda3/bin/conda init bash
RUN echo "conda activate GraphDRP" >> ~/.bashrc

WORKDIR /app
# install improve library
RUN git clone https://github.com/JDACS4C-IMPROVE/IMPROVE

WORKDIR /app/GraphDRP

# working dir currently has csa data...
#RUN sh download_csa.sh
RUN echo "export PYTHONPATH=/app/IMPROVE:$PYTHONPATH" >> ~/.bashrc
RUN echo "export IMPROVE_DATA_DIR=/app/GraphDRP/csa_data" >> ~/.bashrc
RUN echo "export ORT_TENSORRT_FP16_ENABLE=1" >> ~/.bashrc