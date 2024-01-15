#!/bin/bash --login

set -e

# Manually run these commands before running this sciprt
# conda create -n GraphDRP_py37 python=3.7 pip --yes
# conda activate GraphDRP_py37

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch --yes
conda install pyg -c pyg -c conda-forge --yes

conda install -c conda-forge matplotlib --yes
conda install -c conda-forge h5py=3.1 --yes

conda install -c bioconda pubchempy --yes
conda install -c rdkit rdkit --yes
conda install -c anaconda networkx --yes
#conda install -c conda-forge pyarrow=10.0 --yes
pip install pyarrow # installed pyarrow=12.0.1
# above comments say to use python=3.7, but pyarrow=10.0 conflicts according to conda

conda install -c pyston psutil --yes

# IMPROVE
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop # CANDLE

# Other
# conda install -c conda-forge ipdb=0.13.9 --yes
# conda install -c conda-forge jupyterlab=3.2.0 --yes
# conda install -c conda-forge python-lsp-server=1.2.4 --yes

# Check installs
# python -c "import torch; print(torch.__version__)"
# python -c "import torch; print(torch.version.cuda)"
# python -c "import torch_geometric; print(torch_geometric.__version__)"
# python -c "import networkx; print(networkx.__version__)"
# python -c "import matplotlib; print(matplotlib.__version__)"
# python -c "import h5py; print(h5py.version.info)"
# python -c "import pubchempy; print(pubchempy.__version__)"
# python -c "import rdkit; print(rdkit.__version__)"
