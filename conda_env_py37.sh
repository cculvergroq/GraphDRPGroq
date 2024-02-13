#!/bin/bash --login

set -e

# Manually run these commands before running this sciprt
# conda create -n GraphDRP
# conda activate GraphDRP
# conda install pip


# conda install pytorch torchvision cudatoolkit=12.2 -c pytorch --yes 
# h100 machine
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

#conda install pyg -c pyg -c conda-forge --yes
pip install torch_geometric


conda install -c conda-forge matplotlib --yes
conda install -c conda-forge h5py --yes

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
