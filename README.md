# GraphDRP
GraphDRP model for drug response prediction (DRP).


# Dependencies
Check [conda_env_py37.sh](./conda_env_py37.sh)

ML framework:
+ [Torch](https://pytorch.org/)
+ [Pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) -- for graph neural network (GNN)

IMPROVE lib:
+ [improve_lib](https://github.com/JDACS4C-IMPROVE/IMPROVE)
+ [candle_lib](https://github.com/ECP-CANDLE/candle_lib) -- improve lib dependency


# Dataset
Benchmark data for cross-study analysis (CSA) can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).
The required data tree is shown below:

```
csa_data/raw_data/
├── splits
│   ├── CCLE_all.txt
│   ├── CCLE_split_0_test.txt
│   ├── CCLE_split_0_train.txt
│   ├── CCLE_split_0_val.txt
│   ├── CCLE_split_1_test.txt
│   ├── CCLE_split_1_train.txt
│   ├── CCLE_split_1_val.txt
│   ├── ...
│   ├── GDSCv2_split_9_test.txt
│   ├── GDSCv2_split_9_train.txt
│   └── GDSCv2_split_9_val.txt
├── x_data
│   ├── cancer_copy_number.tsv
│   ├── cancer_discretized_copy_number.tsv
│   ├── cancer_DNA_methylation.tsv
│   ├── cancer_gene_expression.tsv
│   ├── cancer_miRNA_expression.tsv
│   ├── cancer_mutation_count.tsv
│   ├── cancer_mutation_long_format.tsv
│   ├── cancer_mutation.parquet
│   ├── cancer_RPPA.tsv
│   ├── drug_ecfp4_nbits512.tsv
│   ├── drug_info.tsv
│   ├── drug_mordred_descriptor.tsv
│   └── drug_SMILES.tsv
└── y_data
    └── response.tsv
```

Note! `./data` contains data files that were used to train and evaluate the GraphDRP for the original paper.


## Source codes
+ `graphdrp_preprocess_improve.py`: creates data files for drug resposne prediction (DRP)
+ `graphdrp_train_improve.py`: trains the GraphDRP model
+ `graphdrp_infer_improve.py`: runs inference with the trained GraphDRP model
+ `graphdrp_params.txt`: parameter file


# Step-by-step running

### 1. Clone the repo
```
git clone https://github.com/JDACS4C-IMPROVE/GraphDRP/tree/develop
cd GraphDRP
```

### 2. Download benchmark data
```
sh ./download_csa.sh
```
This will download the cross-study benchmark data into `./csa_data/`.

### 3. Set computational environment
* Install dependencies (check [conda_env_py37.sh](./conda_env_py37.sh))
* Set the required environment variables to point towards the data folder and improve lib. You need to download the improve lib repo (follow this repo for more info `https://github.com/JDACS4C-IMPROVE/IMPROVE`).
```bash
export IMPROVE_DATA_DIR="./csa_data/"
export PYTHONPATH=$PYTHONPATH:/lambda_stor/data/apartin/projects/IMPROVE/pan-models/IMPROVE
```

### 4. Preprocess benchmark data (_raw data_) to construct model input data (_ML data_)
```bash
python graphdrp_preprocess_improve.py
```
Generates:
* three model input data files: `train_data.pt`, `val_data.pt`, `test_data.pt`
* three tabular data files, each containing y data (responses) and metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`

```
ml_data
└── GDSCv1-CCLE
    └── split_0
        ├── processed
        │   ├── test_data.pt
        │   ├── train_data.pt
        │   └── val_data.pt
        ├── test_y_data.csv
        ├── train_y_data.csv
        ├── val_y_data.csv
        └── x_data_gene_expression_scaler.gz
```

### 5. Train GraphDRP model
```bash
python graphdrp_train_improve.py
```
Trains GraphDRP using the processed data: `train_data.pt` (training), `val_data.pt` (for early stopping).

Generates:
* trained model: `model.pt`
* predictions on val data (tabular data): `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`
```
out_models
└── GDSCv1
    └── split_0
        ├── best -> /lambda_stor/data/apartin/projects/IMPROVE/pan-models/GraphDRP/out_models/GDSCv1/split_0/epochs/002
        ├── epochs
        │   ├── 001
        │   │   ├── ckpt-info.json
        │   │   └── model.h5
        │   └── 002
        │       ├── ckpt-info.json
        │       └── model.h5
        ├── last -> /lambda_stor/data/apartin/projects/IMPROVE/pan-models/GraphDRP/out_models/GDSCv1/split_0/epochs/002
        ├── model.pt
        ├── out_models
        │   └── GDSCv1
        │       └── split_0
        │           └── ckpt.log
        ├── val_scores.json
        └── val_y_data_predicted.csv
```

### 6. Run the trained model in inference mode on test data
```python graphdrp_infer_improve.py```
This script uses the processed data and the trained model to evaluate performance.

Generates:
* predictions on test data (tabular data): `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`
```
out_infer
└── GDSCv1-CCLE
    └── split_0
        ├── test_scores.json
        └── test_y_data_predicted.csv
```
