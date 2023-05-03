from collections import OrderedDict
from pathlib import Path
from pubchempy import *
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from utils import *
import argparse
import csv
import h5py
import json, pickle
import math
import matplotlib.pyplot as plt
import networkx as nx
import numbers
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys

import candle
# from improve_utils import imp_globals, load_rsp_data, read_df, get_common_samples, load_ge_data
import improve_utils as imp
# import improve_utils
from improve_utils import improve_globals as ig


"""
Functions below are used to generate graph molecular structures.
"""

# fdir = os.path.dirname(os.path.abspath(__file__)) # parent dir
fdir = Path(__file__).resolve().parent


def atom_features(atom):
    """ (ap) Extract atom features and put into array. """
    # a1 = one_of_k_encoding_unk(atom.GetSymbol(), [
    #         'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
    #         'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
    #         'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
    #         'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
    #     ])
    # a2 = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # a3 = one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # a4 = one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # a5 = [atom.GetIsAromatic()]
    # arr = np.array(a1 + a2 + a3 + a4 + a5)
    # return arr
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
            'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
            'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
            'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]) + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetImplicitValence(),
                                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    """ (ap) Convert SMILES to graph. """
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()  # num atoms in molecule

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()  # return a directed graph
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    # (ap) How is edges list different from edge_index list??
    # It seems that len(edge_index) is twice the size of len(edges)
    return c_size, features, edge_index


"""
The functions below generate datasets for CSG (data from July 2020) - Start
"""

# These are globals for all models
# TODO:
# Where do these go?
# Solution:
# ... 
# raw_data_dir_name = "raw_data"
# x_datad_ir_name = "x_data"
# y_data_dir_name = "y_data"
# canc_col_name = "CancID"
# drug_col_name = "DrugID"
# ge_fname = "ge.parquet"  # cancer feature
# smiles_fname = "smiles.csv"  # drug feature
# y_file_substr = "rsp"


def scale_fea(xdata, scaler_name='stnd', dtype=np.float32, verbose=False):
    """ Returns the scaled dataframe of features. """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    if scaler_name is None:
        if verbose:
            print('Scaler is None (not scaling).')
        return xdata
    
    if scaler_name == 'stnd':
        scaler = StandardScaler()
    elif scaler_name == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_name == 'rbst':
        scaler = RobustScaler()
    else:
        print(f'The specified scaler {scaler_name} is not supported (not scaling).')
        return xdata

    cols = xdata.columns
    return pd.DataFrame(scaler.fit_transform(xdata), columns=cols, dtype=dtype)


def raw_drp_data_to_ml_data(args):
    """
    Model-specific func.
    Generate a single ML data file from raw DRP data.
    The raw DRP data is defined in the IMPROVE doc website:
    https://jdacs4c-improve.github.io/docs/content/drp_overview.html#raw-drp-data.
    """
    # import ipdb; ipdb.set_trace()

    # -------------------
    # root = fdir/args.outdir
    root = args.outdir
    os.makedirs(root, exist_ok=True)

    # -------------------
    # Main data dir (e.g., CANDLE_DATA_DIR, IMPROVE_DATA_DIR)
    # TODO:
    # What shoud it be and how this should be specified? config_file?
    # main_data_dir = fdir/"improve_data_dir"
    ### new ... ig.main_data_dir
    ### main_data_dir = Path(args.main_data_dir)

    # -------------------
    # Specify path for the raw DRP data (download data from ftp?)
    # E.g., improve_data_dir/raw_data/data.ccle
    ### raw_data_dir = main_data_dir/ig.raw_data_dir_name  # contains folders "data.{source_data_name}" with raw DRP data
    ### new ... ig.raw_data_dir
    ### src_raw_data_dir = raw_data_dir/f"data.{args.source_data_name}"  # folder of specific data source with raw DRP data
    ### new ... not required

    # Download raw DRP data (which inludes the data splits)
    # download = True
    download = False
    # TODO:
    # Should the preprocess script take care of this downloading?
    # Where this should be specified? config_file?
    if download:
        # ftp_origin = f"https://ftp.mcs.anl.gov/pub/candle/public/improve/CSG_data"
        ftp_origin = f"https://ftp.mcs.anl.gov/pub/candle/public/improve/IMP_data"
        data_file_list = [f"data.{args.source_data_name}.zip"]
        for f in data_file_list:
            candle.get_file(fname=f,
                            origin=os.path.join(ftp_origin, f.strip()),
                            unpack=False, md5_hash=None,
                            cache_subdir=None,
                            datadir=raw_data_dir) # cache_subdir=args.cache_subdir

    # -------------------
    # Response data (general func for all models)
    # TODO: use this in all models
    ### rs = imp.load_rsp_data(src_raw_data_dir, y_col_name=args.y_col_name)  # TODO: use this in all models
    ### new ... 
    rs = imp.load_single_drug_response_data_new(
        source=args.source_data_name,
        split_file_name=args.split_file_name,
        y_col_name=args.y_col_name)

    # -------------------
    # Cancer data (general func for all models)
    # ge = read_df(src_raw_data_dir/ig.x_data_dir_name/ig.ge_fname)
    ### ge = imp.load_ge_data(src_raw_data_dir)  # TODO: use this in all models
    # import pdb; pdb.set_trace()
    ge = imp.load_gene_expression_data(gene_system_identifier="Gene_Symbol") ### new ...
    ge = ge.reset_index()
    print(ge[ig.canc_col_name].nunique())  # TODO: there are name duplicates??

    # Retain (canc, drug) response samples for which we have omic data
    rs, ge = imp.get_common_samples(df1=rs, df2=ge, ref_col=ig.canc_col_name)
    print(rs[[ig.canc_col_name, ig.drug_col_name]].nunique())

    # Use landmark genes (for gene selection)
    # TODO:
    # Eventually, lincs genes will provided with the raw DRP data (check with data curation team).
    # Thus, it will be part of CANDLE/IMPROVE functions.
    use_lincs = True
    if use_lincs:
        # with open(Path(src_raw_data_dir)/"../landmark_genes") as f:
        with open(fdir/"landmark_genes") as f:
            genes = [str(line.rstrip()) for line in f]
        # genes = ["ge_" + str(g) for g in genes]  # This is for our legacy data
        print("Genes count: {}".format(len(set(genes).intersection(set(ge.columns[1:])))))
        genes = list(set(genes).intersection(set(ge.columns[1:])))
        cols = [ig.canc_col_name] + genes
        ge = ge[cols]

    # Scale gene expression data
    # TODO:
    # We might need to save the scaler object (needs to be applied to test/infer data).
    ge_x_data = ge.iloc[:, 1:]
    ge_x_data_scaled = scale_fea(ge_x_data, scaler_name="stnd", dtype=np.float32, verbose=False)
    import pdb; pdb.set_trace()
    print(ge.shape)
    print(ge_x_data_scaled.shape)
    ge = pd.concat([ge[[ig.canc_col_name]], ge_x_data_scaled], axis=1)  # TODO: 

    # Below is omic data preparation for GraphDRP
    # ge = ge.iloc[:, :1000]  # Take subset of cols (genes)
    import pdb; pdb.set_trace()
    c_dict = {v: i for i, v in enumerate(ge[ig.canc_col_name].values)}  # cell_dict; len(c_dict): 634
    c_feature = ge.iloc[:, 1:].values  # cell_feature
    cc = {c_id: ge.iloc[i, 1:].values for i, c_id in enumerate(ge[ig.canc_col_name].values)}  # cell_dict; len(c_dict): 634

    # -------------------
    # Drugs data (general func for all models)
    # smi = imp.read_df(src_raw_data_dir/ig.x_data_dir_name/ig.smiles_fname)
    smi = imp.load_smiles_data(src_raw_data_dir)

    # Drug featurization for GraphDRP (specific to GraphDRP)
    d_dict = {v: i for i, v in enumerate(smi[ig.drug_col_name].values)}  # drug_dict; len(d_dict): 311
    d_smile = smi["SMILES"].values  # drug_smile
    smile_graph = {}  # smile_graph
    dd = {d_id: s for d_id, s in zip(smi[ig.drug_col_name].values, smi["SMILES"].values)}
    for smile in d_smile:
        g = smile_to_graph(smile)  # g: [c_size, features, edge_index]
        smile_graph[smile] = g

    # print("Unique drugs: {}".format(len(d_dict)))
    # print("Unique smiles: {}".format(len(smile_graph)))

    # -------------------
    # Get data splits and create folder for saving the generated ML data
    ids = imp.get_data_splits(src_raw_data_dir, args.splitdir_name, args.split_file_name, rs)

    # -------------------
    # Index data
    import ipdb; ipdb.set_trace()
    rsp_data = imp.get_subset_df(rs, ids)
    rsp_data.to_csv(root/"rsp.csv", index=False)

    # -------------------
    # Folder for saving the generated ML data
    # _data_dir = os.path.split(args.cache_subdir)[0]
    # root = os.getenv('CANDLE_DATA_DIR') + '/' + _data_dir
    # -----
    # ML_DATADIR = main_data_dir/"ml_data"
    # root = ML_DATADIR/f"data.{args.source_data_name}"/outdir_name # ML data
    # -----
    # root = fdir/args.outdir
    # os.makedirs(root, exist_ok=True)

    # -------------------
    # Extract features and reponse data
    def extract_data_vars(df, d_dict, c_dict, d_smile, c_feature, dd, cc, y_col_name="AUC"):
        """ Get drug and cancer feature data, and response values. """
        xd = []
        xc = []
        y = []
        xd_ = []
        xc_ = []
        nan_rsp_list = []
        miss_cell = []
        miss_drug = []
        meta = []
        # import ipdb; ipdb.set_trace()
        for i in range(df.shape[0]):  # tuples of (drug name, cell id, IC50)
            if i>0 and (i%15000 == 0):
                print(i)
            drug, cell, rsp = df.iloc[i, :].values.tolist()
            if np.isnan(rsp):
                nan_rsp_list.append(rsp)
            # If drug and cell features are available
            if drug in d_dict and cell in c_dict:  # len(drug_dict): 223, len(cell_dict): 990
                xd.append(d_smile[d_dict[drug]])   # xd contains list of smiles
                # xd_.append(dd[drug])   # xd contains list of smiles
                xc.append(c_feature[c_dict[cell]]) # xc contains list of feature vectors
                # xc_.append(cc[cell]) # xc contains list of feature vectors
                y.append(rsp)
                meta.append([drug, cell, rsp])
            elif cell not in c_dict:
                # import ipdb; ipdb.set_trace()
                miss_cell.append(cell)
            elif drug not in d_dict:
                # import ipdb; ipdb.set_trace()
                miss_drug.append(drug)

        # Three arrays of size 191049, as the number of responses
        xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)
        xd_, xc_ = np.asarray(xd_), np.asarray(xc_)
        meta = pd.DataFrame(meta, columns=[ig.drug_col_name, ig.canc_col_name, y_col_name])

        return xd, xc, y

    import ipdb; ipdb.set_trace()
    xd, xc, y = extract_data_vars(rsp_data, d_dict, c_dict, d_smile, c_feature, dd, cc)
    print("xd ", xd.shape, "xc ", xc.shape, "y_all ", y.shape)

    # -------------------
    # Create and save PyTorch data
    # TODO:
    # should DATA_FILE_NAME be global?
    DATA_FILE_NAME = "data"  # TestbedDataset() appends this string with ".pt"
    pt_data = TestbedDataset(
        root=root,
        dataset=DATA_FILE_NAME,
        xd=xd,
        xt=xc,
        y=y,
        smile_graph=smile_graph)
    return root


def parse_args(args):
    """ Parse input args. """
    # import ipdb; ipdb.set_trace(context=5)
    # print(args)

    parser = argparse.ArgumentParser(description="prepare dataset to train model")

    # IMPROVE required args
    parser.add_argument(
        "--source_data_name",
        type=str,
        required=True,
        help="Data source name.")
    parser.add_argument(
        "--target_data_name",
        type=str,
        default=None,
        required=False,
        help="Data target name (not required for GraphDRP).")
    # parser.add_argument(
    #     "--splitdir_name",
    #     type=str,
    #     required=True,
    #     help="Dir that stores the split files (e.g., 'splits', 'lc_splits'.")
    parser.add_argument(
        "--split_file_name",
        type=str,
        nargs="+",
        required=True,
        help="The path to the file that contains the split ids (e.g., 'split_0_tr_id',  'split_0_vl_id').")
    parser.add_argument(
        "--y_col_name",
        type=str,
        required=True,
        help="Drug sensitivity score to use as the target variable (e.g., IC50, AUC).")
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output dir to store the generated ML data files (e.g., 'split_0_tr').")
    # parser.add_argument(
    #     "--main_data_dir",
    #     type=str,
    #     required=True,
    #     help="Full path to main data dir such as CANDLE_DATADIR.")
    parser.add_argument(
        "--receipt",
        type=str,
        required=False,
        help="...")

    parser.add_argument(
        "--split",
        type=int,
        required=True,
        help="...")

    # Model-specific args
    # TODO: Where these should go?

    args = parser.parse_args(args)
    return args


def main(args):
    # print(args)
    args = parse_args(args)
    ml_data_path = raw_drp_data_to_ml_data(args)
    print(f"\nML data path:\t\n{ml_data_path}")
    print("\nFinished pre-processing (converted raw DRP data to model input ML data).")


if __name__ == "__main__":
    main(sys.argv[1:])
    # print(f"Finished {fdir}")
