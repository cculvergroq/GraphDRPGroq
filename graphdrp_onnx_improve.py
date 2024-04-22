""" Inference with GraphDRP for drug response prediction.

Required outputs
----------------
All the outputs from this infer script are saved in params["infer_outdir"].

1. Predictions on test data.
   Raw model predictions calcualted using the trained model on test data. The
   predictions are saved in test_y_data_predicted.csv

2. Prediction performance scores on test data.
   The performance scores are calculated using the raw model predictions and
   the true values for performance metrics specified in the metrics_list. The
   scores are saved as json in test_scores.json
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm

# Model-specific imports
from model_utils.torch_utils import (
    build_GraphDRP_dataloader,
    determine_device,
    set_GraphDRP,
    onnx_export,
)
from model_utils.models.gat import GATNet
from model_utils.models.gat_gcn import GAT_GCN
from model_utils.models.gcn import GCNNet
from model_utils.models.ginconv import GINConvNet
from model_utils.models.groq_wrapper import GroqWrapper

# [Req] Imports from preprocess and train scripts
from graphdrp_preprocess_improve import preprocess_params
from graphdrp_train_improve import metrics_list, train_params

filepath = Path(__file__).resolve().parent # [Req]

import numpy as np
import torch
torch.manual_seed(0)
np.random.seed(0)

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_infer_params
# 2. model_infer_params
# 
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params in this script.
app_infer_params = []

# 2. Model-specific params (Model: GraphDRP)
# All params in model_infer_params are optional.
# If no params are required by the model, then it should be an empty list.
model_infer_params = []

# [Req] Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
infer_params = app_infer_params + model_infer_params
# ---------------------


# [Req]
def run(params):
    """ Run model inference.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """
    # import ipdb; ipdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create output dir
    # ------------------------------------------------------
    frm.create_outdir(outdir="onnx_outdir")

    # ------------------------------------------------------
    # [Req] Create data names for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_name(params, stage="test")

    # GraphDRP -- remove data_format
    test_data_fname = test_data_fname.split(params["data_format"])[0]

    # ------------------------------------------------------
    # Prepare dataloaders to load model input data (ML data)
    # ------------------------------------------------------
    print("\nTest data:")
    print(f"test_ml_data_dir: {params['test_ml_data_dir']}")
    print(f"test_batch: {params['test_batch']}")
    test_loader = build_GraphDRP_dataloader(params["test_ml_data_dir"],
                                            test_data_fname,
                                            params["test_batch"],
                                            shuffle=False)

    # ------------------------------------------------------
    # CUDA/CPU device
    # ------------------------------------------------------
    # Determine CUDA/CPU device and configure CUDA device if available
    device = determine_device(params["cuda_name"])

    torchModel = set_GraphDRP(params, device)
    torchModel.eval()

    class GNN(GroqWrapper, globals()[params['model_arch']]):
        pass
    
    if params['model_arch']=="GCNNet":
        groqModel = GNN(dropout=0.5).to(device)
    else:
        groqModel = GNN().to(device)
    print(torchModel)
    print(groqModel)
    # Compute predictions
    onnx_export(groqModel, torchModel, device, data_loader=test_loader)


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        # default_model="graphdrp_default_model.txt",
        # default_model="graphdrp_params.txt",
        # default_model="params_ws.txt",
        default_model="params_cs.txt",
        additional_definitions=additional_definitions,
        # required=req_infer_args,
        required=None,
    )
    run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
