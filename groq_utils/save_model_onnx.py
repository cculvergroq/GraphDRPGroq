# TODO saving an ONNX model should be minimal overhead 
# probably makes sense to add this to the checkpointing system

import sys
import os
from pathlib import Path, PurePath
from typing import Dict

import torch
import pandas as pd

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from runner import ModelRunner


# [Req] Imports from preprocess and train scripts
from graphdrp_preprocess_improve import preprocess_params
from graphdrp_train_improve import metrics_list, train_params

filepath = Path(__file__).resolve().parent.parent # [Req]

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
    model_runner = ModelRunner(params)

    dummy_batch = next(iter(model_runner.data_loader)).to(model_runner.device)
    #print(dummy_batch)
    dummy_data = dummy_batch.get_example(0)
    input_names=['x', 'edge_index', 'batch', 'target']
    dummy_inputs=(dummy_data.x, dummy_data.edge_index, dummy_data.batch, dummy_data.target)
    
    output_names=['out','x']
    save_file = PurePath.joinpath(Path.cwd(), 'out', 'infer.onnx')
    torch.onnx.export(
        model_runner.model, 
        dummy_inputs, 
        save_file, 
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
    )
    


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
