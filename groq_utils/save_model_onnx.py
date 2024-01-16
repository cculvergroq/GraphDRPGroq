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

# Model-specific imports
from model_utils.torch_utils import (
    build_GraphDRP_dataloader,
    determine_device,
    load_GraphDRP,
    predicting,
)
from model_utils.models.ginconv import GroqGINConvNet

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
def load(params):
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
    frm.create_outdir(outdir=params["infer_outdir"])

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

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)
    modelpath = frm.build_model_path(params, model_dir=params["model_dir"]) # [Req]
    model = load_GraphDRP(params, modelpath, device)
    model.eval()
    
    groqModel = GroqGINConvNet()
    groqModel.load_state_dict(model.state_dict())
    print("TrainedModel")
    print(model.state_dict())
    print("GroqModel")
    print(groqModel.state_dict())
    msd = model.state_dict()
    gsd = groqModel.state_dict()
    converted=True
    for key in msd:
        if not torch.allclose(msd[key], gsd[key]):
            print("State dict error at {}".format(key))
            converted=False
    if not converted:
        raise ValueError("State dicts are not the same for the Groq model")
    dummy_batch = next(iter(test_loader)).to(device)
    #print(dummy_batch)
    dummy_data = dummy_batch.get_example(0)
    #print(type(dummy_data))
    #print(dummy_data)
    
    return groqModel, dummy_batch

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
    model, dummy_data = load(params)

    dummy_data=dummy_data 
    input_names=['x', 'edge_index', 'batch', 'target']
    dummy_inputs=[dummy_data.x, dummy_data.edge_index, dummy_data.batch, dummy_data.target]
    
    output_names=['out','x']
    save_file = PurePath.joinpath(Path.cwd(), 'out', 'infer.onnx')
    torch.onnx.export(
        model, 
        dummy_inputs, 
        save_file, 
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
    )
    
   


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
