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
from runner import BaseRunner

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
        default_model="params_cs.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    
    model_runner = BaseRunner(params)
    model_runner.model.eval()
    data0 = next(iter(model_runner.data_loader))
    
    input_tensors = [data0.x, data0.edge_index, data0.batch, data0.target]
    maxSizeTensors = input_tensors
    # TODO: this is temporary - NEED to understand model to understand the true max that could be passed
    # in - this only works for THIS inference input set...
    shapeMax=[[0 for i in range(d.dim())] for d in input_tensors]
    maxEdgeIdx=0
    
    for data in model_runner.data_loader:
        # get max edge_index value
        for edges in data.edge_index:
            for edge in edges:
                if edge>maxEdgeIdx:
                    maxEdgeIdx=edge

        tensors=[data.x, data.edge_index, data.batch, data.target]
        for i,tensor in enumerate(tensors):
            for j in range(tensor.dim()):
                new_max=max(shapeMax[i][j],tensor.size(j))
                if new_max!=shapeMax[i][j]:
                    shapeMax[i][j]=new_max
                    maxSizeTensors[i]=tensor

    input_names=['x', 'edge_index', 'batch', 'target']

    # the paddedTensors is a list of [x,edge_index,batch,target] and each of these is padded as follows
    # x[0] -> number of nodes -> add 2 nodes - features are all set to zero
    # x[1] -> number of features on a node -> no padding
    # edge_index[0] -> just formats that an edge has two nodes no padding
    # edge_index[1] -> add 1 extra edge
    # batch -> x[0]+2 - set to ones - so that we imitate a batch with the extra nodes/edges 
    # target[0] -> 2 - because there's the graph we care about and our dummy batch
    # target[1] -> no padding - initialized to zero.
    print(shapeMax)
    paddedTensors=[torch.zeros((shapeMax[0][0]+2,shapeMax[0][1]), dtype=torch.float32),
                    torch.randint(low=shapeMax[0][0],high=shapeMax[0][0]+2,size=(2,shapeMax[1][1]+1), dtype=torch.int64),
                    torch.ones((shapeMax[0][0]+2), dtype=torch.int64),
                    torch.zeros((2,shapeMax[3][1]), dtype=torch.float32)]
    
    paddedTensors[0][:maxSizeTensors[0].size(0),:maxSizeTensors[0].size(1)]=maxSizeTensors[0]
    paddedTensors[1][:maxSizeTensors[1].size(0),:maxSizeTensors[1].size(1)]=maxSizeTensors[1]
    paddedTensors[2][:maxSizeTensors[2].size(0)]=maxSizeTensors[2]
    paddedTensors[3][:maxSizeTensors[3].size(0),:maxSizeTensors[3].size(1)]=maxSizeTensors[3]
    
    dummy_inputs=tuple(tensor.to(model_runner.device) for tensor in paddedTensors)
    
    output_names=['out','xOut']
    save_file = PurePath.joinpath(Path.cwd(), 'out', 'infer.onnx')
    torch.onnx.export(
        model_runner.model, 
        dummy_inputs, 
        str(save_file), 
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
    )
    

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
