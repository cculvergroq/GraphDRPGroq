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
import os
from pathlib import Path
from typing import Dict

import pandas as pd

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# [Req] Imports from preprocess and train scripts
from graphdrp_preprocess_improve import preprocess_params
from graphdrp_train_improve import train_params

from runner import VerifyWrapper

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
    model_runner = VerifyWrapper(params)
    test_scores = model_runner.run_predictions()
    print("\nFinished model inference.")
    print("  test_scores = ", test_scores)


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
