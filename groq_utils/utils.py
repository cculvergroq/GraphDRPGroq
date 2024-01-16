import torch

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Model-specific imports
from model_utils.torch_utils import (
    build_GraphDRP_dataloader,
    determine_device,
    load_GraphDRP,
    predicting,
)
from model_utils.models.ginconv import GroqGINConvNet
from graphdrp_train_improve import metrics_list

class ModelRunner:
    def __init__(self, params):
        self.params=params
        self.device=self.get_device()
        self.model=self.load_model()
        self.data_loader=self.load_infer_data()
        
    def get_device(self):
        return determine_device(self.params["cuda_name"])

    # [Req]
    def load_model(self):
        """ Load model.

        Args:
            params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

        Returns:
            model: best trained model
        """

        # Load the best saved model (as determined based on val data)
        modelpath = frm.build_model_path(self.params, model_dir=self.params["model_dir"]) # [Req]
        model = load_GraphDRP(self.params, modelpath, self.device)
        model.eval()
        
        # Load the Groq version, which has a different forward function call 
        # to accomodate onnx exporting, exact same model though.
        ## TODO: This is a bit wasteful to load the model twice
        ##       Better way would be to instantiate the Groq model with trained weights directly
        groqModel = GroqGINConvNet().to(self.device)
        groqModel.load_state_dict(model.state_dict())
        groqModel.eval()
        # print("TrainedModel")
        # print(model.state_dict())
        # print("GroqModel")
        # print(groqModel.state_dict())
        msd = model.state_dict()
        gsd = groqModel.state_dict()
        converted=True
        for key in msd:
            if not torch.allclose(msd[key], gsd[key]):
                print("State dict error at {}".format(key))
                converted=False
        if not converted:
            raise ValueError("State dicts are not the same for the Groq model")
        
        return groqModel

    def load_infer_data(self):
        """ Load inference data
        
        Args:
            params (dict): dict of CANDLE/IMPROVE parameters and parsed values.
            
        Returns:
            data: data for inference predicting
        """
        
        # ------------------------------------------------------
        # [Req] Create output dir
        # ------------------------------------------------------
        frm.create_outdir(outdir=self.params["infer_outdir"])

        # ------------------------------------------------------
        # [Req] Create data names for test set
        # ------------------------------------------------------
        test_data_fname = frm.build_ml_data_name(self.params, stage="test")

        # GraphDRP -- remove data_format
        test_data_fname = test_data_fname.split(self.params["data_format"])[0]

        # ------------------------------------------------------
        # Prepare dataloaders to load model input data (ML data)
        # ------------------------------------------------------
        print("\nTest data:")
        print(f"test_ml_data_dir: {self.params['test_ml_data_dir']}")
        print(f"test_batch: {self.params['test_batch']}")
        test_loader = build_GraphDRP_dataloader(self.params["test_ml_data_dir"],
                                                test_data_fname,
                                                self.params["test_batch"],
                                                shuffle=False)
        
        return test_loader


    def run_predictions(self):
        """ Run predictions
        
        """
        # Compute predictions
        test_true, test_pred = predicting(self.model, self.device, data_loader=self.data_loader)

        # ------------------------------------------------------
        # [Req] Save raw predictions in dataframe
        # ------------------------------------------------------
        frm.store_predictions_df(
            self.params,
            y_true=test_true, y_pred=test_pred, stage="test",
            outdir=self.params["infer_outdir"]
        )

        # ------------------------------------------------------
        # [Req] Compute performance scores
        # ------------------------------------------------------
        test_scores = frm.compute_performace_scores(
            self.params,
            y_true=test_true, y_pred=test_pred, stage="test",
            outdir=self.params["infer_outdir"], metrics=metrics_list
        )

        return test_scores