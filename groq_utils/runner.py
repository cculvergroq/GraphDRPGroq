import torch
import onnx 
import onnxruntime as ort

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
        self.model.eval()
        total_labels=torch.Tensor()
        total_preds=torch.Tensor()
        print("Make prediction for {} samples...".format(len(self.data_loader.dataset)))
        with torch.no_grad():
            for data in self.data_loader:
                #print("type(data)=",type(data))
                #print(len(data.to_data_list()))
                #print(type(data.to_data_list()[0]))
                #print("data.batch=", data.batch)
                torch.set_printoptions(profile="full")
                
                # TODO: hardcoded from ONNX save script output...
                # BATCH=1 : x=[62,78], edge=[2,136], batch=[62], target=[1,958]
                x=data.x
                x=torch.zeros((70,78),dtype=torch.float32)
                x[:data.x.size(0),:data.x.size(1)]=data.x
                
                maxVertex=torch.max(data.edge_index)+1
                maxVertexPossible=x.size(0)
                ###no self connections if original graph has self connections
                if maxVertex<maxVertexPossible:
                    edge_index=torch.randint(low=maxVertex, high=maxVertexPossible, size=(2,150), dtype=torch.int64)
                    edge_index[:data.edge_index.size(0),:data.edge_index.size(1)]=data.edge_index
                else:
                    edge_index=data.edge_index
                    print("no edge padding, edge_index.shape={}, x.shape={}".format(edge_index.shape, data.x.shape))
                
                edge_index=data.edge_index

                batch=torch.ones((70),dtype=torch.int64)
                batch[:data.batch.size(0)]=data.batch
                #batch=torch.ones(1,dtype=torch.int64)
                #batch=torch.zeros(1,dtype=torch.int64)  
                target=torch.zeros((2,958),dtype=torch.float32)
                target[:data.target.size(0), :data.target.size(1)]=data.target
                #print("x=",x)
                #print("datalistx=",data.to_data_list()[0].x)
                #print("edge=",edge_index)
                #print("datalistedge=",data.to_data_list()[0].edge_index)
                #print("batch=",batch)
                #print("databatch=",data.to_data_list()[0].batch)
                #print("target=",target)
                #print("datatarget=",data.to_data_list()[0].target)
                
                # the below produce [0.9149288, 0.9257006] for batch 2 
                # and correctly does [0.9149288] for batch 1...               
                # x=data.x
                # edge_index=data.edge_index
                # batch=data.batch
                # target=data.target
                
                x=x.to(self.device)
                edge_index=edge_index.to(self.device)
                batch=batch.to(self.device)
                target=target.to(self.device)
                outputPad, _ = self.model(x, edge_index, batch, target)
                data=data.to(self.device)
                output, _ = self.model(data.x, data.edge_index, data.batch, data.target)
                
                print("Padded = {}, Normal = {}".format(outputPad.cpu(), output.cpu()))
                #data=data.to(self.device)
                #output, _ = self.model(data.x, data.edge_index, data.batch, data.target)
                #print([output, _])
                total_labels = torch.cat((total_labels, data.y.view(-1,1).cpu()), 0)
                total_preds = torch.cat((total_preds, outputPad.cpu()[0]), 0)
                #break
                #expected mse=0.0102783227
                
        total_labels=total_labels.numpy().flatten()
        total_preds=total_preds.numpy().flatten()
        print(total_preds)
        # ------------------------------------------------------
        # [Req] Save raw predictions in dataframe
        # ------------------------------------------------------
        frm.store_predictions_df(
            self.params,
            y_true=total_labels, y_pred=total_preds, stage="test",
            outdir=self.params["infer_outdir"]
        )

        # ------------------------------------------------------
        # [Req] Compute performance scores
        # ------------------------------------------------------
        test_scores = frm.compute_performace_scores(
            self.params,
            y_true=total_labels, y_pred=total_preds, stage="test",
            outdir=self.params["infer_outdir"], metrics=metrics_list
        )

        return test_scores
    

    
class OnnxRunner:
    def __init__(self, params):
        self.params=params
        self.data_loader=self.load_infer_data()
        self.model=self.load_model()
    
    def load_model(self):
        onnx_model = onnx.load(self.params['onnx_name'])
        onnx.checker.check_model(onnx_model)
        self.model=onnx_model
    
    def get_device(self):
        return determine_device(self.params["cuda_name"])
    
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
        total_labels=torch.Tensor()
        total_preds=torch.Tensor()
        ort_session = ort.InferenceSession(self.params['onnx_name'])
        print("Make prediction for {} samples...".format(len(self.data_loader.dataset)))
        with torch.no_grad():
            for data in self.data_loader:
            # for batch in self.data_loader:
            #    for data in batch:
            #       do stuff
                
                # data=data.to(self.device)
                # output, _ = self.model(data.x, data.edge_index, data.batch, data.target)
                
                # TODO: hardcoded from ONNX save script output...
                x=torch.zeros((124,78),dtype=torch.float32)
                x[:data.x.size(0),:data.x.size(1)]=data.x
                edge_index=torch.zeros((2,272),dtype=torch.int64)
                edge_index[:data.edge_index.size(0),:data.edge_index.size(1)]=data.edge_index
                batch=torch.zeros((124),dtype=torch.int64)
                batch[:data.batch.size(0)]=data.batch
                target=torch.zeros((2,958),dtype=torch.float32)
                target[:data.target.size(0), :data.target.size(1)]=data.target
                torch.set_printoptions(profile="full")
                print("x=",x)
                print("edge=",edge_index)
                print("batch=",batch)
                print("target=",target)
                outputs = ort_session.run(["out", "xOut"], {
                    "x": x.numpy(),
                    "edge_index": edge_index.numpy(),
                    "batch": batch.numpy(), 
                    "target": target.numpy(),
                })
                print(outputs)
                total_labels = torch.cat((total_labels, data.y.view(-1,1).cpu()), 0)
                total_preds = torch.cat((total_preds, torch.tensor(outputs[0][:data.target.size(0)])), 0)
                break
        total_labels=total_labels.numpy().flatten()
        total_preds=total_preds.numpy().flatten()
        print(total_preds)
        # ------------------------------------------------------
        # [Req] Save raw predictions in dataframe
        # ------------------------------------------------------
        frm.store_predictions_df(
            self.params,
            y_true=total_labels, y_pred=total_preds, stage="test",
            outdir=self.params["infer_outdir"]
        )

        # ------------------------------------------------------
        # [Req] Compute performance scores
        # ------------------------------------------------------
        test_scores = frm.compute_performace_scores(
            self.params,
            y_true=total_labels, y_pred=total_preds, stage="test",
            outdir=self.params["infer_outdir"], metrics=metrics_list
        )

        return test_scores