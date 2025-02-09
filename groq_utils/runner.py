import torch
import onnx 
import onnxruntime as ort

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
import numpy as np

import sys
import os

from timer import Timer

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Model-specific imports
from model_utils.torch_utils import (
    build_GraphDRP_dataloader,
    determine_device,
    load_GraphDRP,
)
from model_utils.models.ginconv import GINConvNet
from model_utils.models.gcn import GCNNet
from model_utils.models.groq_wrapper import GroqWrapper
from graphdrp_train_improve import metrics_list

from utils import get_padded_data

# TODO: When padding don't allocate the tensory memory
# every time, just overwrite the values from the new graph

class BaseRunner:
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
        class GNN(GroqWrapper, globals()[self.params['model_arch']]):
            pass
        
        groqModel = GNN().to(self.device)
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
        raise NotImplementedError("Haven't implemented run_predictions for ", type(self))


class ModelRunner(BaseRunner):
    def run_predictions(self):
        total_labels=torch.Tensor()
        total_preds=torch.Tensor()

        print("Comparing all {} samples for both models".format(len(self.data_loader.dataset)))
        timer_results = {'to-gpu': [], 'compute': [], 'to-cpu': []}
        to_gpu_timer = Timer('to-gpu')
        compute_timer = Timer('compute')
        to_cpu_timer = Timer('to-cpu')
        self.model.half()
        with torch.no_grad():
            for i,data in enumerate(self.data_loader):
                x, edge_index, batch, target = get_padded_data(data)
                x=x.half()
                target=target.half()
                
                # checked that x is on CPU with x.get_device()
                to_gpu_timer.start()
                x=x.to(self.device)
                edge_index=edge_index.to(self.device)
                batch=batch.to(self.device)
                target=target.to(self.device)
                timer_results['to-gpu'].append(to_gpu_timer.stop())
                
                compute_timer.start()
                outputGPU, _ = self.model(x, edge_index, batch, target)
                timer_results['compute'].append(compute_timer.stop())
                
                to_cpu_timer.start()
                outputCPU = outputGPU.cpu()
                timer_results['to-cpu'].append(to_cpu_timer.stop())
                
                total_labels = torch.cat((total_labels, data.y.view(-1,1).cpu()), 0)
                total_preds = torch.cat((total_preds, torch.tensor(outputCPU[0])), 0)
        
        for key,value in timer_results.items():
            timer_results[key]=np.array(value)
        np.savez('out/timer_results_torch.npz', **timer_results)
        print("All timing saved to out/timer_results_torch.npz")
        print("Average timings (ns):")
        for key,value in timer_results.items():
            print("{}: {:.2f} +/- {:.2f} us".format(key, value.mean(), value.std()))
        
        total_labels=total_labels.numpy().flatten()
        total_preds=total_preds.numpy().flatten()
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


class OnnxRunner(BaseRunner):
    def run_predictions(self):
        """ Run predictions
        
        """
        # Compute predictions
        total_labels=torch.Tensor()
        total_preds=torch.Tensor()
        ort_session = ort.InferenceSession(self.params['onnx_name'], providers=['TensorrtExecutionProvider'])
        
        print("Make prediction for {} samples...".format(len(self.data_loader.dataset)))
        timer_results = {'end-to-end': []}
        etoe_timer = Timer('end-to-end')
        with torch.no_grad():
            for i,data in enumerate(self.data_loader):
                x, edge_index, batch, target = get_padded_data(data)

                etoe_timer.start()
                outputs = ort_session.run(["out", "xOut"], {
                    "x": x.numpy(),
                    "edge_index": edge_index.numpy(),
                    "batch": batch.numpy(), 
                    "target": target.numpy(),
                })
                timer_results['end-to-end'].append(etoe_timer.stop())
                
                total_labels = torch.cat((total_labels, data.y.view(-1,1).cpu()), 0)
                total_preds = torch.cat((total_preds, torch.tensor(outputs[0][0])), 0)
                
        for key,value in timer_results.items():
            timer_results[key]=np.array(value)
        np.savez('out/timer_results_onnx.npz', **timer_results)
        print("All timing saved to out/timer_results_onnx.npz")
        print("Average timings (ns):")
        for key,value in timer_results.items():
            print("{}: {:.2f} +/- {:.2f} us".format(key, value.mean(), value.std()))        
        
        total_labels=total_labels.numpy().flatten()
        total_preds=total_preds.numpy().flatten()
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


class OnnxVerifier(BaseRunner):
    def run_predictions(self):
        """ Run predictions
        
        """
        # Compute predictions
        total_labels=torch.Tensor()
        total_preds=torch.Tensor()
        ort_session = ort.InferenceSession(self.params['onnx_name'], providers=['TensorrtExecutionProvider'])
        
        print("Make prediction for {} samples...".format(len(self.data_loader.dataset)))
        with torch.no_grad():
            for i,data in enumerate(self.data_loader):
                x, edge_index, batch, target = get_padded_data(data)

                outputs = ort_session.run(["out", "xOut"], {
                    "x": x.numpy(),
                    "edge_index": edge_index.numpy(),
                    "batch": batch.numpy(), 
                    "target": target.numpy(),
                })
                
                x=x.to(self.device)
                edge_index=edge_index.to(self.device)
                batch=batch.to(self.device)
                target=target.to(self.device)
                outputPad, _ = self.model(x, edge_index, batch, target)
                
                if not np.isclose(outputPad.cpu().numpy()[0], outputs[0][0]):
                    print(i, outputPad.cpu().numpy().flatten(), "   ", outputs[0].flatten())
                    raise ValueError("ONNX runtime doesn't match torch runtime")
                total_labels = torch.cat((total_labels, data.y.view(-1,1).cpu()), 0)
                total_preds = torch.cat((total_preds, torch.tensor(outputs[0][0])), 0)
                
        total_labels=total_labels.numpy().flatten()
        total_preds=total_preds.numpy().flatten()
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
    

# from groq.runner import tsp   
class GroqRunner:
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
                x=data.x
                x=torch.zeros((70,78),dtype=torch.float16)
                x[:data.x.size(0),:data.x.size(1)]=data.x
                
                maxVertex=torch.max(data.edge_index)+1
                maxVertexPossible=70
                ###no self connections if original graph has self connections
                if maxVertex<maxVertexPossible:
                    edge_index=torch.randint(low=64, high=70, size=(2,150), dtype=torch.int64)
                    edge_index[:data.edge_index.size(0),:data.edge_index.size(1)]=data.edge_index
                else:
                    edge_index=data.edge_index
                    print("no edge padding, edge_index.shape={}, x.shape={}".format(edge_index.shape, data.x.shape))
                
                #edge_index=data.edge_index

                batch=torch.ones((70),dtype=torch.int64)
                batch[:data.batch.size(0)]=data.batch
                #batch=torch.ones(1,dtype=torch.int64)
                #batch=torch.zeros(1,dtype=torch.int64)  
                target=torch.zeros((2,958),dtype=torch.float16)
                target[:data.target.size(0), :data.target.size(1)]=data.target

                outputs = self.model(
                    x=x,
                    edge_index=edge_index,
                    batch=batch,
                    target=target
                )
                print(outputs[0][0])
                total_labels = torch.cat((total_labels, data.y.view(-1,1).cpu()), 0)
                total_preds = torch.cat((total_preds, torch.tensor(outputs[0][0])), 0)
                
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
    
    
class VerifyWrapper(BaseRunner):
    def run_predictions(self):
        # Lets rebuild the original GraphDRP calling convention - reminder the class holds GroqGINConvnet
        modelpath = frm.build_model_path(self.params, model_dir=self.params["model_dir"]) 
        base_model = load_GraphDRP(self.params, modelpath, self.device)
        base_model.eval()
        
        total_labels=torch.Tensor()
        total_preds=torch.Tensor()

        print("Comparing all {} samples for both models".format(len(self.data_loader.dataset)))
        with torch.no_grad():
            for i,data in enumerate(self.data_loader):
                x, edge_index, batch, target = get_padded_data(data)
                
                x=x.to(self.device)
                edge_index=edge_index.to(self.device)
                batch=batch.to(self.device)
                target=target.to(self.device)
                outputPad, _ = self.model(x, edge_index, batch, target)
                
                data.to(self.device)
                baseOut, _base = base_model(data)
                if not torch.isclose(baseOut.cpu()[0], outputPad.cpu()[0][0]):
                    raise ValueError("Groq model with padding has different prediction")
                
                total_labels = torch.cat((total_labels, data.y.view(-1,1).cpu()), 0)
                total_preds = torch.cat((total_preds, torch.tensor(outputPad.cpu()[0])), 0)
        print("Success, Groq model with padding verified!")
        total_labels=total_labels.numpy().flatten()
        total_preds=total_preds.numpy().flatten()
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