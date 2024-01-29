import torch

def get_padded_data(data):
    # TODO: hardcoded from ONNX save script output...
    # BATCH=1 : x=[62,78], edge=[2,136], batch=[62], target=[1,958]
    x=data.x
    x=torch.zeros((64,78),dtype=torch.float32)
    x[:data.x.size(0),:data.x.size(1)]=data.x
    
    ###no self connections if original graph has self connections
    edge_index=torch.randint(low=62, high=64, size=(2,137), dtype=torch.int64)
    edge_index[:data.edge_index.size(0),:data.edge_index.size(1)]=data.edge_index

    batch=torch.ones((64),dtype=torch.int64)
    batch[:data.batch.size(0)]=data.batch

    target=torch.zeros((2,958),dtype=torch.float32)
    target[:data.target.size(0), :data.target.size(1)]=data.target
    
    return x, edge_index, batch, target