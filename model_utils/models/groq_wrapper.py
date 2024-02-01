import torch_geometric

class GroqWrapper(object):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super().__init__(n_output, num_features_xd, num_features_xt, n_filters, embed_dim, output_dim, dropout)
        
    def forward(self, x, edge_index, batch, target):
        return super().forward(
            torch_geometric.data.Data(
                x=x, 
                edge_index=edge_index,
                batch=batch,
                target=target
            )
        )