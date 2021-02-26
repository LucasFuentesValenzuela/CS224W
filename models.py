import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
from typing import Tuple, Optional
# Some built in PyG layers
from torch_geometric.nn import GCNConv


# Graph Convolutional Neural Network
class GCN(nn.Module):
    def __init__(self, data, args):
        super().__init__()

        # architecture
        input_dim = data.num_features
        hidden_dim = args.hidden_dim
        output_dim = args.output_dim
        num_layers = args.num_layers

        # layers
        conv_layers = [GCNConv(input_dim, hidden_dim)] + \
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)] + \
            [GCNConv(hidden_dim, output_dim)]
        self.convs = conv_layers

        bns_layers = [nn.BatchNorm1d(num_features=hidden_dim)
                      for _ in range(num_layers)]
        self.bns = nn.ModuleList(bns_layers)
        self.dropout = args.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: torch.Tensor, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
            x: Tensor shape (num_nodes, initial_embedding_dim)
            adj_t: SparseTensor shape (num_nodes, num_nodes)
            edges: Tensor shape (2, num_query_edges)
        Outputs:
            prediction: Tensor shape (num_query_edges,) with scores between 0 and 1 for each edge in `edges`.
        '''

        # Building new node embeddings with GCNConv layers
        for k in range(len(self.convs)-1):
            x = self.convs[k](x, adj_t)
            x = self.bns[k](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        # Decode
        # source nodes embeddings, shape (num_query_edges, final_embedding_dim)
        x_s = x[edges[0, :]]
        # target nodes embeddings, shape (num_query_edges, final_embedding_dim)
        x_t = x[edges[1, :]]
        out = torch.sum(x_s*x_t, (1))  # dot product decoder
        return torch.sigmoid(out)  # cast values between 0 and 1


def get_model(model_name: str) -> type:
    models = [GCN]
    for m in models:
        if m.__name__ == model_name:
            return m
    assert False, f'Could not find model {model_name}!'
