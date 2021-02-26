import torch
from torch import nn
import torch_sparse
from typing import Tuple, Optional


class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.weight = nn.Parameter(torch.tensor(0.))

    def forward(self, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
            adj_t: SparseTensor shape (num_nodes, num_nodes)
            edges: Tensor shape (2, num_query_edges)
        Outputs:
            prediction: Tensor shape (num_query_edges,) with scores between 0 and 1 for each edge in `edges`.
        '''
        return torch.zeros(edges.shape[1], dtype=torch.float32, device=self.weight.device) * self.weight


def get_model(model_name: str) -> type:
    models = [GNN]
    for m in models:
        if m.__name__ == model_name:
            return m
    assert False, f'Could not find model {model_name}!'
