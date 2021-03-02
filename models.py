import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
from typing import Tuple, Optional
# Some built in PyG layers
from torch_geometric.nn import GCNConv, GATConv


# Graph Convolutional Neural Network
class GCN(nn.Module):
    def __init__(
        self,
        embedding_shape: Tuple[int, int],
        embedding_dim=256,
        hidden_dim=256,
        output_dim=256,
        num_layers=2,
        dropout=0.5,
    ):
        super().__init__()

        # architecture
        input_dim = embedding_shape[1]
        self.dropout = dropout

        # layers
        self.embedding = nn.Embedding(
            embedding_shape[0], embedding_dim - input_dim)

        conv_layers = [GCNConv(embedding_dim, hidden_dim)] + \
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)] + \
            [GCNConv(hidden_dim, output_dim)]
        self.convs = nn.ModuleList(conv_layers)

        bns_layers = [nn.BatchNorm1d(num_features=hidden_dim)
                      for _ in range(num_layers)]
        self.bns = nn.ModuleList(bns_layers)

        #predictor
        self.predictor = LinkPredictor(output_dim, hidden_dim, 1, num_layers, self.dropout)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: torch.Tensor, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
            x: Tensor shape (num_nodes, input_dim)
            adj_t: SparseTensor shape (num_nodes, num_nodes)
            edges: Tensor shape (2, num_query_edges)
        Outputs:
            prediction: Tensor shape (num_query_edges,) with scores between 0 and 1 for each edge in `edges`.
        '''

        # Initial embedding lookup
        # shape num_nodes, embedding_dim
        x = torch.cat([self.embedding.weight, x], dim=1)

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

        return self.predictor(x_s, x_t)

# Graph Attention Networks

class GAT(nn.Module):
    def __init__(
        self,
        embedding_shape: Tuple[int, int],
        embedding_dim=256,
        hidden_dim=256,
        output_dim=256,
        num_layers=2,
        dropout=0.5,
        heads=1,
        concat=True,
        negative_slope=.2,
        bias=True,
    ):
        super().__init__()

        # architecture
        input_dim = embedding_shape[1]
        self.dropout = dropout

        # multiplicative factor if multiple heads are used
        if concat:
            mult_factor = heads
        else:
            mult_factor = 1

        # layers
        self.embedding = nn.Embedding(
            embedding_shape[0], embedding_dim - input_dim)

        #TODO: decide on whether to use concat for other layers than the first one
        conv_layers = [
            GATConv(embedding_dim, hidden_dim,
            heads = heads, concat = concat, negative_slope = negative_slope,
            dropout = self.dropout, bias = bias)
            ] + \
            [GATConv(mult_factor*hidden_dim, mult_factor*hidden_dim, heads = heads,
            concat = False, negative_slope = negative_slope,
            dropout = self.dropout, bias = bias)
            for _ in range(num_layers-2)] + \
            [GATConv(mult_factor*hidden_dim, output_dim, heads = heads, concat = False,
            negative_slope = negative_slope,
            dropout = self.dropout, bias = bias)]

        self.convs = nn.ModuleList(conv_layers)

        bns_layers = [nn.BatchNorm1d(num_features=mult_factor*hidden_dim)
                      for _ in range(num_layers)]
        self.bns = nn.ModuleList(bns_layers)

        #predictor
        self.predictor = LinkPredictor(output_dim, hidden_dim, 1, num_layers, self.dropout)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: torch.Tensor, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
            x: Tensor shape (num_nodes, input_dim)
            adj_t: SparseTensor shape (num_nodes, num_nodes)
            edges: Tensor shape (2, num_query_edges)
        Outputs:
            prediction: Tensor shape (num_query_edges,) with scores between 0 and 1 for each edge in `edges`.
        '''

        # Initial embedding lookup
        # shape num_nodes, embedding_dim
        x = torch.cat([self.embedding.weight, x], dim=1)

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

        return self.predictor(x_s, x_t)


def get_model(model_name: str) -> type:
    models = [GCN, GAT]
    for m in models:
        if m.__name__ == model_name:
            return m
    assert False, f'Could not find model {model_name}!'


################
# Predictors
################

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x).flatten()
