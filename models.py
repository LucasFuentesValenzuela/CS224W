import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
from typing import Tuple, Optional
# Some built in PyG layers
from torch_geometric.nn import GCNConv, GATConv
from predictors import LinkPredictor, MADpredictor
from global_attention_layer import LowRankAttention, weight_init

# TODO: make sure that the MAD predictor is permutation invariant. That is the prediction for (u,v) should
# be the same as that for (v,u)
# TODO: in MAD, are we giving the right edges to the predictor? does it not need access to all edges every time?


def get_model(model_name: str) -> type:
    models = [GCN, GAT, MAD, MAD_GCN, GCN_LRGA, MAD_GCN_LRGA]
    for m in models:
        if m.__name__ == model_name:
            return m
    assert False, f'Could not find model {model_name}!'

# Graph Convolutional Neural Network


class GCN(nn.Module):
    def __init__(
        self,
        embedding_shape: Tuple[int, int],
        adj_t: torch_sparse.SparseTensor,
        embedding_dim=256,
        hidden_dim=256,
        output_dim=256,
        num_layers=2,
        dropout=0.5,
        cache=True,
    ):
        super().__init__()

        # architecture
        self.dropout = dropout

        # layers
        self.embedding = nn.Embedding(
            embedding_shape[0], embedding_dim)

        nn.init.xavier_uniform_(self.embedding.weight)

        conv_layers = [GCNConv(embedding_dim, hidden_dim, cached=cache)] + \
            [GCNConv(hidden_dim, hidden_dim, cached=cache) for _ in range(num_layers-2)] + \
            [GCNConv(hidden_dim, output_dim, cached=cache)]
        self.convs = nn.ModuleList(conv_layers)

        bns_layers = [nn.BatchNorm1d(num_features=hidden_dim)
                      for _ in range(num_layers)]
        self.bns = nn.ModuleList(bns_layers)

        # predictor
        self.predictor = LinkPredictor(
            output_dim, hidden_dim, 1, num_layers, self.dropout)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
            adj_t: SparseTensor shape (num_nodes, num_nodes)
            edges: Tensor shape (2, num_query_edges)
        Outputs:
            prediction: Tensor shape (num_query_edges,) with scores between 0 and 1 for each edge in `edges`.
        '''

        # Initial embedding lookup
        # shape num_nodes, embedding_dim
        x = self.embedding.weight

        # Building new node embeddings with GCNConv layers
        for k in range(len(self.convs)-1):
            x = self.convs[k](x, adj_t)
            # x = self.bns[k](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        return self.predictor(x, edges)

# Graph Attention Networks


class GAT(nn.Module):
    def __init__(
        self,
        embedding_shape: Tuple[int, int],
        adj_t: torch_sparse.SparseTensor,
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
        self.dropout = dropout

        # multiplicative factor if multiple heads are used
        if concat:
            mult_factor = heads
        else:
            mult_factor = 1

        # layers
        self.embedding = nn.Embedding(
            embedding_shape[0], embedding_dim)

        # TODO: decide on whether to use concat for other layers than the first one
        conv_layers = [
            GATConv(embedding_dim, hidden_dim,
                    heads=heads, concat=concat, negative_slope=negative_slope,
                    dropout=self.dropout, bias=bias)
        ] + \
            [GATConv(mult_factor*hidden_dim, mult_factor*hidden_dim, heads=heads,
                     concat=False, negative_slope=negative_slope,
                     dropout=self.dropout, bias=bias)
             for _ in range(num_layers-2)] + \
            [GATConv(mult_factor*hidden_dim, output_dim, heads=heads, concat=False,
                     negative_slope=negative_slope,
                     dropout=self.dropout, bias=bias)]

        self.convs = nn.ModuleList(conv_layers)

        bns_layers = [nn.BatchNorm1d(num_features=mult_factor*hidden_dim)
                      for _ in range(num_layers)]
        self.bns = nn.ModuleList(bns_layers)

        # predictor
        self.predictor = LinkPredictor(
            output_dim, hidden_dim, 1, num_layers, self.dropout)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
            adj_t: SparseTensor shape (num_nodes, num_nodes)
            edges: Tensor shape (2, num_query_edges)
        Outputs:
            prediction: Tensor shape (num_query_edges,) with scores between 0 and 1 for each edge in `edges`.
        '''

        # Initial embedding lookup
        # shape num_nodes, embedding_dim
        x = self.embedding.weight

        # Building new node embeddings with GCNConv layers
        for k in range(len(self.convs)-1):
            x = self.convs[k](x, adj_t)
            x = self.bns[k](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        return self.predictor(x, edges)

# Memory Adaptive Differential Learning
# mainly from https://github.com/cf020031308/mad-learning


class MAD(nn.Module):
    def __init__(
        self,
        embedding_shape: Tuple[int, int],
        adj_t: torch_sparse.SparseTensor,
        embedding_dim=12,
        n_heads=12,
        n_samples=8,
        n_sentinels=8,
        n_nearest=8,
        field_NN=False
    ):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_nodes = embedding_shape[0]
        self.n_samples = n_samples
        self.n_heads = n_heads
        self.n_sentinels = n_sentinels
        self.n_nearest = n_nearest

        self.embeds = nn.Parameter(
            torch.rand((self.n_heads, self.n_nodes, embedding_dim)))
        self.predictor = MADpredictor(
            self.embedding_dim, adj_t, self.n_nodes, n_heads=self.n_heads,
            n_samples=self.n_samples, n_sentinels=self.n_sentinels,
            n_nearest=self.n_nearest, field_NN=field_NN).to(adj_t.device())

    def forward(self, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
            adj_t: SparseTensor shape (num_nodes, num_nodes)
            edges: Tensor shape (2, num_query_edges)
        Outputs:
            prediction: Tensor shape (num_query_edges,) with scores between 0 and 1 for each edge in `edges`.
        '''
        return self.predictor(self.embeds, edges)

    def reset_parameters(self):
        # TODO: implement?
        pass

# Combination of MAD and GCN-learned embeddings


class MAD_GCN(nn.Module):
    def __init__(
        self,
        embedding_shape: Tuple[int, int],
        adj_t: torch_sparse.SparseTensor,
        embedding_dim=256,
        n_heads=12,
        n_samples=8,
        n_sentinels=8,
        n_nearest=8,
        hidden_dim=256,
        output_dim=12,
        num_layers=1,
        dropout=0.5,
        cache=True,
        field_NN=False,
    ):

        super().__init__()
        # architecture
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.n_nodes = embedding_shape[0]
        self.n_samples = n_samples
        self.n_heads = n_heads
        self.n_sentinels = n_sentinels
        self.n_nearest = n_nearest

        self.output_dim = output_dim

        self.embeds = nn.Embedding(
            embedding_shape[0], embedding_dim*self.n_heads)
        nn.init.xavier_uniform_(self.embeds.weight)

        conv_layers = [GCNConv(embedding_dim*self.n_heads, hidden_dim*self.n_heads, cached=cache)] + \
            [GCNConv(hidden_dim*self.n_heads, hidden_dim*self.n_heads, cached=cache) for _ in range(num_layers-2)] + \
            [GCNConv(hidden_dim*self.n_heads, output_dim *
                     self.n_heads, cached=cache)]
        self.convs = nn.ModuleList(conv_layers)

        self.predictor = MADpredictor(
            output_dim, adj_t, self.n_nodes, n_heads=self.n_heads,
            n_samples=self.n_samples, n_sentinels=self.n_sentinels,
            n_nearest=self.n_nearest, field_NN=field_NN).to(adj_t.device())

        self.gcn_cache = None

    def forward(self, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
            adj_t: SparseTensor shape (num_nodes, num_nodes)
            edges: Tensor shape (2, num_query_edges)
        Outputs:
            prediction: Tensor shape (num_query_edges,) with scores between 0 and 1 for each edge in `edges`.
        '''
        if self.training:
            self.gcn_cache = None

        if self.gcn_cache == None:

            # Initial embedding lookup
            # shape num_nodes, embedding_dim
            x = self.embeds.weight

            # Building new node embeddings with GCNConv layers
            for k in range(len(self.convs)-1):
                x = self.convs[k](x, adj_t)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, adj_t)  # shape (n_nodes, output_dim * n_heads)

            x = torch.reshape(x, (self.n_nodes, self.output_dim, self.n_heads))
            x = x.permute(2, 0, 1)  # shape (n_heads, n_nodes, output_dim)

            if not self.training:
                self.gcn_cache = torch.clone(x.detach(), memory_format=torch.contiguous_format)

        if not self.training:
            x = self.gcn_cache

        return self.predictor(x, edges)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


# Low Rank Global Attention
# mainly from https://github.com/omri1348/LRGA
class GCN_LRGA(torch.nn.Module):
    def __init__(
        self,
        embedding_shape: Tuple[int, int],
        adj_t: torch_sparse.SparseTensor,
        embedding_dim=256,
        hidden_dim=256,
        output_dim=256,
        num_layers=2,
        dropout=.5,
        k=50,
        cache=True,
    ):
        '''
        k: rank of the low-rank approximation
        '''
        super().__init__()
        self.k = k
        self.embedding_dim = embedding_shape[0]
        self.hidden_dim = hidden_dim
        self.out_dim = output_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            embedding_shape[0], embedding_dim)

        # convolutional layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(embedding_dim, hidden_dim, cached=cache))
        # attention layer
        self.attention = torch.nn.ModuleList()
        self.attention.append(LowRankAttention(self.k, embedding_dim, dropout))
        # dimension_reduce #TODO: clarify
        self.dimension_reduce = torch.nn.ModuleList()
        self.dimension_reduce.append(nn.Sequential(nn.Linear(2*self.k + hidden_dim,
                                                             hidden_dim), nn.ReLU()))
        # batch normalization layers
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim)
                                 for _ in range(num_layers-1)])

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, cached=cache))
            self.attention.append(LowRankAttention(
                self.k, hidden_dim, dropout))
            self.dimension_reduce.append(nn.Sequential(nn.Linear(2*self.k + hidden_dim,
                                                                 hidden_dim)))
        self.dimension_reduce[-1] = nn.Sequential(nn.Linear(2*self.k + hidden_dim,
                                                            output_dim))
        self.dropout = dropout

        self.predictor = LinkPredictor(
            self.out_dim, self.hidden_dim, 1, self.num_layers, self.dropout)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for glob_attention in self.attention:
            glob_attention.apply(weight_init)
        for dim_reduce in self.dimension_reduce:
            dim_reduce.apply(weight_init)
        for batch_norm in self.bn:
            batch_norm.reset_parameters()

    def forward(self, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:

        # Initial embedding lookup
        # shape num_nodes, embedding_dim
        x = self.embedding.weight

        # TODO: make sure to understand all the steps here
        for i, conv in enumerate(self.convs[:-1]):
            x_local = F.relu(conv(x, adj_t))
            x_local = F.dropout(x_local, p=self.dropout,
                                training=self.training)
            x_global = self.attention[i](x)
            x = self.dimension_reduce[i](torch.cat((x_global, x_local), dim=1))
            x = F.relu(x)
            x = self.bn[i](x)
        x_local = F.relu(self.convs[-1](x, adj_t))
        x_local = F.dropout(x_local, p=self.dropout, training=self.training)
        x_global = self.attention[-1](x)
        x = self.dimension_reduce[-1](torch.cat((x_global, x_local), dim=1))

        return self.predictor(x, edges)



# Low Rank Global Attention
# mainly from https://github.com/omri1348/LRGA
class MAD_GCN_LRGA(torch.nn.Module):
    def __init__(
        self,
        embedding_shape: Tuple[int, int],
        adj_t: torch_sparse.SparseTensor,
        embedding_dim=256,
        hidden_dim=256,
        output_dim=12,
        num_layers=2,
        dropout=.5,
        k=50,
        cache=True,
        n_heads=12,
        n_samples=8,
        n_sentinels=8,
        n_nearest=8,
        field_NN=False
    ):
        '''
        k: rank of the low-rank approximation
        '''
        super().__init__()
        self.k = k
        self.embedding_dim = embedding_shape[0]
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            embedding_shape[0], embedding_dim)

        # convolutional layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(embedding_dim, hidden_dim, cached=cache))
        # attention layer
        self.attention = torch.nn.ModuleList()
        self.attention.append(LowRankAttention(self.k, embedding_dim, dropout))
        # dimension_reduce #TODO: clarify
        self.dimension_reduce = torch.nn.ModuleList()
        self.dimension_reduce.append(nn.Sequential(nn.Linear(2*self.k + hidden_dim,
                                                             hidden_dim), nn.ReLU()))
        # batch normalization layers
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim)
                                 for _ in range(num_layers-1)])

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, cached=cache))
            self.attention.append(LowRankAttention(
                self.k, hidden_dim, dropout))
            self.dimension_reduce.append(nn.Sequential(nn.Linear(2*self.k + hidden_dim,
                                                                 hidden_dim)))
        self.dimension_reduce[-1] = nn.Sequential(nn.Linear(2*self.k + hidden_dim,
                                                            output_dim))
        self.dropout = dropout

        self.predictor = MADpredictor(
            output_dim, adj_t, self.n_nodes, n_heads=self.n_heads,
            n_samples=self.n_samples, n_sentinels=self.n_sentinels,
            n_nearest=self.n_nearest, field_NN=field_NN).to(adj_t.device())

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for glob_attention in self.attention:
            glob_attention.apply(weight_init)
        for dim_reduce in self.dimension_reduce:
            dim_reduce.apply(weight_init)
        for batch_norm in self.bn:
            batch_norm.reset_parameters()
        self.predictor.reset_parameters()

    def forward(self, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:

        # Initial embedding lookup
        # shape num_nodes, embedding_dim
        x = self.embedding.weight

        # TODO: make sure to understand all the steps here
        for i, conv in enumerate(self.convs[:-1]):
            x_local = F.relu(conv(x, adj_t))
            x_local = F.dropout(x_local, p=self.dropout,
                                training=self.training)
            x_global = self.attention[i](x)
            x = self.dimension_reduce[i](torch.cat((x_global, x_local), dim=1))
            x = F.relu(x)
            x = self.bn[i](x)
        x_local = F.relu(self.convs[-1](x, adj_t))
        x_local = F.dropout(x_local, p=self.dropout, training=self.training)
        x_global = self.attention[-1](x)
        x = self.dimension_reduce[-1](torch.cat((x_global, x_local), dim=1))

        x = torch.clone(x, memory_format=torch.contiguous_format)

        return self.predictor(x, edges)