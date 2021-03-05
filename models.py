import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
from typing import Tuple, Optional
# Some built in PyG layers
from torch_geometric.nn import GCNConv, GATConv

# TODO: make sure that the MAD predictor is permutation invariant. That is the prediction for (u,v) should
# be the same as that for (v,u)
# TODO: in MAD, are we giving the right edges to the predictor? does it not need access to all edges every time?

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


class MAD(nn.Module):
    def __init__(
        self,
        embedding_shape: Tuple[int, int],
        embedding_dim=32,
        n_heads=4,
        n_samples=64,
        n_sentinels=8,
        n_nearest=8
        ):

        super().__init__()
        self.n_nodes = embedding_shape[0]
        self.n_samples = n_samples
        self.n_heads = n_heads
        self.n_sentinels = n_sentinels
        self.n_nearest = n_nearest

        self.embeds = nn.Parameter(
            torch.rand((self.n_heads, self.n_nodes, embedding_dim)))

        # self.uncertainty #TODO: figure it out
        self.predictor = MADpredictor(
            embedding_dim, self.n_nodes, n_heads=self.n_heads,
            n_samples=self.n_samples, n_sentinels=self.n_sentinels,
            n_nearest=self.n_nearest)

    def forward(self, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
            adj_t: SparseTensor shape (num_nodes, num_nodes)
            edges: Tensor shape (2, num_query_edges)
        Outputs:
            prediction: Tensor shape (num_query_edges,) with scores between 0 and 1 for each edge in `edges`.
        '''
        return self.predictor(self.embeds, edges)


def get_model(model_name: str) -> type:
    models = [GCN, GAT, MAD]
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

    def forward(self, x, edges):
        '''
        x: embeddings
        edges:
        '''

        # Decode
        # source nodes embeddings, shape (num_query_edges, final_embedding_dim)
        x_s = x[edges[0, :]]
        # target nodes embeddings, shape (num_query_edges, final_embedding_dim)
        x_t = x[edges[1, :]]

        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x).flatten()


class MADpredictor(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_nodes,
        n_heads=4,
        n_samples=256,
        n_sentinels=8,
        n_nearest=8
        ):
        '''
        in_channels: dimensions of the embeddings before prediction
        n_nodes: number of nodes in the graph
        n_heads: number of parallel models to run
        n_samples: number of reference points to sample #TODO: clarify
        n_sentinels: number of soft sentinels to sample to regularize the softmin function
        '''
        super().__init__()
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.n_heads = n_heads
        self.n_sentinels = n_sentinels
        self.n_nearest = n_nearest

        self.field = nn.Parameter(
            torch.rand((n_heads, n_nodes, embedding_dim)))

    def forward(self, embeds, batch_edges):
        '''
        embeds: embeddings for all the nodes in the graph (n_heads, n_nodes, n_features)
        edges: batch of edges to predict
        '''
        n_batch = batch_edges.shape[1]

        # TODO: make sure it is invariant to permutation!
        # Actually one way to make it permutation invariant is to
        # predict in "both directions", which I think is what they do

        # logits when the source is considered static
        src_logits, src_diff = self.build_logits(
            embeds, batch_edges, node_type='source')
        # logits when the target is considered static
        tgt_logits, tgt_diff = self.build_logits(
            embeds, batch_edges, node_type='target')

        # logits now is shape (n_heads, n_batch, 2*n_samples)
        # the reason there are 2 per sample is that we consider it
        # once for the src node and once for the tgt node
        logits = torch.cat((src_logits, tgt_logits), dim=2)
        # dist is shape (n_heads, n_batch, 2*n_samples) (as the feature
        # dimension has been reduced by the norm operator)
        dist = torch.cat((src_diff, tgt_diff), dim=2).norm(dim=3)
        preds = self.aggregate_references(logits, dist, n_batch)
        return torch.sigmoid(preds).flatten()

    def aggregate_references(self, logits, dist, n_batch):
        '''
        Aggregate the reference points
        '''

        # Handling Sentinels
        # Reminder: sentinels are used to avoid giving too much weight to distant references
        logits=torch.cat((
            logits, torch.zeros(self.n_heads, n_batch, self.n_sentinels)
        ), dim = 2)
        dist=torch.cat((
            dist, torch.ones(self.n_heads, n_batch, self.n_sentinels)
        ), dim = 2)

        # Softmin
        softmin_=(
            logits.unsqueeze(2) @ torch.softmax(1-dist, dim=2).unsqueeze(3)
        ).squeeze(2).squeeze(2)

        # return the average over the different heads for prediction
        return softmin_.mean(0)

    def build_logits(self, embeds, batch_edges, node_type='source'):

        src_nodes, tgt_nodes = batch_edges[0, :].T, batch_edges[1, :].T

        if node_type == 'source':
            nodes_ = src_nodes
        elif node_type == 'target':
            nodes_ = tgt_nodes

        n_batch = batch_edges.shape[1]
        # Sample reference points
        # shape is (self.n_heads, n_batch, self.n_samples)
        samples = torch.randint(
        0, self.n_nodes, (self.n_heads, n_batch, self.n_samples))

        # TODO include what happens when no training
        if self.n_nearest and not self.training:
            # Grab closest pos vectors during test time, rather than using random ones
            samples = (
                embeds[:, nodes_].unsqueeze(2) - embeds.unsqueeze(1)
            ).norm(dim=3).topk(1+self.n_nearest, largest=False).indices[:, :, 1:]
            

        # Compute (u - u_0)
        # Notes:
        # embeds is shape (n_heads, n_nodes, node_feats)
        # so embeds[:, src_samples] retrieves the source nodes and their features for all the heads
        # embeds[:, src_samples] should be shape (n_heads, batch, node_features)
        #   with unsqueeze, add one dimension. so srcdiff is actually a 4-tensor
        heads_v=torch.arange(self.n_heads).unsqueeze(1).unsqueeze(2)
        # diff is shape (n_heads, n_batch, n_samples, n_features)
        diff=embeds[:, nodes_].unsqueeze(2) - embeds[heads_v, samples]

        # logits should be shape (n_heads, n_batch, n_samples)
        logits=(
        (diff.unsqueeze(3) @ (self.field[:, nodes_].unsqueeze(2).unsqueeze(4))
        ).squeeze(3).squeeze(3)
        # + self.uncertainty * self.edge[mid0, dst.unsqueeze(0).unsqueeze(2)]#TODO: handle this uncertainty term
        )

        return logits, diff

