import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
from typing import Tuple, Optional

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

        x = x_s * x_t
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x).flatten()

# mainly from https://github.com/cf020031308/mad-learning
class MADpredictor(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_nodes,
        n_heads=4,
        n_samples=256,
        n_sentinels=8,
        n_nearest=8,
        adj_t = None
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
        self.src = adj_t.to_torch_sparse_coo_tensor().coalesce().indices()[0, :]
        self.dst = adj_t.to_torch_sparse_coo_tensor().coalesce().indices()[1, :]
        self.adj = adj_t.to_dense()
        self.uncertainty = nn.Parameter(torch.ones(1,1))
        self.field = nn.Parameter(
            torch.rand((n_heads, n_nodes, embedding_dim)))

    def forward(self, embeds, batch_edges):
        '''
        embeds: embeddings for all the nodes in the graph (n_heads, n_nodes, n_features)
        edges: batch of edges to predict
        '''
        n_batch = batch_edges.shape[1]
        src, dst = batch_edges[0, :], batch_edges[1, :]

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
            logits, torch.zeros((self.n_heads, n_batch, self.n_sentinels), device=dist.device)
        ), dim = 2)
        dist=torch.cat((
            dist, torch.ones((self.n_heads, n_batch, self.n_sentinels), device=dist.device)
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

        # label
        if node_type == 'source':
            label = self.uncertainty * self.adj[samples, nodes_.unsqueeze(0).unsqueeze(2)]
        elif node_type == 'target':
            label = self.uncertainty * self.adj[nodes_.unsqueeze(0).unsqueeze(2), samples]


        # logits should be shape (n_heads, n_batch, n_samples)
        logits=(
        (diff.unsqueeze(3) @ (self.field[:, nodes_].unsqueeze(2).unsqueeze(4))
        ).squeeze(3).squeeze(3)
        + label
        )

        return logits, diff

