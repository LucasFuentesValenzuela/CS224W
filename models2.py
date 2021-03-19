import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
from typing import Tuple, Optional
# Some built in PyG layers
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from predictors import LinkPredictor


def get_model(model_name: str) -> type:
    models = [GCN_Linear, MAD_Model, MAD_GCN, MAD_Field_NN,
              MAD_GCN_Field_NN, MAD_SAGE, MAD_SAGE2, SAGE_Linear]
    for m in models:
        if m.__name__.lower() == model_name.lower():
            return m
    assert False, f'Could not find model {model_name}!'


class GCN_Linear(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        adj_t: torch_sparse.SparseTensor,
        output_dim=256,
        dropout=0.5,
    ):
        super(GCN_Linear, self).__init__()

        self.network = GCN(num_nodes, output_dim=output_dim, dropout=dropout)
        self.predictor = LinkPredictor(
            in_channels=output_dim,
            hidden_channels=output_dim,
            out_channels=1,
            num_layers=2,
            dropout=dropout,
        )

    def forward(
        self,
        adj_t: torch_sparse.SparseTensor,
        edges: torch.Tensor
    ) -> torch.Tensor:
        x = self.network(adj_t, edges)
        return self.predictor(x, edges)


class SAGE_Linear(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        adj_t: torch_sparse.SparseTensor,
        output_dim=256,
        dropout=0.5,
    ):
        super(SAGE_Linear, self).__init__()

        self.network = SAGE(
            num_nodes=num_nodes,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.predictor = LinkPredictor(
            in_channels=output_dim,
            hidden_channels=output_dim,
            out_channels=1,
            num_layers=2,
            dropout=dropout,
        )

    def forward(
        self,
        adj_t: torch_sparse.SparseTensor,
        edges: torch.Tensor
    ) -> torch.Tensor:
        x = self.network(adj_t, edges)
        return self.predictor(x, edges)


class GCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
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
            num_nodes, embedding_dim)

        nn.init.xavier_uniform_(self.embedding.weight)

        conv_layers = [GCNConv(embedding_dim, hidden_dim, cached=cache)] + \
            [GCNConv(hidden_dim, hidden_dim, cached=cache) for _ in range(num_layers-2)] + \
            [GCNConv(hidden_dim, output_dim, cached=cache)]
        self.convs = nn.ModuleList(conv_layers)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

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
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        return x


class SAGE(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        super(SAGE, self).__init__()

        self.embedding = nn.Embedding(
            num_nodes, embedding_dim)

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(embedding_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, output_dim))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor):
        x = self.embedding.weight

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class MAD_GCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        adj_t: torch_sparse.SparseTensor,
        num_heads: int = 12,
        mad_size: int = 12,
        embed_size: int = 200,
        hidden_size: int = 256,
        dropout: float = 0.5,
    ):
        super(MAD_GCN, self).__init__()

        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.mad_size = mad_size

        self.network = GCN(
            num_nodes=num_nodes,
            embedding_dim=embed_size,
            hidden_dim=hidden_size,
            output_dim=num_heads * mad_size * 2,
            num_layers=2,
            dropout=dropout,
        )
        self.predictor = MADEdgePredictor(
            num_nodes=num_nodes,
            adj_t=adj_t,
            num_heads=num_heads,
            embedding_dim=mad_size,
            num_sentinals=8,
            num_samples=8,
        )
        self.gcn_cache = None

    def forward(self, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:

        x = self.network(adj_t, edges)
        x = torch.reshape(
            x, (self.num_nodes, self.num_heads, 2 * self.mad_size))
        x = torch.clone(x, memory_format=torch.contiguous_format)
        pos = x[:, :, :self.mad_size]
        grad = x[:, :, self.mad_size:]
        return self.predictor(pos, grad, edges)


class MAD_SAGE(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        adj_t: torch_sparse.SparseTensor,
        num_heads: int = 12,
        mad_size: int = 12,
        hidden_size: int = 256,
        embed_size: int = 200,
        dropout: float = 0.5,
    ):
        super(MAD_SAGE, self).__init__()

        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.mad_size = mad_size

        self.network = SAGE(
            num_nodes=num_nodes,
            embedding_dim=embed_size,
            hidden_dim=hidden_size,
            output_dim=num_heads * mad_size * 2,
            num_layers=2,
            dropout=dropout,
        )
        self.predictor = MADEdgePredictor(
            num_nodes=num_nodes,
            adj_t=adj_t,
            num_heads=num_heads,
            embedding_dim=mad_size,
            num_sentinals=8,
            num_samples=8,
            k_nearest=32,
        )
        self.gcn_cache = None

    def forward(self, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:

        x = self.network(adj_t, edges)
        x = torch.reshape(
            x, (self.num_nodes, self.num_heads, 2 * self.mad_size))
        x = torch.clone(x, memory_format=torch.contiguous_format)
        pos = x[:, :, :self.mad_size]
        grad = x[:, :, self.mad_size:]
        return self.predictor(pos, grad, edges)



class MAD_SAGE2(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        adj_t: torch_sparse.SparseTensor,
        num_heads: int = 12,
        mad_size: int = 12,
        hidden_size: int = 256,
        embed_size: int = 200,
        dropout: float = 0.5,
    ):
        super(MAD_SAGE2, self).__init__()

        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.mad_size = mad_size

        self.embedding = nn.Parameter(torch.empty((self.num_nodes, self.num_heads, self.mad_size)))
        nn.init.xavier_normal_(self.embedding)

        self.network = SAGE(
            num_nodes=num_nodes,
            embedding_dim=embed_size,
            hidden_dim=hidden_size,
            output_dim=num_heads * mad_size,
            num_layers=2,
            dropout=dropout,
        )
        self.predictor = MADEdgePredictor(
            num_nodes=num_nodes,
            adj_t=adj_t,
            num_heads=num_heads,
            embedding_dim=mad_size,
            num_sentinals=8,
            num_samples=8,
            distance='euclidian',
            k_nearest=32,
            sample_weights='softmin',
        )
        self.gcn_cache = None

    def forward(self, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:

        x = self.network(adj_t, edges)
        x = torch.reshape(
            x, (self.num_nodes, self.num_heads, self.mad_size))
        x = torch.clone(x, memory_format=torch.contiguous_format)
        pos = self.embedding
        grad = x
        return self.predictor(pos, grad, edges)


class MAD_GCN_Field_NN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        adj_t: torch_sparse.SparseTensor,
        num_heads: int = 12,
        mad_size: int = 12,
        hidden_size: int = 256,
        embed_size: int = 200,
        dropout: float = 0.5,
    ):
        super(MAD_GCN_Field_NN, self).__init__()

        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.mad_size = mad_size

        self.network = GCN(
            num_nodes=num_nodes,
            embedding_dim=embed_size,
            hidden_dim=hidden_size,
            output_dim=num_heads * mad_size,
            num_layers=2,
            dropout=dropout,
        )
        self.predictor = MADEdgePredictor(
            num_nodes=num_nodes,
            adj_t=adj_t,
            num_heads=num_heads,
            embedding_dim=mad_size,
            num_sentinals=8,
            num_samples=8,
        )

        self.field_nn = FieldPredictor(
            mad_size, num_heads, num_nodes, dropout=0.5, num_layers=3)
        self.gcn_cache = None

    def forward(self, adj_t: torch_sparse.SparseTensor, edges: torch.Tensor) -> torch.Tensor:

        x = self.network(adj_t, edges)
        x = torch.reshape(x, (self.num_nodes, self.num_heads, self.mad_size))
        x = torch.clone(x, memory_format=torch.contiguous_format)
        pos = x
        grad = self.field_nn(x)
        return self.predictor(pos, grad, edges)


class MAD_Field_NN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        adj_t: torch_sparse.SparseTensor,
        num_heads: int = 12,
        embedding_dim: int = 12,
    ):
        super(MAD_Field_NN, self).__init__()
        self.pos_embs = nn.Parameter(
            torch.empty((num_nodes, num_heads, embedding_dim)))

        self.field_nn = FieldPredictor(
            embedding_dim, num_heads, num_nodes, dropout=0.5, num_layers=3)

        self.predictor = MADEdgePredictor(
            num_nodes=num_nodes,
            adj_t=adj_t,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            num_sentinals=8,
            num_samples=8,
        )

        nn.init.xavier_uniform_(self.pos_embs)

    def forward(
        self,
        adj_t: torch_sparse.SparseTensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Inputs:
            adj_t: sparse tensor containing graph adjacency matrix.
            edges: Tensor of shape (2, batch_size)

        Returns:
            predictions: Tensor of shape (batch_size,)
        '''
        pos = self.pos_embs
        grads = self.field_nn(pos)
        return self.predictor(pos, grads, edges)


class MAD_Model(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        adj_t: torch_sparse.SparseTensor,
        num_heads: int = 12,
        embedding_dim: int = 12,
    ):
        super(MAD_Model, self).__init__()
        self.pos_embs = nn.Parameter(
            torch.empty((num_nodes, num_heads, embedding_dim)))

        self.grad_embs = nn.Parameter(
            torch.empty((num_nodes, num_heads, embedding_dim)))

        self.predictor = MADEdgePredictor(
            num_nodes=num_nodes,
            adj_t=adj_t,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            num_sentinals=0,
            num_samples=8,
            k_nearest=8,
            sentinal_dist=1,
            distance="euclidian",
            sample_weights='attention',
            num_weight_layers=2,
            hidden_weight_dim=48,
            thresh_weight=1
        )
        nn.init.xavier_uniform_(self.pos_embs)
        nn.init.xavier_uniform_(self.grad_embs)

    def forward(
        self,
        adj_t: torch_sparse.SparseTensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Inputs:
            adj_t: sparse tensor containing graph adjacency matrix.
            edges: Tensor of shape (2, batch_size)

        Returns:
            predictions: Tensor of shape (batch_size,)
        '''
        pos = self.pos_embs
        grads = self.grad_embs
        return self.predictor(pos, grads, edges)


class MADAttention(torch.nn.Module):
    def __init__(
        self, embedding_dim, hidden_channels,
        out_channels, num_layers, dropout=.5
        ):
        super().__init__()

        self.lins = torch.nn.ModuleList()

        if num_layers == 1:
            self.lins.append(nn.Linear(2*embedding_dim, out_channels))
        else:
            self.lins.append(torch.nn.Linear(2*embedding_dim, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x1, x0, return_softmin=True):
        '''
        shapes:
        # x [pos_src, pos_dst]: (batch_size, num_heads, embedding_dim)
        # x0 [pos_src0, pos_dst0]: (batch_size, num_heads, num_samples, embedding_dim)
        '''
        x1_exp = x1.unsqueeze(2).repeat((1, 1, x0.shape[2], 1))
        x = torch.cat([x1_exp, x0], dim=3)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        x = x.squeeze(3)

        if return_softmin:
            out = F.softmin(x, dim=2)
        else:
            out = torch.sigmoid(x) #do not normalize the softmins
        return out


class MADEdgePredictor(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        adj_t: torch_sparse.SparseTensor,
        num_heads: int,
        embedding_dim: int,
        num_sentinals: int,
        num_samples: int,
        k_nearest: int = 8,
        sentinal_dist: float = 1.,
        distance: str = 'euclidian',
        sample_weights: str = 'softmin',
        num_weight_layers: int = 3,
        hidden_weight_dim: int = 32,
        thresh_weight: int = 1
    ):
        '''
        distance: how to compute the weighting of different samples
        sample_weights: how the weight for each sample is computed
        num_weight_layers: how many layers in the weight NN
        hidden_weight_dim: size of weight computing layer in the weight NN
        '''

        super(MADEdgePredictor, self).__init__()
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.num_sentinals = num_sentinals
        self.embedding_dim = embedding_dim
        self.num_samples = num_samples
        self.k_nearest = k_nearest
        self.distance = distance
        self.sentinal_dist = sentinal_dist
        self.sample_weights = sample_weights
        self.thresh_weight = thresh_weight

        # nn to apply to adj_t labels
        self.label_nn = nn.Linear(1, 1, bias=False)
        self.adj = adj_t.to_dense() * 2 - 1  # Scale from -1 to 1

        # nn.init.xavier_normal(self.label_nn.weight)
        self.label_nn.weight = nn.Parameter(
            torch.ones_like(self.label_nn.weight))

        assert self.distance in ['euclidian',
                                 'inner', 'dot'], 'Distance metric invalid'
        assert self.sample_weights in [
            'softmin', 'attention'], 'Sample weight method invalid'

        # weighted inner product xTWx
        if self.distance == 'inner':
            self.W = nn.Linear(self.embedding_dim, self.embedding_dim)

        if self.sample_weights == 'attention':
            self.atn = MADAttention(
                embedding_dim, hidden_weight_dim, 1, num_weight_layers)

    def forward(self, pos: torch.Tensor, grads: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        '''
        pos Tensor of shape (num_nodes, num_heads, embedding_dim)
        grads Tensor of shape (num_nodes, num_heads, embedding_dim)
        edges Tensor of shape (2, batch_size)
        '''

        # Model link prediction problem as:
        # Pred = r(pos[src], pos[dst])
        #     ~=~ [
        #              r(src0, dst) + (pos[src] - pos[src0]) grads(dst),
        #              r(src, dst0) + (pos[dst] - pos[dst0]) grads(src),
        #         ]
        # Average preds over several heads, weighed by distance
        # of [src, dst] to [src0, dst0]

        device = edges.device

        batch_size = edges.shape[1]
        src = edges[0]  # (batch_size,)
        dst = edges[1]  # (batch_size,)

        heads_idx = torch.arange(
            0, self.num_heads, device=device)  # (num_heads,)

        # (batch_size, num_heads, embedding_dim)
        pos_src = pos[src.view(batch_size, 1),
                      heads_idx.view(1, self.num_heads)]
        pos_dst = pos[dst.view(batch_size, 1),
                      heads_idx.view(1, self.num_heads)]

        # Indices of sampled nodes for gradient estimation.
        # (batch_size, num_heads, num_samples)
        if self.training:
            # At training time just sample randomly.
            src0 = torch.randint(0, self.num_nodes, size=(
                batch_size, self.num_heads, self.num_samples), device=device)
            dst0 = torch.randint(0, self.num_nodes, size=(
                batch_size, self.num_heads, self.num_samples), device=device)
        else:
            # Grab TopK closest src0 and dst0 nodes to src and dst
            if self.sample_weights=='softmin':
                if self.distance == 'euclidian' or self.distance == 'inner':
                    # (num_nodes, batch_size, num_heads)
                    src_norm = torch.norm(
                        pos.view(self.num_nodes, 1, self.num_heads,
                                self.embedding_dim)
                        - pos_src.view(1, batch_size, self.num_heads,
                                    self.embedding_dim),
                        dim=3,
                    )
                    dst_norm = torch.norm(
                        pos.view(self.num_nodes, 1, self.num_heads,
                                self.embedding_dim)
                        - pos_dst.view(1, batch_size, self.num_heads,
                                    self.embedding_dim),
                        dim=3,
                    )
                elif self.distance == 'dot':
                    # num_nodes, num_heads, embed_dim
                    pos_norm = pos / torch.norm(pos, dim=2, keepdim=True)

                    distance_shape = (self.num_nodes, batch_size, self.num_heads)
                    # batch_size, num_heads, embedding_dim
                    pos_src_norm = pos_src / \
                        torch.norm(pos_src, dim=2, keepdim=True)
                    src_norm = -torch.sum(
                        pos_src_norm.view(1, batch_size, self.num_heads, self.embedding_dim) *
                        pos_norm.view(self.num_nodes, 1,
                                    self.num_heads, self.embedding_dim),
                        dim=3
                    ).view(distance_shape)

                    # batch_size, num_heads, embedding_dim
                    pos_dst_norm = pos_dst / \
                        torch.norm(pos_dst, dim=2, keepdim=True)
                    dst_norm = -torch.sum(
                        pos_dst_norm.view(1, batch_size, self.num_heads, self.embedding_dim) *
                        pos_norm.view(self.num_nodes, 1,
                                    self.num_heads, self.embedding_dim),
                        dim=3
                    ).view(distance_shape)

            elif self.sample_weights=='attention':
                #shapes in attention mechanism:
                # x [pos_src, pos_dst]: (batch_size, num_heads, embedding_dim)
                # x0 [pos_src0, pos_dst0]: (batch_size, num_heads, num_samples, embedding_dim)

                pos_ = pos.permute(1, 0, 2).unsqueeze(0).repeat((pos_src.shape[0], 1, 1, 1))
                # # (batch_size, num_heads, num_nodes)
                src_norm = -self.atn(pos_src, pos_, return_softmin=False).permute(2, 0, 1)
                dst_norm = -self.atn(pos_dst, pos_, return_softmin=False).permute(2, 0, 1)

                # choose random nodes for performance
                # src0 = torch.randint(0, self.num_nodes, size=(
                #     batch_size, self.num_heads, self.k_nearest), device=device)
                # dst0 = torch.randint(0, self.num_nodes, size=(
                #     batch_size, self.num_heads, self.k_nearest), device=device)

            # if self.sample_weights != 'attention':
            # (k_nearest, batch_size, num_heads)
            src0 = torch.topk(src_norm, k=self.k_nearest+1,
                                largest=False, sorted=False, dim=0).indices[1:]
            dst0 = torch.topk(dst_norm, k=self.k_nearest+1,
                                largest=False, sorted=False, dim=0).indices[1:]
            # (batch_size, num_heads, k_nearest)
            src0 = src0.permute(1, 2, 0)
            dst0 = dst0.permute(1, 2, 0)


        # (batch_size, num_heads, num_samples, embedding_dim)
        pos_src0 = pos[src0, heads_idx.view(1, self.num_heads, 1)]
        pos_dst0 = pos[dst0, heads_idx.view(1, self.num_heads, 1)]
        num_samples = pos_src0.shape[2]

        # pos[src] - pos[src0]
        # (batch_size, num_heads, num_samples, embedding_dim)
        src_dist = pos_src.view(
            batch_size, self.num_heads, 1, self.embedding_dim) - pos_src0
        dst_dist = pos_dst.view(
            batch_size, self.num_heads, 1, self.embedding_dim) - pos_dst0

        # grads(src), grads(dst)
        # (batch_size, num_heads, embedding_dim)
        grads_src = grads[dst.view(batch_size, 1),
                          heads_idx.view(1, self.num_heads)]
        grads_dst = grads[src.view(batch_size, 1),
                          heads_idx.view(1, self.num_heads)]

        # Take dot product to eliminate embedding_dim dimension
        # (batch_size, num_heads, num_samples)
        src_contrib = torch.matmul(
            src_dist.view(batch_size, self.num_heads,
                          num_samples, 1, self.embedding_dim),
            grads_src.view(batch_size, self.num_heads,
                           1, self.embedding_dim, 1)
        ).view(batch_size, self.num_heads, num_samples)
        dst_contrib = torch.matmul(
            dst_dist.view(batch_size, self.num_heads,
                          num_samples, 1, self.embedding_dim),
            grads_dst.view(batch_size, self.num_heads,
                           1, self.embedding_dim, 1)
        ).view(batch_size, self.num_heads, num_samples)

        # (batch_size, num_heads, num_samples)
        src_logits = self.label_nn.weight * \
            self.adj[src0, dst.view(batch_size, 1, 1)] + src_contrib
        dst_logits = self.label_nn.weight * \
            self.adj[src.view(batch_size, 1, 1), dst0] + dst_contrib

        # (batch_size, num_heads, num_samples * 2)
        logits = torch.cat([src_logits, dst_logits], dim=2)

        if self.sample_weights == 'softmin':
            # Weigh according to softmin distance, sentinals thing
            # Sum across num_samples should be 1.
            if self.distance == 'euclidian':
                # (batch_size, num_heads, 2 * num_samples)
                distance = torch.norm(
                    torch.cat([src_dist, dst_dist], dim=2), dim=3)
            elif self.distance == 'inner':
                pos_src_ = pos_src.view(
                    batch_size, self.num_heads, 1, self.embedding_dim)
                pos_src0_proj = self.W(pos_src0)
                inner_src = torch.sum(
                    pos_src_*pos_src0_proj,
                    dim=3
                )/(torch.norm(pos_src_, dim=3)*(torch.norm(pos_src0_proj, dim=3)))

                pos_dst_ = pos_dst.view(
                    batch_size, self.num_heads, 1, self.embedding_dim)
                pos_dst0_proj = self.W(pos_dst0)
                inner_dst = torch.sum(
                    pos_dst_*pos_dst0_proj,
                    dim=3
                )/(torch.norm(pos_dst_, dim=3)*(torch.norm(pos_dst0_proj, dim=3)))
                distance = torch.cat([inner_src, inner_dst], dim=2)
            elif self.distance == 'dot':
                # Negative dot product, so that smaller is closer.
                distance_shape = (batch_size, self.num_heads, num_samples)
                # (batch_size, num_heads, embed_size)
                pos_src_norm = pos_src / \
                    torch.norm(pos_src, dim=2, keepdim=True)
                # (batch_size, num_heads, num_samples, embed_size)
                pos_src0_norm = pos_src0 / \
                    torch.norm(pos_src0, dim=2, keepdim=True)
                inner_src = -torch.sum(
                    pos_src_norm.view(batch_size, self.num_heads, 1, self.embedding_dim) *
                    pos_src0_norm.view(
                        batch_size, self.num_heads, num_samples, self.embedding_dim),
                    dim=3
                ).view(distance_shape)

                pos_dst_norm = pos_dst / \
                    torch.norm(pos_dst, dim=2, keepdim=True)
                pos_dst0_norm = pos_dst0 / \
                    torch.norm(pos_dst0, dim=2, keepdim=True)
                inner_dst = -torch.sum(
                    pos_dst_norm.view(batch_size, self.num_heads, 1, self.embedding_dim) *
                    pos_dst0_norm.view(
                        batch_size, self.num_heads, num_samples, self.embedding_dim),
                    dim=3
                ).view(distance_shape)

                # (batch_size, num_heads, 2 * num_samples)
                distance = torch.cat([inner_src, inner_dst], dim=2)

            # (batch_size, num_heads, 2 * num_samples + num_sentinals)
            if self.num_sentinals == 0:
                norm_sentinals = distance
            else:
                norm_sentinals = torch.cat([distance, torch.full(
                    (batch_size, self.num_heads, self.num_sentinals), self.sentinal_dist, device=device)], dim=2)

            # Get softmax weights, strip out sentinals
            # (batch_size, num_heads, num_samples * 2)
            # logit_weight = F.softmin(norm_sentinals, dim=2)[:, :, :self.num_samples*2]
            logit_weight = F.softmin(norm_sentinals, dim=2)[
                :, :, :num_samples*2]

        # Compute logit weight based on bespoke attention mechanism
        elif self.sample_weights == 'attention':
            # (batch_size, num_heads, num_samples)
            src_weight = self.atn(pos_src, pos_src0, return_softmin=False)
            dst_weight = self.atn(pos_dst, pos_dst0, return_softmin=False)

            total_weight = torch.sum(src_weight, dim=2, keepdim=True) + torch.sum(dst_weight, dim=2, keepdim=True)
            total_weight = total_weight + self.num_sentinals * torch.sigmoid(torch.tensor(self.sentinal_dist, device=device, dtype=torch.float))

            # (batch_size, num_heads, 2*num_samples)
            logit_weight = torch.cat([src_weight, dst_weight], dim=2) / total_weight

        # Weigh logits according to weights and take mean
        # (batch_size, num_heads)
        weighed_logits = torch.sum(logit_weight * logits, dim=2)

        output_logits = torch.mean(weighed_logits, dim=1)
        return torch.sigmoid(output_logits)


class FieldPredictor(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_heads,
        n_nodes,
        dropout=.5,
        num_layers=2
    ):
        # shape of field
        # (n_heads, n_nodes, embedding_dim)

        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_nodes = n_nodes
        self.dropout = dropout

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(
            n_heads*embedding_dim, n_heads*embedding_dim))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(
                n_heads*embedding_dim, n_heads*embedding_dim))
        self.lins.append(torch.nn.Linear(
            n_heads*embedding_dim, n_heads*embedding_dim))

        bns_layers = [nn.BatchNorm1d(num_features=embedding_dim)
                      for _ in range(num_layers)]
        self.bns = nn.ModuleList(bns_layers)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        '''
        x are shape (num_nodes, num_heads, embedding_dim)
        '''
        # reshape to (n_nodes, n_heads*embedding_dim)
        x = x.view(self.n_nodes, self.n_heads * self.embedding_dim)

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        # reshape to (n_heads, n_nodes, embedding_dim)
        x = x.view(self.n_nodes, self.n_heads, self.embedding_dim)
        return x
