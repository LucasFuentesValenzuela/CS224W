import os
from typing import Tuple, List, Dict
import random

import pickle
import numpy as np # type: ignore
import torch
from torch import nn
from tqdm import tqdm # type: ignore
from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric as pyg
import torch_geometric.transforms as T
import networkx as nx



def save_model(model: nn.Module, path: str) -> nn.Module:
    model = model.cpu()
    torch.save(model.state_dict(), path)
    model = model.to(get_device())
    return model


def load_model(model: nn.Module, path: str) -> nn.Module:
    model = model.cpu()
    model.load_state_dict(torch.load(path))
    model = model.to(get_device())
    return model


def load_training_data() -> Tuple[
    pyg.data.Data,
    pyg.data.Data,
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor]
]:
    '''
    Returns Tuple
        train_graph Graph containing a subset of the training edges
        valid_graph Graph containing all training edges
        train_edges Dict of positive edges across entire train split
        eval_edges Dict of positive edges from the training edges set that aren't in eval_graph
        valid_edges Dict of positive and negative edges not in train_graph.
    '''
    dataset = PygLinkPropPredDataset(name='ogbl-ddi')
    transform = T.ToSparseTensor(False)
    edge_split = dataset.get_edge_split()
    train_edges = edge_split['train']
    valid_edges = edge_split['valid']
    train_graph = dataset[0]
    valid_graph = train_graph.clone()

    # Partition training edges
    torch.manual_seed(12345)
    perm = torch.randperm(train_edges['edge'].shape[0])
    eval_idxs, train_idxs = perm[:valid_edges['edge'].shape[0]], perm[valid_edges['edge'].shape[0]:]
    eval_edges = {'edge': train_edges['edge'][eval_idxs]}
    train_edges = {'edge': train_edges['edge'][train_idxs]}

    # Update graph object to have subset of edges and adj_t matrix
    train_edge_index = torch.cat([train_edges['edge'].T, train_edges['edge'][:, [1, 0]].T], dim=1)
    train_graph.edge_index = train_edge_index
    train_graph = transform(train_graph)
    valid_graph = transform(valid_graph)

    return train_graph, valid_graph, edge_split['train'], eval_edges, valid_edges


def load_test_data() -> Tuple[
    pyg.torch_geometric.data.Data,
    pyg.torch_geometric.data.Data,
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor]
]:
    '''
    Returns Tuple
        valid_graph Graph containing all training edges
        test_graph Graph containing all training edges, plus validation edges
        valid_edges Dict of positive and negative edges from validation edge split (not in train_graph)
        test_edges Dict of positive and negative edges from test edge split (not in valid_graph)
    '''
    dataset = PygLinkPropPredDataset(name='ogbl-ddi')
    transform = T.ToSparseTensor(False)
    edge_split = dataset.get_edge_split()
    valid_edges = edge_split['valid']
    test_edges = edge_split['test']
    valid_graph = dataset[0]
    test_graph = valid_graph.clone()

    # Add validation edges to valid_graph for test inference
    valid_edge_index = torch.cat([
            test_graph.edge_index,
            valid_edges['edge'].T,
            valid_edges['edge'][:, [1, 0]].T
        ], dim=1)
    test_graph.edge_index = valid_edge_index

    valid_graph = transform(valid_graph)
    test_graph = transform(test_graph)
    return valid_graph, test_graph, valid_edges, test_edges

def initialize_embeddings(graph: pyg.data.Data, filename: str, refresh: bool=False):
    '''
    Input
        graph: PyG graph
        filename: File to read/write embeddings to
        refresh: if True, recompute embeddings.
    '''
    if not refresh:
        features = torch.load(filename)
        graph.x = features
        return graph

    g_nx = pyg.torch_geometric.utils.convert.to_networkx(graph, to_undirected=True)
    print('  Computing pagerank')
    pageranks = nx.pagerank(g_nx)
    pagerank_vector = torch.zeros((graph.num_nodes, 1), dtype=torch.float32)
    for node in range(graph.num_nodes):
        pagerank_vector[node] = pageranks[node]

    print('  Computing degrees')
    degree_vector = torch.zeros((graph.num_nodes, 1), dtype=torch.float32)
    for node in range(graph.num_nodes):
        degree_vector[node] = g_nx.degree[node]

    print('  Computing clustering coefficients')
    clustering_coeffs = nx.clustering(g_nx)
    cc_vector = torch.zeros((graph.num_nodes, 1), dtype=torch.float32)
    for node in range(graph.num_nodes):
        cc_vector[node] = clustering_coeffs[node]

    features = torch.cat([degree_vector, cc_vector, pagerank_vector], dim=1)
    graph.x = features

    torch.save(features, filename)
    return graph


def check_membership(edge, edge_list):
    '''
    Input
        edge torch.tensor shape (2,)
        edge_list torch.tensor shape (E, 2)
    Returns True if edge is in edge_list.
    '''
    return torch.sum(torch.sum(edge_list == edge, dim=1) == 2) == 1


def get_device() -> torch.device:
    '''
    Guesses the best device for the current machine.
    '''
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def l1_norm_loss(input: torch.Tensor, pos_target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(input - pos_target))

def l2_norm_loss(input: torch.Tensor, pos_target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.pow(input - pos_target, 2))

def verify_versions() -> None:
    # Version 1.5.0 has a bug where certain type annotations don't pass typecheck
    assert torch.__version__.startswith('1.7.1'), 'Incorrect torch version installed!'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
