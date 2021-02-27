import argparse
import os

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch_geometric as pyg
from tqdm import tqdm # type: ignore
from ogb.linkproppred import Evaluator

from args import add_model_args, add_test_args, add_common_args
import models
import model_utils


@torch.no_grad()
def test_model(
    test_graph: pyg.torch_geometric.data.Data,
    dev_dl: data.DataLoader,
    model: nn.Module,
    evaluator: Evaluator,
    args: argparse.Namespace,
) -> nn.Module:

    device = model_utils.get_device()

    print('\nRunning test metrics...')
    # Forward inference on model
    print('  Running forward inference...')
    pos_pred = []
    neg_pred = []
    with tqdm(total=args.batch_size * len(dev_dl)) as progress_bar:

        adj_t = test_graph.adj_t.to(device)
        edge_index = test_graph.edge_index.to(device)
        x = test_graph.x.to(device)

        for i, (edges_batch, y_batch) in enumerate(dev_dl):
            edges_batch = edges_batch.T.to(device)
            y_batch = y_batch.to(device)

            # Forward pass on model
            y_pred = model(x, adj_t, edges_batch)

            # TODO: Process y_pred in the optimal way (round it off, etc)

            # TODO: Log statistics
            pos_pred += [y_pred[y_batch == 1].cpu()]
            neg_pred += [y_pred[y_batch == 0].cpu()]

            progress_bar.update(edges_batch.shape[1])

            del edges_batch
            del y_pred
            del y_batch

        del adj_t
        del edge_index

    pos_pred = torch.cat(pos_pred, dim=0)
    neg_pred = torch.cat(neg_pred, dim=0)


    print(f'\n  Calculating overall metrics...')
    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        # train_hits = evaluator.eval({
        #     'y_pred_pos': pos_train_pred,
        #     'y_pred_neg': neg_valid_pred,
        # })[f'hits@{K}']
        # valid_hits = evaluator.eval({
        #     'y_pred_pos': pos_valid_pred,
        #     'y_pred_neg': neg_valid_pred,
        # })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (test_hits,)#(train_hits, valid_hits, test_hits)


    print()
    print('*' * 30)
    for k, v in results.items():
        print(f'{k}: {v}')
    print('*' * 30)

    return model, results


def main():
    parser = argparse.ArgumentParser()
    add_test_args(parser)
    add_common_args(parser)
    add_model_args(parser)
    args = parser.parse_args()

    device = model_utils.get_device()

    # Load dataset from disk
    # TODO: Load test data
    valid_graph, test_graph, valid_edges, test_edges = model_utils.load_test_data()
    dev_dl = data.DataLoader(
        data.TensorDataset(
            torch.cat([test_edges['edge'], test_edges['edge_neg']], dim=0),
            torch.cat([torch.ones(test_edges['edge'].shape[0]), torch.zeros(test_edges['edge_neg'].shape[0])], dim=0)
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Initialize node embeddings
    test_graph = model_utils.initialize_embeddings(test_graph, 'test_embeddings.pt', args.refresh_embeddings)
    valid_graph = model_utils.initialize_embeddings(valid_graph, 'valid_embeddings.pt', args.refresh_embeddings)

    # Initialize a model
    model = models.get_model(args.model)(test_graph.x.shape, args)

    # load from checkpoint if path specified
    assert args.load_path is not None
    model = model_utils.load_model(model, args.load_path)
    model.eval()

    # Move model to GPU if necessary
    model.to(device)

    # Stats evaluator
    evaluator = Evaluator(name='ogbl-ddi')

    # test!
    test_model(
        test_graph,
        dev_dl,
        model,
        evaluator,
        args,
    )


if __name__ == '__main__':
    model_utils.verify_versions()
    main()


