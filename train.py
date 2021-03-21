import argparse
import os
from typing import Dict

import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # type: ignore
import torch_geometric as pyg
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import Evaluator

from args import *
import model_utils
import models2 as models


def train_model(
    train_graph: pyg.torch_geometric.data.Data,
    valid_graph: pyg.torch_geometric.data.Data,
    train_dl: data.DataLoader,
    dev_dl: data.DataLoader,
    evaluator: Evaluator,
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
) -> nn.Module:

    device = model_utils.get_device()
    loss_fn = nn.functional.binary_cross_entropy
    val_loss_fn = nn.functional.binary_cross_entropy
    best_val_loss = torch.tensor(float('inf'))
    best_val_hits = torch.tensor(0.0)
    saved_checkpoints = []
    writer = SummaryWriter(log_dir=f'{args.log_dir}/{args.experiment}')

    for e in range(1, args.train_epochs + 1):
        print(f'Training epoch {e}...')

        # Training portion
        torch.cuda.empty_cache()
        torch.set_grad_enabled(True)
        with tqdm(total=args.train_batch_size * len(train_dl)) as progress_bar:
            model.train()

            # Load graph into GPU
            adj_t = train_graph.adj_t.to(device)
            edge_index = train_graph.edge_index.to(device)
            x = train_graph.x.to(device)

            pos_pred = []
            neg_pred = []

            for i, (y_pos_edges,) in enumerate(train_dl):
                y_pos_edges = y_pos_edges.to(device).T
                y_neg_edges = negative_sampling(
                    edge_index,
                    num_nodes=train_graph.num_nodes,
                    num_neg_samples=y_pos_edges.shape[1]
                ).to(device)
                y_batch = torch.cat([torch.ones(y_pos_edges.shape[1]), torch.zeros(
                    y_neg_edges.shape[1])], dim=0).to(device)  # Ground truth edge labels (1 or 0)

                # Forward pass on model
                optimizer.zero_grad()
                y_pred = model(adj_t, torch.cat(
                    [y_pos_edges, y_neg_edges], dim=1))
                loss = loss_fn(y_pred, y_batch)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                if args.use_scheduler:
                    lr_scheduler.step(loss)

                batch_acc = torch.mean(
                    1 - torch.abs(y_batch.detach() - torch.round(y_pred.detach()))).item()

                pos_pred += [y_pred[y_batch == 1].detach()]
                neg_pred += [y_pred[y_batch == 0].detach()]

                progress_bar.update(y_pos_edges.shape[1])
                progress_bar.set_postfix(loss=loss.item(), acc=batch_acc)
                writer.add_scalar(
                    "train/Loss", loss, ((e - 1) * len(train_dl) + i) * args.train_batch_size)
                writer.add_scalar("train/Accuracy", batch_acc,
                                  ((e - 1) * len(train_dl) + i) * args.train_batch_size)

                del y_pos_edges
                del y_neg_edges
                del y_pred
                del loss

            del adj_t
            del edge_index
            del x

            # Training set evaluation Hits@K Metrics
            pos_pred = torch.cat(pos_pred, dim=0)
            neg_pred = torch.cat(neg_pred, dim=0)
            results = {}
            for K in [10, 20, 30]:
                evaluator.K = K
                hits = evaluator.eval({
                    'y_pred_pos': pos_pred,
                    'y_pred_neg': neg_pred,
                })[f'hits@{K}']
                results[f'Hits@{K}'] = hits
            print()
            print(f'Train Statistics')
            print('*' * 30)
            for k, v in results.items():
                print(f'{k}: {v}')
                writer.add_scalar(
                    f"train/{k}", v, (pos_pred.shape[0] + neg_pred.shape[0]) * e)
            print('*' * 30)

            del pos_pred
            del neg_pred

        # Validation portion
        torch.cuda.empty_cache()
        torch.set_grad_enabled(False)
        with tqdm(total=args.val_batch_size * len(dev_dl)) as progress_bar:
            model.eval()

            adj_t = valid_graph.adj_t.to(device)
            edge_index = valid_graph.edge_index.to(device)
            x = valid_graph.x.to(device)

            val_loss = 0.0
            accuracy = 0
            num_samples_processed = 0
            pos_pred = []
            neg_pred = []
            for i, (edges_batch, y_batch) in enumerate(dev_dl):
                edges_batch = edges_batch.T.to(device)
                y_batch = y_batch.to(device)

                # Forward pass on model in validation environment
                y_pred = model(adj_t, edges_batch)
                loss = val_loss_fn(y_pred, y_batch)

                num_samples_processed += edges_batch.shape[1]
                batch_acc = torch.mean(
                    1 - torch.abs(y_batch - torch.round(y_pred))).item()
                accuracy += batch_acc * edges_batch.shape[1]
                val_loss += loss.item() * edges_batch.shape[1]

                pos_pred += [y_pred[y_batch == 1].detach()]
                neg_pred += [y_pred[y_batch == 0].detach()]

                progress_bar.update(edges_batch.shape[1])
                progress_bar.set_postfix(
                    val_loss=val_loss / num_samples_processed,
                    acc=accuracy/num_samples_processed)
                writer.add_scalar(
                    "Val/Loss", loss, ((e - 1) * len(dev_dl) + i) * args.val_batch_size)
                writer.add_scalar(
                    "Val/Accuracy", batch_acc, ((e - 1) * len(dev_dl) + i) * args.val_batch_size)

                del edges_batch
                del y_batch
                del y_pred
                del loss

            del adj_t
            del edge_index
            del x

            # Validation evaluation Hits@K Metrics
            pos_pred = torch.cat(pos_pred, dim=0)
            neg_pred = torch.cat(neg_pred, dim=0)
            results = {}
            for K in [10, 20, 30]:
                evaluator.K = K
                hits = evaluator.eval({
                    'y_pred_pos': pos_pred,
                    'y_pred_neg': neg_pred,
                })[f'hits@{K}']
                results[f'Hits@{K}'] = hits
            print()
            print(f'Validation Statistics')
            print('*' * 30)
            for k, v in results.items():
                print(f'{k}: {v}')
                writer.add_scalar(
                    f"Val/{k}", v, (pos_pred.shape[0] + neg_pred.shape[0]) * e)
            print('*' * 30)

            del pos_pred
            del neg_pred

            # Save model if it's the best one yet.
            if results['Hits@20'] > best_val_hits:
                best_val_hits = results['Hits@20']
                filename = f'{args.save_path}/{args.experiment}/{model.__class__.__name__}_best_val.checkpoint'
                model_utils.save_model(model, filename)
                print(f'Model saved!')
                print(f'Best validation Hits@20 yet: {best_val_hits}')
            # Save model on checkpoints.
            if e % args.checkpoint_freq == 0:
                filename = f'{args.save_path}/{args.experiment}/{model.__class__.__name__}_epoch_{e}.checkpoint'
                model_utils.save_model(model, filename)
                print(f'Model checkpoint reached!')
                saved_checkpoints.append(filename)
                # Delete checkpoints if there are too many
                while len(saved_checkpoints) > args.num_checkpoints:
                    os.remove(saved_checkpoints.pop(0))

    return model


def main():
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    add_experiment(args)
    device = model_utils.get_device()

    # Load dataset from disk
    print('Loading train data...')
    train_graph, valid_graph, train_edges, eval_edges, valid_edges = model_utils.load_training_data()
    if args.train_partial_graph:
        train_edges['edge'] = eval_edges['edge']

    train_dl = data.DataLoader(
        data.TensorDataset(train_edges['edge']),
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    dev_dl = data.DataLoader(
        data.TensorDataset(
            torch.cat([valid_edges['edge'], valid_edges['edge_neg']], dim=0),
            torch.cat([torch.ones(valid_edges['edge'].shape[0]),
                       torch.zeros(valid_edges['edge_neg'].shape[0])], dim=0),
        ),
        batch_size=args.val_batch_size,
        shuffle=True,
    )

    # Initialize node embeddings
    print('Computing initial embeddings')
    train_graph = model_utils.initialize_embeddings(
        train_graph, 'train_embeddings.pt', args.refresh_embeddings)
    valid_graph = model_utils.initialize_embeddings(
        valid_graph, 'valid_embeddings.pt', args.refresh_embeddings)
    if not args.train_partial_graph:
        train_graph = valid_graph

    # Stats evaluator
    evaluator = Evaluator(name='ogbl-ddi')

    # Initialize a model
    model = models.get_model(args.model)(
        # train_graph.x.shape, train_graph.adj_t.to(device)
        num_nodes=train_graph.num_nodes, adj_t=train_graph.adj_t.to(device)
    )

    # load from checkpoint if path specified
    if args.load_path is not None:
        model = model_utils.load_model(model, args.load_path)
    print(f"Parameters: {model_utils.count_parameters(model)}")

    # Move model to GPU if necessary
    model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=30,
        verbose=True,
    )

    os.makedirs(f'{args.save_path}/{args.experiment}')
    print(f'Created new experiment: {args.experiment}')
    save_arguments(args, f'{args.save_path}/{args.experiment}/args.txt')

    # Train!
    trained_model = train_model(
        train_graph,
        valid_graph,
        train_dl,
        dev_dl,
        evaluator,
        model,
        optimizer,
        scheduler,
        args,
    )

    # Save trained model
    filename = f'{args.save_path}/{args.experiment}/{model.__class__.__name__}_trained.checkpoint'
    model_utils.save_model(trained_model, filename)


if __name__ == '__main__':
    model_utils.verify_versions()
    main()
