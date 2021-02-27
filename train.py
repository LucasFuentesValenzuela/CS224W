import argparse
import os
from typing import Dict

import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # type: ignore
import torch_geometric as pyg
from torch_geometric.utils import negative_sampling

from args import add_model_args, add_train_args, add_experiment, add_common_args, save_arguments
import models
import model_utils


def train_model(
    train_graph: pyg.torch_geometric.data.Data,
    valid_graph: pyg.torch_geometric.data.Data,
    train_dl: data.DataLoader,
    dev_dl: data.DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
) -> nn.Module:

    device = model_utils.get_device()
    loss_fn = nn.functional.binary_cross_entropy # TODO: Set this to the correct loss fn
    val_loss_fn = nn.functional.binary_cross_entropy # TODO: Set this to the correct loss fn
    best_val_loss = torch.tensor(float('inf'))
    saved_checkpoints = []
    writer = SummaryWriter(log_dir=f'{args.log_dir}/{args.experiment}')

    for e in range(1, args.train_epochs + 1):
        print(f'Training epoch {e}...')

        # Training portion
        torch.cuda.empty_cache()
        with tqdm(total=args.train_batch_size * len(train_dl)) as progress_bar:
            model.train()

            # Load graph into GPU
            adj_t = train_graph.adj_t.to(device)
            edge_index = train_graph.edge_index.to(device)

            for i, (y_pos_edges,) in enumerate(train_dl):
                y_pos_edges = y_pos_edges.to(device).T
                y_neg_edges = negative_sampling(
                    edge_index,
                    num_nodes=train_graph.num_nodes,
                    num_neg_samples=y_pos_edges.shape[1]
                ).to(device)
                y_batch = torch.cat([torch.ones(y_pos_edges.shape[1]), torch.zeros(y_neg_edges.shape[1])], dim=0).to(device) # Ground truth edge labels (1 or 0)

                # Forward pass on model
                optimizer.zero_grad()
                y_pred = model(train_graph.x, adj_t, torch.cat([y_pos_edges, y_neg_edges], dim=1))
                loss = loss_fn(y_pred, y_batch)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                if args.use_scheduler:
                    lr_scheduler.step(loss)

                progress_bar.update(y_pos_edges.shape[1])
                progress_bar.set_postfix(loss=loss.item())
                writer.add_scalar("train/Loss", loss, ((e - 1) * len(train_dl) + i) * args.train_batch_size)

                del y_pos_edges
                del y_neg_edges
                del y_pred
                del loss

            del adj_t
            del edge_index

        # Validation portion
        torch.cuda.empty_cache()
        with tqdm(total=args.val_batch_size * len(dev_dl)) as progress_bar:
            model.eval()

            adj_t = valid_graph.adj_t.to(device)
            edge_index = valid_graph.edge_index.to(device)

            val_loss = 0.0
            accuracy = 0
            num_samples_processed = 0
            for i, (edges_batch, y_batch) in enumerate(dev_dl):
                edges_batch = edges_batch.T.to(device)
                y_batch = y_batch.to(device)

                # Forward pass on model in validation environment
                y_pred = model(adj_t, edges_batch)
                y_pred = torch.round(y_pred)
                loss = val_loss_fn(y_pred, y_batch)

                num_samples_processed += edges_batch.shape[1]
                accuracy += torch.sum(1 - torch.abs(y_batch - torch.round(y_pred))).item()
                val_loss += loss.item() * edges_batch.shape[1]

                progress_bar.update(edges_batch.shape[1])
                progress_bar.set_postfix(val_loss=val_loss / num_samples_processed, acc=accuracy/num_samples_processed)
                writer.add_scalar("Val/Loss", loss, ((e - 1) * len(dev_dl) + i) * args.val_batch_size)

                del edges_batch
                del y_batch
                del y_pred
                del loss

            del adj_t
            del edge_index

            # Save model if it's the best one yet.
            if val_loss / num_samples_processed < best_val_loss:
                best_val_loss = val_loss / num_samples_processed
                filename = f'{args.save_path}/{args.experiment}/{model.__class__.__name__}_best_val.checkpoint'
                model_utils.save_model(model, filename)
                print(f'Model saved!')
                print(f'Best validation loss yet: {best_val_loss}')
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
    add_model_args(parser)
    args = parser.parse_args()
    add_experiment(args)
    device = model_utils.get_device()

    # Load dataset from disk
    print('Loading train data...')
    train_graph, valid_graph, eval_edges, valid_edges = model_utils.load_training_data()
    train_dl = data.DataLoader(
        data.TensorDataset(eval_edges['edge']),
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    dev_dl = data.DataLoader(
        data.TensorDataset(
            torch.cat([valid_edges['edge'], valid_edges['edge_neg']], dim=0),
            torch.cat([torch.ones(valid_edges['edge'].shape[0]), torch.zeros(valid_edges['edge_neg'].shape[0])], dim=0),
        ),
        batch_size=args.val_batch_size,
        shuffle=True,
    )

    # Initialize node embeddings
    print('Computing initial embeddings')
    train_graph = model_utils.initialize_embeddings(train_graph, 'train_embeddings.pt', args.refresh_embeddings)
    valid_graph = model_utils.initialize_embeddings(valid_graph, 'valid_embeddings.pt', args.refresh_embeddings)

    # Initialize a model
    model = models.get_model(args.model)(train_graph.x.shape, args)

    # load from checkpoint if path specified
    if args.load_path is not None:
        model = model_utils.load_model(model, args.load_path)

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
