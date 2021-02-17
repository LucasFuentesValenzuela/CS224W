import argparse
import os

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # type: ignore

from args import add_test_args, add_common_args
import models
import model_utils


def test_model(
    dev_dl: data.DataLoader,
    model: nn.Module,
    args: argparse.Namespace,
) -> nn.Module:

    device = model_utils.get_device()
    loss_fn = model_utils.l1_norm_loss # TODO: Use correct loss fn

    print('\nRunning test metrics...')

    # Forward inference on model
    print('  Running forward inference...')
    num_correct = 0
    total = 0
    with tqdm(total=args.batch_size * len(dev_dl)) as progress_bar:
        # TODO: read correct values from dataloader
        for i, (x_batch, y_batch) in enumerate(dev_dl):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass on model
            y_pred = model(x_batch)

            # TODO: Process y_pred in the optimal way (round it off, etc)
            y_pred = torch.round(y_pred)

            # TODO: Log statistics
            loss = loss_fn(y_pred, y_batch).item()
            num_correct += torch.sum(y_pred == y_batch).item()
            total += len(len(x_batch))

            progress_bar.update(len(x_batch))

            del x_batch
            del y_pred
            del y_batch


    print(f'\n  Calculating overall metrics...')
    print()
    print('*' * 30)
    print(f'Accuracy: {num_correct / total}')
    print('*' * 30)

    return model


def main():
    parser = argparse.ArgumentParser()
    add_test_args(parser)
    add_common_args(parser)
    args = parser.parse_args()

    device = model_utils.get_device()

    # Load dataset from disk
    # TODO: Load test data
    dev_dl = None # TODO: make dataloader object

    # Initialize a model
    model = models.get_model(args.model)()

    # load from checkpoint if path specified
    assert args.load_path is not None
    model = model_utils.load_model(model, args.load_path)
    model.eval()

    # Move model to GPU if necessary
    model.to(device)

    # test!
    test_model(
        dev_dl,
        model,
        args,
    )


if __name__ == '__main__':
    model_utils.verify_versions()
    main()


