import argparse
import os

import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # type: ignore

from args import add_train_args, add_experiment, add_common_args, save_arguments
import models
import model_utils


def train_model(
    train_dl: data.DataLoader,
    dev_dl: data.DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
) -> nn.Module:

    device = model_utils.get_device()
    loss_fn = nn.functional.binary_cross_entropy # TODO: Set this to the correct loss fn
    val_loss_fn = model_utils.l1_norm_loss # TODO: Set this to the correct loss fn
    best_val_loss = torch.tensor(float('inf'))
    saved_checkpoints = []
    writer = SummaryWriter(log_dir=f'{args.log_dir}/{args.experiment}')

    for e in range(1, args.train_epochs + 1):
        print(f'Training epoch {e}...')

        # Training portion
        torch.cuda.empty_cache()
        with tqdm(total=args.train_batch_size * len(train_dl)) as progress_bar:
            model.train()
            # TODO: Read correct values from dataloader
            for i, (x_batch, y_batch) in enumerate(train_dl):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward pass on model
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                if args.use_scheduler:
                    lr_scheduler.step(loss)

                progress_bar.update(len(x_batch))
                progress_bar.set_postfix(loss=loss.item())
                writer.add_scalar("train/Loss", loss, ((e - 1) * len(train_dl) + i) * args.train_batch_size)

                del x_batch
                del y_batch
                del y_pred
                del loss

        # Validation portion
        torch.cuda.empty_cache()
        with tqdm(total=args.val_batch_size * len(dev_dl)) as progress_bar:
            model.eval()
            val_loss = 0.0
            num_batches_processed = 0
            # TODO: Read correct values from dataloader
            for i, (x_batch, y_batch) in enumerate(dev_dl):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward pass on model in validation environment
                y_pred = model(x_batch)
                y_pred = torch.round(y_pred)

                loss = val_loss_fn(y_pred, y_batch)

                val_loss += loss.item()
                num_batches_processed += 1

                progress_bar.update(len(x_batch))
                progress_bar.set_postfix(val_loss=val_loss / num_batches_processed)
                writer.add_scalar("Val/Loss", loss, ((e - 1) * len(dev_dl) + i) * args.val_batch_size)

                del x_batch
                del y_batch
                del y_pred
                del loss

            # Save model if it's the best one yet.
            if val_loss / num_batches_processed < best_val_loss:
                best_val_loss = val_loss / num_batches_processed
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
    args = parser.parse_args()
    add_experiment(args)
    device = model_utils.get_device()

    # Load dataset from disk
    # TODO
    train_dl = None # TODO
    dev_dl = None # TODO

    # Initialize a model
    model = models.get_model(args.model)()

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
