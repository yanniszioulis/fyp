import os
import random
import torch
import argparse

from torch.utils.data import DataLoader, random_split

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from torchgeo.datasets import SSL4EOLBenchmark

from src.models import HOTFor2DSegmentation

L.seed_everything(43)


def main():
    parser = argparse.ArgumentParser(
        "HOT-Segmentation", add_help=False
    )
    parser.add_argument(
        "--data_path", default='data/ssl4eo-l', type=str, help="dataset path"
    )
    parser.add_argument(
        "--task", default='nlcd', type=str, help="nlcd or cdl"
    )
    parser.add_argument(
        "--patch_size", default=4, type=int, help="patch size"
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="dropout"
    )
    parser.add_argument(
        "--weight_decay", default=1e-2, type=float, help="weight decay"
    )
    parser.add_argument(
        "--d_hidden", default=128, type=int, help="hidden dimension"
    )
    parser.add_argument(
        "--d_mlp", default=512, type=int, help="MLP dimension"
    )
    parser.add_argument(
        "--num_blocks", default=4, type=int, help="number of blocks"
    )
    parser.add_argument(
        "--num_heads", default=8, type=int, help="number of attention heads"
    )
    parser.add_argument(
        "--attention_type", default='kronecker_product', type=str, help="attention type"
    )
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="learning rate"
    )
    parser.add_argument(
        "--num_epochs", default=50, type=int, help="max number of epochs"
    )
    args = parser.parse_args()

    # Load the dataset
    train_dataset = SSL4EOLBenchmark(
        root=args.data_path, 
        sensor='oli_sr', 
        product=args.task, 
        split='train'
    )
    test_dataset = SSL4EOLBenchmark(
        root=args.data_path, 
        sensor='oli_sr', 
        product=args.task, 
        split='test'
    )

    train_data, test_data = torch.load('/home/mila/s/soroush.omranpour/scratch/SSL4EOLBenchmark-cdl.pt', weights_only=False)
    n = len(train_dataset)
    t = n // 10
    train_dataset, val_dataset = random_split(train_dataset, [n-t, t])

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=96, num_workers=6)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=96, num_workers=6)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=96, num_workers=6)

    ## Load the model
    model = HOTFor2DSegmentation(
        d_hidden=args.d_hidden,
        d_mlp=args.d_mlp,
        n_blocks=args.num_blocks, 
        n_head=args.num_heads, 
        patch_size=args.patch_size,
        attention_type=args.attention_type,
        pe=args.pe,
        n_class=134 if args.task == 'cdl' else 17,
        mode=args.mode,
        dropout=args.dropout, 
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ## Trainer
    wandb_logger = WandbLogger(
        project='HOT-SSL4EO',
    )
    early_stop_callback = EarlyStopping(
        monitor="val_acc", min_delta=0.01, patience=20, verbose=False, mode="max"
    )
    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        devices=1,
        accelerator="gpu", 
        num_nodes=1,
        logger=wandb_logger,
        callbacks=[early_stop_callback],
        accumulate_grad_batches=1,
        gradient_clip_val=1.,
        enable_progress_bar=False
    )
    trainer.fit(model, train_dataloader, val_dataloader) 
    trainer.test(model, test_dataloader)
