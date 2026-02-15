import argparse

import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.ts_data import TSDataModule, SPXSurfaceTensorDataModule
from src.models import HOTForTimeseriesForecasting, HOTForSurfaceTimeseriesForecasting
import os
import numpy as np
import torch

L.seed_everything(43)


def main():
    parser = argparse.ArgumentParser(
         "HOT-timeseries", add_help=False
    )
    parser.add_argument(
        "--name", default='weather', type=str, help="dataset name"
    )
    parser.add_argument(
        "--data_path", default='data/timeseries', type=str, help="dataset path"
    )
    parser.add_argument(
        "--csv_path",
        default="",
        type=str,
        help="CSV path for SPX surfaces (defaults to HOT/dataset/SPX_surfaces.csv)",
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
        "--patch_size", default=4, type=int, help="patch size"
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
    parser.add_argument(
        "--logger",
        default="wandb",
        type=str,
        choices=["wandb", "csv", "none"],
        help="logger backend",
    )
    args = parser.parse_args()

    if args.name.lower() in {"spx_iv", "spx", "spx_surfaces"}:
        csv_path = args.csv_path.strip()
        if not csv_path:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cand_hot = os.path.join(repo_root, "HOT", "dataset", "SPX_surfaces.csv")
            csv_path = cand_hot

        datamod = SPXSurfaceTensorDataModule(
            csv_path=csv_path,
            split_sizes=[0.7, 0.1, 0.2],
            context_length=21,
            batch_size=24,
            prediction_length=63,
            normalize=True,
            num_workers=2,
        )

        model = HOTForSurfaceTimeseriesForecasting(
            d_hidden=args.d_hidden,
            d_mlp=args.d_mlp,
            n_blocks=args.num_blocks,
            n_head=args.num_heads,
            patch_size=args.patch_size,
            context_length=21,
            prediction_length=63,
            attention_type=args.attention_type,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        datamod = TSDataModule(
            data_path=args.data_path,
            name=args.name,
            split_sizes=[0.7, 0.1, 0.2],
            context_length=96,
            batch_size=128,
            prediction_length=720,
            normalize=True,
            num_workers=2,
        )

        model = HOTForTimeseriesForecasting(
            d_hidden=args.d_hidden,
            d_mlp=args.d_mlp,
            n_blocks=args.num_blocks,
            n_head=args.num_heads,
            patch_size=args.patch_size,
            attention_type=args.attention_type,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    ## Trainer
    if args.logger == "wandb":
        try:
            logger = WandbLogger(project="HOT-Timeseries")
        except ModuleNotFoundError:
            logger = CSVLogger(save_dir="logs", name="HOT-Timeseries")
    elif args.logger == "csv":
        logger = CSVLogger(save_dir="logs", name="HOT-Timeseries")
    else:
        logger = None

    monitor_metric = "val-mae" if args.name.lower() in {"spx_iv", "spx", "spx_surfaces"} else "val-avg-mae"
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric, min_delta=0.005, patience=10, verbose=False, mode="min"
    )
    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        devices=1,
        accelerator="gpu", 
        num_nodes=1,
        logger=logger,
        callbacks=[early_stop_callback],
        accumulate_grad_batches=1,
        gradient_clip_val=1.,
        enable_progress_bar=False
    )
    trainer.fit(model, datamod.train_dataloader(), datamod.val_dataloader()) 
    trainer.test(model, datamod.test_dataloader())

    if args.name.lower() in {"spx_iv", "spx", "spx_surfaces"}:
        preds_batches = trainer.predict(model, dataloaders=datamod.test_dataloader())
        preds = torch.cat([p.detach().cpu() for p in preds_batches], dim=0).numpy()  # [N,H,W,pred]
        start_dates = datamod.test_start_dates()

        setting = (
            f"SPX_IV_21_63_HOT_tensor_"
            f"dh{args.d_hidden}_mlp{args.d_mlp}_b{args.num_blocks}_h{args.num_heads}_"
            f"p{args.patch_size}_{args.attention_type}"
        )
        out_dir = os.path.join("HOT", "results", setting)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "pred.npy"), preds.astype(np.float32, copy=False))
        np.save(os.path.join(out_dir, "start_dates.npy"), start_dates)

if __name__ == '__main__':
    main()