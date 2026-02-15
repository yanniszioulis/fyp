import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from medmnist import (
    OrganMNIST3D, 
    NoduleMNIST3D, 
    AdrenalMNIST3D, 
    FractureMNIST3D, 
    VesselMNIST3D, 
    SynapseMNIST3D,
    Evaluator
)

from src.models import MedMNISTHOT


class Transform3D:
    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):
   
        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()
        
        return voxel.astype(np.float32)


L.seed_everything(43)

def main():
    parser = argparse.ArgumentParser(
        "HOT-medmnist", add_help=False
    )
    parser.add_argument(
        "--name", default='organ', type=str, help="dataset"
    )
    parser.add_argument(
        "--data_path", default='data/medmnist3d', type=str, help="dataset path"
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
    args = parser.parse_args()

    ## Dataset
    train_transform = Transform3D(mul='random')
    eval_transform = Transform3D(mul='0.5')
    data_cls = {
        'fracture' : FractureMNIST3D,
        'organ' : OrganMNIST3D, 
        'nodule' : NoduleMNIST3D, 
        'adrenal' : AdrenalMNIST3D, 
        'vessel' : VesselMNIST3D, 
        'synapse' : SynapseMNIST3D
    }[args.name]
    train_dataset = data_cls(split="train", transform=train_transform, download=True, root=args.data_path)
    val_dataset = data_cls(split="val", transform=eval_transform, download=True, root=args.data_path)
    test_dataset = data_cls(split="test", transform=eval_transform, download=True, root=args.data_path)
    num_classes = len(train_dataset.info['label'])
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True,
        batch_size=128,
        num_workers=2,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=128)
    test_dataloader = DataLoader(test_dataset, batch_size=128)

    weight = np.histogram(train_dataset.labels, bins=num_classes)[0]
    weight = weight.sum() / weight
    weight = torch.from_numpy(weight / weight.sum()).cuda().float()
    
    evaluators = {
        'train': Evaluator(flag=data_cls.flag, split='train', root=args.data_path),
        'val':  Evaluator(flag=data_cls.flag, split='test', root=args.data_path),
        'test' :  Evaluator(flag=data_cls.flag, split='val', root=args.data_path)
    }
    
    ## Model
    model = MedMNISTHOT(
        d_hidden=args.d_hidden,
        d_mlp=args.d_mlp,
        n_blocks=args.num_blocks, 
        n_head=args.num_heads, 
        patch_size=args.patch_size,
        pe=args.pe,
        n_class=num_classes,
        evaluators=evaluators,
        attention_type=args.attention_type,
        dropout=args.dropout, 
        lr=args.lr,
        weight_decay=args.weight_decay,
        ce_weight=weight
    )

    ## Trainer
    wandb_logger = WandbLogger(
        project='HOT-MedMNIST3D'
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
        num_sanity_val_steps=0,
        enable_progress_bar=False
    )
    trainer.fit(model, train_dataloader, val_dataloader) 
    trainer.test(model, test_dataloader)


if __name__ == '__main__':
    main()

