import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

from src.models.base import BaseModel
from src.modules.embeddings import LAPE, SinPE1D
from src.modules.transformer import TransformerBlock

def mean_pool(x):
    return x.mean(dim=2)

def flatten_pool(x):
    return x.flatten(start_dim=2)

class HOTForTimeseriesForecasting(BaseModel):
    def __init__(
        self, 
        d_hidden,
        d_mlp,
        n_blocks, 
        n_head, 
        pe='rope',
        patch_size=4,
        input_size=96,
        attention_type='kronecker_product',
        dropout=0., 
        lr=1e-3,
        weight_decay=1e-2,
    ):  
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters()
        
        input_size = input_size // patch_size
        if pe in ['nope', 'rope']:
            self.pos_emb = lambda x: torch.zeros_like(x).to(x.device)
        elif pe == 'lape':
            self.pos_emb = LAPE(input_size, d_hidden, order=1)
        elif pe == 'sin':
            self.pos_emb = SinPE1D(input_size, d_hidden)

        self.emb = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=d_hidden, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emb_norm = nn.LayerNorm(d_hidden)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_hidden=d_hidden, 
                d_mlp=d_mlp, 
                n_head=n_head, 
                dropout=dropout, 
                attention_type=attention_type, 
                num_modes=2, 
                rope_dims=[2] if pe == 'rope' else [], 
                input_size=input_size
            )
             for _ in range(n_blocks)
        ])
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 720)
        )


    def forward(self, x):
        bs, n, t = x.shape
        mu = x.mean(2, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x - mu) / std
        
        h = x_norm.view(bs * n, t).unsqueeze(1)  ## (bs * n, 1, t)
        h = self.emb(h).transpose(1, 2)         ## (bs * n, t, d)
        h = self.emb_norm(h)
        h = h.view(bs, n, *h.shape[1:])          ## (bs, n, t, d)
        pos = self.pos_emb(h[:, 0]).unsqueeze(1).repeat(1, n, 1, 1)
        h = h + pos
        
        for block in self.blocks:
            h = block(h)

        logits = self.head(h.mean(dim=2))
        return (logits * std) + mu

    def calc_metrics(self, preds, labels):  
        return F.mse_loss(preds, labels), F.l1_loss(preds, labels)
    

    def step(self, batch, split='train'):
        x, y = batch
        preds = self.forward(x)
        
        avg_mse = 0
        avg_mae = 0
        horizons = [96, 192, 336, 720]
        for l in horizons:
            mse, mae = self.calc_metrics(preds[:, :, :l], y[:, :, :l])
            self.log(f"{split}-mse-{l}", mse.item())
            self.log(f"{split}-mae-{l}", mae.item())
            avg_mse += mse
            avg_mae += mae

        avg_mse /= len(horizons)
        avg_mae /= len(horizons)
        self.log(f"{split}-avg-mse", avg_mse.item())
        self.log(f"{split}-avg-mae", avg_mae.item())
        return avg_mse
