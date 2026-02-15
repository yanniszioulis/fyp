import math

import torch
from torch import nn
import torch.nn.functional as F

from src.models.base import BaseModel
from src.modules.embeddings import LAPE, SinPE3D
from src.modules.transformer import TransformerBlock


class HOTForSurfaceTimeseriesForecasting(BaseModel):
    def __init__(
        self,
        d_hidden,
        d_mlp,
        n_blocks,
        n_head,
        pe="rope",
        patch_size=4,
        context_length=21,
        prediction_length=63,
        attention_type="kronecker_product",
        dropout=0.0,
        lr=1e-3,
        weight_decay=1e-2,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters()

        self.patch_size = patch_size
        self.context_length = context_length
        self.prediction_length = prediction_length

        t_patches = math.ceil(context_length / patch_size)
        if pe in ["nope", "rope"]:
            self.pos_emb = lambda x: torch.zeros_like(x).to(x.device)
        elif pe == "lape":
            self.pos_emb = LAPE(max_size=max(t_patches, 64), d_model=d_hidden, order=3)
        elif pe == "sin":
            self.pos_emb = SinPE3D(max_size=max(t_patches, 64), d_model=d_hidden)
        else:
            raise ValueError(f"Unknown pe={pe!r}")

        self.emb = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=d_hidden, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emb_norm = nn.LayerNorm(d_hidden)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_hidden=d_hidden,
                    d_mlp=d_mlp,
                    n_head=n_head,
                    dropout=dropout,
                    attention_type=attention_type,
                    num_modes=3,
                    rope_dims=[3] if pe == "rope" else [],
                    input_size=t_patches,
                )
                for _ in range(n_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, prediction_length),
        )

    def forward(self, x):
        # x: [bs, H, W, T]
        bs, H, W, T = x.shape

        mu = x.mean(dim=-1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x - mu) / std

        if T % self.patch_size != 0:
            pad = self.patch_size - (T % self.patch_size)
            x_pad = torch.cat([x_norm, x_norm[..., -1:].repeat(1, 1, 1, pad)], dim=-1)
        else:
            x_pad = x_norm

        Tp = x_pad.shape[-1]
        h = x_pad.reshape(bs * H * W, Tp).unsqueeze(1)  # [bs*H*W,1,Tp]
        h = self.emb(h).transpose(1, 2)  # [bs*H*W,Tp',d]
        h = self.emb_norm(h)
        Tp2 = h.shape[1]
        h = h.view(bs, H, W, Tp2, h.shape[-1])  # [bs,H,W,Tp',d]

        h = h + self.pos_emb(h)

        for block in self.blocks:
            h = block(h)

        logits = self.head(h.mean(dim=3))  # [bs,H,W,pred]
        return (logits * std) + mu

    def step(self, batch, split="train"):
        x, y = batch  # [bs,H,W,ctx], [bs,H,W,pred]
        preds = self.forward(x)

        mse = F.mse_loss(preds, y)
        mae = F.l1_loss(preds, y)
        self.log(f"{split}-mse", mse.item())
        self.log(f"{split}-mae", mae.item())
        return mse

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _y = batch
        return self.forward(x)

