import torch
from torch import nn

from src.models.base import BaseModel
from src.modules.embeddings import PatchEmbed, LAPE, SinPE2D
from src.modules.transformer import TransformerBlock
from src.modules.transformer import RMSNorm

class HOTFor2DSegmentation(BaseModel):
    def __init__(
        self, 
        d_hidden,
        d_mlp,
        n_blocks, 
        n_head, 
        n_class,
        pe='rope',
        patch_size=4,
        input_size=128,
        attention_type='kronecker_product',
        dropout=0., 
        lr=1e-3,
        weight_decay=1e-2,
        ce_weight=None,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters()
        self.n_class = n_class
        self.patch_size = patch_size

        input_size = input_size // patch_size
        if pe in ['nope', 'rope']:
            self.pos_emb = lambda x: torch.zeros_like(x).to(x.device)
        elif pe == 'lape':
            self.pos_emb = LAPE(input_size, d_hidden, order=2)
        elif pe == 'sin':
            self.pos_emb = SinPE2D(input_size, d_hidden)

        self.emb = PatchEmbed(7, d_hidden, patch_size, 2, dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_hidden=d_hidden, 
                d_mlp=d_mlp, 
                n_head=n_head, 
                dropout=dropout, 
                attention_type=attention_type, 
                num_modes=2, 
                rope_dims=[1,2] if pe == 'rope' else [], 
                input_size=input_size
            )
             for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_class * patch_size * patch_size)
        )
        self.criterion = nn.CrossEntropyLoss(weight=ce_weight)

    def forward(self, x):                                    
        # x: (bs, c, w, h)
        h = self.emb(x)
        # h: (bs, w, h, d)
        h = h + self.pos_emb(h)
        for block in self.blocks:
            h = block(h)
        h = self.head(h)
        h = h.unflatten(-1, (self.patch_size, self.patch_size, self.n_class)) # bs, w, h, p, p, c
        h = h.permute(0, 1, 3, 2, 4, 5) # bs, w, p, h, p, c
        h = h.flatten(1,2).flatten(2,3)
        return h

    def step(self, batch, split='train'):
        x, y = batch['image'], batch['mask'].squeeze(1)
        logits = self.forward(x)
        loss = self.criterion(logits.permute(0, 3, 1, 2), y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log(f"{split}_loss", loss.item())
        self.log(f"{split}_acc", acc.item())
        return loss
