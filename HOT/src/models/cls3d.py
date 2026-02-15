import torch
from torch import nn
import torch.nn.functional as F

from src.models.base import BaseModel
from src.modules.embeddings import PatchEmbed, LAPE, SinPE3D
from src.modules.transformer import TransformerBlock


class HOTFor3DClassification(BaseModel):
    def __init__(
        self, 
        d_hidden,
        d_mlp,
        n_blocks, 
        n_head, 
        n_class,
        evaluators,
        pe='rope',
        patch_size=4,
        input_size=28,
        attention_type='kronecker_product',
        dropout=0., 
        lr=1e-3,
        weight_decay=1e-2,
        ce_weight=None,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters()

        input_size = input_size // patch_size
        if pe in ['nope', 'rope']:
            self.pos_emb = lambda x: torch.zeros_like(x).to(x.device)
        elif pe == 'lape':
            self.pos_emb = LAPE(input_size, d_hidden, order=3)
        elif pe == 'sin':
            self.pos_emb = SinPE3D(input_size, d_hidden)

        self.emb = PatchEmbed(1, d_hidden, patch_size, 3, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_hidden=d_hidden, 
                d_mlp=d_mlp, 
                n_head=n_head, 
                dropout=dropout, 
                attention_type=attention_type, 
                num_modes=3,
                rope_dims=[1,2,3] if pe == 'rope' else [], 
                input_size=input_size
            ) 
            for _ in range(n_blocks)
        ])
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(
                d_hidden, 
                n_class
            )
        )
        self.criterion = nn.CrossEntropyLoss(weight=ce_weight)
        self.evaluators = evaluators
        self.step_outputs = {'train' : [], 'val':[], 'test':[]}
        
    def forward(self, x):                                    
        # x: (bs, 1, w, h, t)
        h = self.emb(x)
        # h: (bs, w, h, t, d)
        h = h + self.pos_emb(h)
        for block in self.blocks:
            h = block(h)
        return self.head(h.mean((1,2,3)))
    

    def step(self, batch, split='train'):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y.squeeze(1))
        self.log(f"{split}_loss", loss.item())
        
        probs = F.softmax(logits, dim=-1)
        self.step_outputs[split].append(probs)
        return loss

    def on_epoch_end(self, split):
        all_preds = torch.cat(self.step_outputs[split], dim=0).detach().cpu().numpy()
        auc, acc = self.evaluators[split].evaluate(all_preds)
        self.log(f"{split}_auc", auc)
        self.log(f"{split}_acc", acc)
        self.step_outputs[split].clear()

    def on_train_epoch_end(self):
        self.on_epoch_end(split='train')

    def on_validation_epoch_end(self):
        self.on_epoch_end(split='val')

    def on_test_epoch_end(self):
        self.on_epoch_end(split='test')