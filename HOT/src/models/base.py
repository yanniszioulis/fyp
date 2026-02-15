import torch
import lightning as L

class BaseModel(L.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-2):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        pass

    def step(self, batch, split='train'):
        pass

    def training_step(self, batch, batch_idx):
        return self.step(batch, split='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, split='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, split='test')
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return [optimizer]

