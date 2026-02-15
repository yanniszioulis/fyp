import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import re

class TSDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_path,
        name,
        split_sizes=[0.7, 0.1, 0.2],
        context_length=96, 
        prediction_length=96, 
        normalize=True,
        batch_size=16, 
        num_workers=2
    ):
        super().__init__()
        assert sum(split_sizes) == 1.

        self.data_path = data_path
        self.name = name
        self.split_sizes = split_sizes
        self.batch_size = batch_size
        self.context_length = context_length
        self.normalize = normalize
        self.num_workers = num_workers
        self.prediction_length = prediction_length
        self.scaler = StandardScaler()
        self.datasets = self.init()
        

    def init(self):
        ts = np.load(f'{self.data_path}/{self.name}/{self.name}.npy')
        self.n_vars = ts.shape[0]
        self.n_timesteps = ts.shape[1]
        train_size = int(self.split_sizes[0] * self.n_timesteps)
        val_size = int(self.split_sizes[1] * self.n_timesteps)
        test_size = self.n_timesteps - val_size - train_size

        if self.normalize:
            self.scaler.fit(ts[:, :train_size].T)
            ts = self.scaler.transform(ts.T).T

        datasets = {}
        datasets['train'] = MultivarTSDataset(
            data=ts[:, :train_size], 
            context_length=self.context_length, 
            prediction_length=self.prediction_length
        )
        datasets['val'] = MultivarTSDataset(
            data=ts[:, train_size - self.context_length : train_size + val_size], 
            context_length=self.context_length, 
            prediction_length=self.prediction_length
        )
        
        datasets['test'] = MultivarTSDataset(
            data=ts[:, - test_size - self.context_length:], 
            context_length=self.context_length, 
            prediction_length=self.prediction_length
        )
        return datasets
        

    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.datasets['test'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )


class MultivarTSDataset(Dataset):
    def __init__(self, data, context_length, prediction_length):
        super().__init__()
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_vars = data.shape[0]
        self.n_timesteps = data.shape[1]

    def __getitem__(self, index):
        x = torch.tensor(
            self.data[:, index:index + self.context_length]
        )
        y = torch.tensor(
            self.data[:, index + self.context_length : index + self.context_length + self.prediction_length]
        )
        return x.float(), y.float()


    def __len__(self):
        return self.data.shape[1] - self.context_length - self.prediction_length


def _load_spx_iv_tensor(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    iv_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("iv_")]
    if not iv_cols:
        raise ValueError("No iv_* columns found in CSV.")

    pattern = re.compile(r"^iv_([\d.]+)_([\d.]+)$")
    pairs = []
    for c in iv_cols:
        m = pattern.match(c)
        if m is None:
            continue
        pairs.append((float(m.group(1)), float(m.group(2)), c))
    if not pairs:
        raise ValueError("No iv_* columns matched expected iv_<moneyness>_<tau> pattern.")

    m_grid = sorted({p[0] for p in pairs})
    t_grid = sorted({p[1] for p in pairs})
    m_index = {v: i for i, v in enumerate(m_grid)}
    t_index = {v: i for i, v in enumerate(t_grid)}

    n_days = len(df)
    H = len(m_grid)
    W = len(t_grid)
    x = np.full((n_days, H, W), np.nan, dtype=np.float32)
    for m, t, c in pairs:
        i = m_index[m]
        j = t_index[t]
        x[:, i, j] = df[c].to_numpy(dtype=np.float32, copy=False)

    if np.isnan(x).any():
        raise ValueError("Surface tensor has NaNs (missing iv_* columns for some grid points).")

    dates = df["date"].to_numpy()
    meta = {"m_grid": m_grid, "t_grid": t_grid, "iv_cols": [p[2] for p in pairs]}
    return dates, x, meta


class SurfaceTensorDataset(Dataset):
    def __init__(self, data: np.ndarray, context_length: int, prediction_length: int):
        super().__init__()
        self.data = data  # [T, H, W]
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_timesteps = data.shape[0]

    def __getitem__(self, index):
        x = torch.tensor(self.data[index : index + self.context_length])  # [ctx, H, W]
        y = torch.tensor(
            self.data[
                index + self.context_length : index + self.context_length + self.prediction_length
            ]
        )  # [pred, H, W]
        return x.permute(1, 2, 0).float(), y.permute(1, 2, 0).float()  # [H,W,ctx], [H,W,pred]

    def __len__(self):
        return self.data.shape[0] - self.context_length - self.prediction_length + 1


class SPXSurfaceTensorDataModule(L.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        split_sizes=[0.7, 0.1, 0.2],
        context_length=21,
        prediction_length=63,
        normalize=True,
        batch_size=24,
        num_workers=2,
    ):
        super().__init__()
        assert sum(split_sizes) == 1.0
        self.csv_path = csv_path
        self.split_sizes = split_sizes
        self.batch_size = batch_size
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.normalize = normalize
        self.num_workers = num_workers
        self.scaler = StandardScaler()
        self.dates, self.tensor, self.meta = _load_spx_iv_tensor(csv_path)
        self._borders = None
        self.datasets = self.init()

    def init(self):
        x = self.tensor  # [T, H, W]
        T, H, W = x.shape
        # Match PatchTST Dataset_Custom exactly:
        # num_train = int(0.7*T), num_test = int(0.2*T), num_val = T - num_train - num_test
        num_train = int(T * 0.7)
        num_test = int(T * 0.2)
        num_val = T - num_train - num_test

        border1s = [0, num_train - self.context_length, T - num_test - self.context_length]
        border2s = [num_train, num_train + num_val, T]
        self._borders = {
            "T": T,
            "num_train": num_train,
            "num_val": num_val,
            "num_test": num_test,
            "border1s": border1s,
            "border2s": border2s,
        }

        if self.normalize:
            train_flat = x[border1s[0] : border2s[0]].reshape(num_train, H * W)
            self.scaler.fit(train_flat)
            x = self.scaler.transform(x.reshape(T, H * W)).reshape(T, H, W).astype(np.float32, copy=False)

        datasets = {}
        datasets["train"] = SurfaceTensorDataset(
            data=x[border1s[0] : border2s[0]],
            context_length=self.context_length,
            prediction_length=self.prediction_length,
        )
        datasets["val"] = SurfaceTensorDataset(
            data=x[border1s[1] : border2s[1]],
            context_length=self.context_length,
            prediction_length=self.prediction_length,
        )
        datasets["test"] = SurfaceTensorDataset(
            data=x[border1s[2] : border2s[2]],
            context_length=self.context_length,
            prediction_length=self.prediction_length,
        )
        return datasets

    def test_start_dates(self):
        if self._borders is None:
            raise RuntimeError("DataModule not initialized.")
        b1 = self._borders["border1s"][2]
        b2 = self._borders["border2s"][2]
        n_samples = (b2 - b1) - self.context_length - self.prediction_length + 1
        start = b1 + self.context_length
        end = start + n_samples
        return self.dates[start:end]

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
        )
