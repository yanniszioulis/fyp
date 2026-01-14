## Colab Guide

This repo is set up to run tuning on Colab. Follow these steps.

### 1) Clone the repo

```
!git clone <YOUR_REPO_URL>
%cd <REPO_FOLDER>
```

### 2) Install dependencies

```
!pip install -r requirements.txt
```

### 3) Provide the data

The data CSV is ignored by git. Use Google Drive or upload directly.

**Option A: Google Drive**
```
from google.colab import drive
drive.mount('/content/drive')
data_path = "/content/drive/MyDrive/SPX_IV_fixed_grid.csv"
```

**Option B: Upload**
```
from google.colab import files
uploaded = files.upload()
data_path = "SPX_IV_fixed_grid.csv"
```

### 4) Run tuning

```
%env FYP_DATA_PATH=$data_path
!python run_model.py --model transformer --amp --save-ckpt --contexts 5,21,63 --horizons 1,5,21
```

Defaults are: window 0, context 21, horizon 5, and grid loaded from
`tuning/transformer_grid.json`.

### 5) Pull results back

Tuning outputs are saved to `results/tuning/` and are tracked by git.
Commit and push from Colab:

```
!git status
!git add results/tuning
!git commit -m "Add transformer tuning results"
!git push
```
