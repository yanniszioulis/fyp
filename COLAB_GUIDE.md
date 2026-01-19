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

### 4) Run model

**For ConvLSTM:**
```
%env FYP_DATA_PATH=$data_path
!python run_model.py --model convlstm --save-ckpt --contexts 5,21,63 --horizons 1,5,21
```

**For Transformer:**
```
%env FYP_DATA_PATH=$data_path
!python run_model.py --model transformer --amp --save-ckpt --contexts 5,21,63 --horizons 1,5,21
```

### 5) Pull results back

**What gets saved and tracked by git:**
- ✓ `results/metrics/{model_id}_results.json` - Summary metrics (TRACKED)
- ✓ `models/{model_id}/checkpoints/*.pt` - Model checkpoints (TRACKED, but large files)
- ✗ `results/forecasts/*.npz` - Individual forecast files (NOT tracked - large binaries)
- ✗ `results/plots/*.png` - Plot images (NOT tracked - can regenerate)

**To push results from Colab:**

```python
# Check what changed
!git status

# Add metrics (small JSON files)
!git add results/metrics/

# Add checkpoints (optional - these are large, you may want to skip)
# !git add models/convlstm/checkpoints/

# Commit and push
!git commit -m "Add convlstm results from Colab"
!git push
```

**To pull results locally:**
```bash
git pull
python plot_results.py --model convlstm
```

**Note:** If checkpoints are too large for git, you can:
1. Skip adding them and download manually from Colab
2. Or use Git LFS for large files
3. Or upload checkpoints to Google Drive and download when needed
