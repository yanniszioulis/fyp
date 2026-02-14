"""
Persistence (naive) baseline for SPX IV surface forecasting.
Prediction = repeat the last day of the context window for all 63 forecast steps.
Uses the exact same data pipeline, scaling, and test split as PatchTST.
Also computes IC (Information Coefficient) for both PatchTST and persistence.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

# --- Config (must match SPX_IV.sh) ---
seq_len = 21
pred_len = 63
label_len = 0
features = 'M'
target = 'iv_1.1_1.0'

# --- Load data (same as Dataset_Custom) ---
df_raw = pd.read_csv('./dataset/SPX_surfaces.csv')

# Reorder columns: date, ..., target (same as Dataset_Custom.__read_data__)
cols = list(df_raw.columns)
cols.remove(target)
cols.remove('date')
df_raw = df_raw[['date'] + cols + [target]]

# Train/val/test split (same as Dataset_Custom)
num_train = int(len(df_raw) * 0.7)
num_test = int(len(df_raw) * 0.2)
num_vali = len(df_raw) - num_train - num_test

border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
border2s = [num_train, num_train + num_vali, len(df_raw)]

print(f"Total rows: {len(df_raw)}")
print(f"Train: {num_train}, Val: {num_vali}, Test: {num_test}")
print(f"Test borders: [{border1s[2]}, {border2s[2]})")

# Features = M: use all columns except 'date'
df_data = df_raw[df_raw.columns[1:]]
n_features = df_data.shape[1]
print(f"Number of features (channels): {n_features}")

# Scale (fit on training data only, same as Dataset_Custom)
scaler = StandardScaler()
train_data = df_data[border1s[0]:border2s[0]]
scaler.fit(train_data.values)
data = scaler.transform(df_data.values)

# Extract test portion (same indexing as Dataset_Custom)
test_border1 = border1s[2]
test_border2 = border2s[2]
test_data = data[test_border1:test_border2]

n_test_samples = len(test_data) - seq_len - pred_len + 1
print(f"Test samples: {n_test_samples}")

# --- Build persistence predictions and ground truth ---
persist_preds = []
trues = []

for i in range(n_test_samples):
    last_day = test_data[i + seq_len - 1]  # shape: [400]
    pred = np.tile(last_day, (pred_len, 1))  # shape: [63, 400]
    true = test_data[i + seq_len : i + seq_len + pred_len]  # shape: [63, 400]
    persist_preds.append(pred)
    trues.append(true)

persist_preds = np.array(persist_preds)  # shape: [776, 63, 400]
trues = np.array(trues)                  # shape: [776, 63, 400]

# --- Load PatchTST predictions ---
patchtst_preds = np.load(
    './results/SPX_IV_21_63_PatchTST_custom_ftM_sl21_ll0_pl63_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0/pred.npy'
)
print(f"\nPersistence preds shape: {persist_preds.shape}")
print(f"PatchTST preds shape:   {patchtst_preds.shape}")
print(f"Ground truth shape:     {trues.shape}")

# PatchTST DataLoader uses drop_last=True with batch_size=24,
# so it evaluated only the first 768 of 776 samples (32 full batches).
# Trim persistence & ground truth to match so all metrics are comparable.
n_patchtst = patchtst_preds.shape[0]
persist_preds = persist_preds[:n_patchtst]
trues = trues[:n_patchtst]
n_test_samples = n_patchtst
print(f"\nAligned to {n_test_samples} samples (drop_last=True, batch_size=24)")
print(f"  Persistence: {persist_preds.shape}, PatchTST: {patchtst_preds.shape}, Truth: {trues.shape}")

# --- Metric functions ---
def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def compute_IC(pred, true):
    """
    Cross-sectional IC (Information Coefficient):
    For each (sample, forecast_step), compute the Spearman rank correlation
    between predicted and actual values across the 400 IV features.
    Returns: overall mean IC, std of IC, and per-horizon mean IC.
    
    pred, true: shape [n_samples, pred_len, n_features]
    """
    n_samples, n_steps, n_feat = pred.shape
    ic_values = np.zeros((n_samples, n_steps))
    
    for i in range(n_samples):
        for t in range(n_steps):
            corr, _ = stats.spearmanr(pred[i, t, :], true[i, t, :])
            ic_values[i, t] = corr
    
    overall_ic = np.nanmean(ic_values)
    overall_std = np.nanstd(ic_values)
    per_horizon_ic = np.nanmean(ic_values, axis=0)  # shape: [pred_len]
    
    return overall_ic, overall_std, per_horizon_ic, ic_values

# --- Compute all metrics ---
print("\nComputing metrics...")

# Persistence
p_mse = MSE(persist_preds, trues)
p_mae = MAE(persist_preds, trues)
p_rse = RSE(persist_preds, trues)
print("Computing Persistence IC...")
p_ic, p_ic_std, p_ic_horizon, p_ic_all = compute_IC(persist_preds, trues)

# PatchTST
t_mse = MSE(patchtst_preds, trues)
t_mae = MAE(patchtst_preds, trues)
t_rse = RSE(patchtst_preds, trues)
print("Computing PatchTST IC...")
t_ic, t_ic_std, t_ic_horizon, t_ic_all = compute_IC(patchtst_preds, trues)

# --- Print results ---
print(f"\n{'='*70}")
print(f"{'Metric':<12} {'Persistence':>16} {'PatchTST':>16}")
print(f"{'='*70}")
print(f"{'MSE':<12} {p_mse:>16.10f} {t_mse:>16.10f}")
print(f"{'MAE':<12} {p_mae:>16.10f} {t_mae:>16.10f}")
print(f"{'RSE':<12} {p_rse:>16.10f} {t_rse:>16.10f}")
print(f"{'IC (mean)':<12} {p_ic:>16.10f} {t_ic:>16.10f}")
print(f"{'IC (std)':<12} {p_ic_std:>16.10f} {t_ic_std:>16.10f}")
print(f"{'='*70}")

# --- Per-horizon IC breakdown ---
print(f"\nPer-horizon IC (averaged across {n_test_samples} test samples):")
print(f"{'Horizon':<10} {'Persistence':>14} {'PatchTST':>14}")
print(f"{'-'*40}")
horizons_to_show = [1, 5, 10, 21, 42, 63]
for h in horizons_to_show:
    if h <= pred_len:
        print(f"  t+{h:<6} {p_ic_horizon[h-1]:>14.6f} {t_ic_horizon[h-1]:>14.6f}")
print(f"{'-'*40}")
