# ConvLSTM Gradient Flow Analysis

## Forward Pass Path

### Input
- **Shape**: `(batch, context_length=21, channels=1, height=20, width=20)`
- **Values**: Raw IV [0.0591, 0.5403] (no normalization)
- **Example**: Mean ~0.19, Std ~0.07

### Layer-by-Layer Flow

#### 1. ConvLSTM Layer 1 (Single Layer Architecture)
For each timestep t in [0, 20]:
- **Input**: `x_t` shape `(batch, 1, 20, 20)`, values [0.0591, 0.5403]
- **Conv2d Operations** (kernel_size=3, stride=1, padding=0):
  - `Wxi(x_t)`: Input-to-input gate → `(batch, 64, 18, 18)` (spatial reduction!)
  - `Wxf(x_t)`: Input-to-forget gate → `(batch, 64, 18, 18)`
  - `Wxg(x_t)`: Input-to-candidate → `(batch, 64, 18, 18)`
  - `Wxo(x_t)`: Input-to-output gate → `(batch, 64, 18, 18)`
  
- **Hidden State Convolutions** (kernel_size=1, no spatial change):
  - `Whi(h_{t-1})`: Hidden-to-input gate → `(batch, 64, 18, 18)`
  - `Whf(h_{t-1})`: Hidden-to-forget gate → `(batch, 64, 18, 18)`
  - `Whg(h_{t-1})`: Hidden-to-candidate → `(batch, 64, 18, 18)`
  - `Who(h_{t-1})`: Hidden-to-output gate → `(batch, 64, 18, 18)`

- **Gates**:
  - `it = sigmoid(Wxi(x_t) + Whi(h_{t-1}))` → [0, 1]
  - `ft = sigmoid(Wxf(x_t) + Whf(h_{t-1}))` → [0, 1]
  - `gt = tanh(Wxg(x_t) + Whg(h_{t-1}))` → [-1, 1]
  - `ot = sigmoid(Wxo(x_t) + Who(h_{t-1}))` → [0, 1]

- **Cell State Update**:
  - `Ct = ft * c_{t-1} + it * gt`
  - **Issue**: `ft` and `it` are [0,1], `gt` is [-1,1], but `c_{t-1}` starts at 0
  - Initial cell state is all zeros, so first update: `Ct = it * gt` (only candidate matters)

- **Hidden State**:
  - `Ht = ot * tanh(Ct)`
  - **Issue**: `ot` is [0,1], `tanh(Ct)` is [-1,1], so `Ht` is [-1,1] initially

#### 2. Final Convolution
- **Input**: Last timestep's hidden state `(batch, 64, 18, 18)`
- **Conv2d**: `(batch, 64, 18, 18)` → `(batch, 1, 20, 20)` (kernel=1, stride=1, padding=1)
- **Output**: `(batch, 1, 20, 20)` - **Note**: padding=1 compensates for dimension reduction!
- **Issue**: This is a workaround - better to use padding=1 in ConvLSTM layers to maintain (20,20) throughout

#### 3. ReLU Activation
- `out = relu(final_conv_out)`
- **Effect**: Clips negative values to 0
- **Bias**: Initialized to 0.2, so outputs start positive

## Gradient Flow (Backward Pass)

### Loss Function: RMSE
```
loss = sqrt(mean((pred - target)^2))
d_loss/d_pred = (pred - target) / (2 * sqrt(mean((pred - target)^2)))
```

### Gradient Path

1. **Loss → Final Conv**
   - Gradient flows through ReLU (gradient = 1 if input > 0, else 0)
   - **Issue**: If predictions are negative before ReLU, gradients are zero!
   - Flows to final_convout weights and bias

2. **Final Conv → ConvLSTM Layer**
   - Gradients flow back through time (BPTT)
   - For each timestep t = 21, 20, ..., 1:
     - Gradients w.r.t. `Ht` flow to:
       - `ot` gate: `dHt/dot = tanh(Ct)`
       - `Ct`: `dHt/dCt = ot * (1 - tanh(Ct)^2)`
     - Gradients w.r.t. `Ct` flow to:
       - `ft` gate: `dCt/dft = c_{t-1}`
       - `it` gate: `dCt/dit = gt`
       - `gt`: `dCt/dgt = it`
       - `c_{t-1}`: `dCt/dc_{t-1} = ft` (gradient multiplier!)

3. **Gates → Convolutions**
   - `dit/dWxi = sigmoid'(...) * x_t` (sigmoid derivative is small!)
   - `dit/dWhi = sigmoid'(...) * h_{t-1}`

### Potential Gradient Issues

1. **Vanishing Gradients**:
   - Sigmoid derivatives: `sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))` → max 0.25
   - Tanh derivatives: `tanh'(x) = 1 - tanh(x)^2` → max 1.0
   - **Problem**: With 21 timesteps, gradients multiply through many sigmoid gates
   - **Effect**: Early timesteps receive very small gradients

2. **Spatial Dimension Reduction**:
   - Input: (20, 20)
   - After Conv2d(kernel=3, padding=0): (18, 18)
   - Final conv with padding=1: (20, 20) - **Workaround that works but is suboptimal**
   - **Better**: Use padding=1 in ConvLSTM layers to maintain (20,20) throughout
   - **Current**: Information loss at edges due to dimension reduction

3. **Initialization Issues**:
   - Cell states start at zero
   - Hidden states start at zero
   - First forward pass: `Ct = it * gt` (only candidate gate matters)
   - If `it` is small (sigmoid near 0), cell state barely updates

4. **Input Scale Issues**:
   - Raw IV values [0.059, 0.540], mean ~0.19, std ~0.07
   - Conv2d weights initialized with default (Kaiming/He initialization)
   - **Problem**: Default initialization assumes inputs ~N(0,1) or uniform
   - Our inputs are positive, mean-centered around 0.19 (not zero-mean!)
   - **Gradient Flow Impact**:
     - Sigmoid gates: `sigmoid(Wx + b)` where W initialized for zero-mean inputs
     - With mean=0.19 inputs, gates may saturate (sigmoid → 0 or 1)
     - Saturated gates → small gradients → slow learning
     - Example: If `Wxi(x) + Whi(h)` is large, `it = sigmoid(...)` → 1.0, derivative → 0
   - **Why this hurts memorization**:
     - Model can't make fine adjustments when gates are saturated
     - Needs precise control to memorize 7 samples exactly
     - Normalized inputs would help gates stay in active region

## Normalization Options

### Option 1: Min-Max Normalization (Like PI-ConvTF for S/K)
```python
# Normalize to [0, 1]
X_min = X_train.min()
X_max = X_train.max()
X_normalized = (X_train - X_min) / (X_max - X_min)

# Denormalize predictions
predictions_denorm = predictions * (X_max - X_min) + X_min
```

**Pros**:
- Values in [0, 1] range
- Preserves relative relationships
- Easy to denormalize

**Cons**:
- Sensitive to outliers
- Different min/max per window → different scaling
- Need to store min/max for denormalization

### Option 2: Standardization (Z-score)
```python
# Normalize to mean=0, std=1
X_mean = X_train.mean()
X_std = X_train.std()
X_normalized = (X_train - X_mean) / X_std

# Denormalize predictions
predictions_denorm = predictions * X_std + X_mean
```

**Pros**:
- Standard neural network practice
- Better for gradient flow
- Less sensitive to outliers

**Cons**:
- Can produce negative values (but ReLU fixes this)
- Different mean/std per window
- Need to store mean/std

### Option 3: Per-Sample Normalization
```python
# Normalize each surface independently
for i in range(n_samples):
    surface = X_train[i]
    X_train[i] = (surface - surface.min()) / (surface.max() - surface.min())
```

**Pros**:
- Each surface normalized independently
- Handles different volatility regimes

**Cons**:
- Loses absolute scale information
- Harder to interpret
- May not help with memorization

### Option 4: Global Normalization (Across All Data)
```python
# Use global statistics from all training data
global_mean = all_data.mean()
global_std = all_data.std()
X_normalized = (X_train - global_mean) / global_std
```

**Pros**:
- Consistent scaling across windows
- Better for generalization

**Cons**:
- Need access to all data
- May not work well if data distribution shifts

### Option 5: Layer Normalization (In-Model)
```python
# Add LayerNorm inside ConvLSTM
self.layer_norm = nn.LayerNorm([feature_channels, height, width])
```

**Pros**:
- Normalizes activations during forward pass
- Helps with gradient flow
- No need to denormalize outputs

**Cons**:
- Changes model architecture
- Not what PI-ConvTF does

## Gradient Flow Summary

### Current Issues Affecting Learning:

1. **Vanishing Gradients Through Time**:
   - 21 timesteps × multiple sigmoid gates = exponential gradient decay
   - Early timesteps get very small gradients
   - **Impact**: Model struggles to learn long-term dependencies

2. **Input Scale Mismatch**:
   - Raw IV [0.06, 0.54] with mean ~0.19
   - Default weight initialization expects zero-mean inputs
   - **Impact**: Gates may saturate, reducing gradient flow

3. **Spatial Dimension Reduction**:
   - (20,20) → (18,18) → (20,20) with padding workaround
   - **Impact**: Edge information lost, suboptimal architecture

4. **Initialization**:
   - Cell/hidden states start at zero
   - First update: `Ct = it * gt` (only candidate matters)
   - **Impact**: Slow start, needs many epochs to build up state

## Normalization Implementation

### Recommended: Standardization (Per-Window)

```python
# In ConvLSTMModel.fit() or before training:
def normalize_data(X, y, fit_stats=None):
    """Normalize data using standardization"""
    if fit_stats is None:
        # Compute statistics from training data
        X_mean = X.mean()
        X_std = X.std()
        y_mean = y.mean()
        y_std = y.std()
        fit_stats = {'X_mean': X_mean, 'X_std': X_std, 
                     'y_mean': y_mean, 'y_std': y_std}
    else:
        # Use provided statistics (for val/test)
        X_mean = fit_stats['X_mean']
        X_std = fit_stats['X_std']
        y_mean = fit_stats['y_mean']
        y_std = fit_stats['y_std']
    
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    
    return X_norm, y_norm, fit_stats

def denormalize_predictions(pred_norm, fit_stats):
    """Denormalize predictions back to original scale"""
    return pred_norm * fit_stats['y_std'] + fit_stats['y_mean']
```

### Why Standardization Helps:

1. **Zero-Mean Inputs**: Matches weight initialization assumptions
2. **Unit Variance**: Prevents gate saturation
3. **Better Gradients**: Gates stay in active region (sigmoid derivative > 0)
4. **Faster Learning**: Model can make fine adjustments
5. **Better Memorization**: With normalized inputs, model can learn exact values

### Trade-offs:

- **Pros**: Better gradient flow, faster learning, easier memorization
- **Cons**: Need to store/apply normalization stats, not matching PI-ConvTF exactly
- **Note**: PI-ConvTF doesn't normalize volatility, but they might have different data characteristics

## Recommendation

**For matching PI-ConvTF exactly**: Keep no normalization (current approach)
- But expect slower learning and difficulty memorizing small datasets

**For better learning/memorization**: 
1. **Add Standardization** (Option 2)
2. Store normalization stats per window
3. Normalize X_train, y_train, X_val, y_val using train stats
4. Denormalize predictions before computing RMSE/metrics
5. This should help the model memorize 7 samples much better

**Also Consider**:
- Lower learning rate (0.001 instead of 0.01) - matches PI-ConvTF
- Use padding=1 in ConvLSTM layers to maintain (20,20) throughout
- These changes together should significantly improve memorization
