## Models

This folder holds model implementations. The transformer model optionally
saves the best checkpoint per window/context/horizon under:

```
models/transformer/checkpoints/
```

To enable checkpoint saving, run:

```
python run_model.py --model transformer --save-ckpt
```

For faster GPU training, enable automatic mixed precision:

```
python run_model.py --model transformer --save-ckpt --amp
```

Delta transformer normalization:
- Inputs are consecutive deltas plus the last surface as an anchor token.
- Delta tokens and delta targets are normalized with delta-specific stats.
- The anchor token uses its own mean/std, so level information is preserved.
