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
