# 05 – GPT Pre-training

Trains a GPT-2-scale decoder-only transformer from scratch.

## Concepts
- Causal self-attention with mask
- `optax.warmup_cosine_decay_schedule`
- Gradient clipping with `optax.clip_by_global_norm`
- `donate_argnums` to donate device buffers in-place

## Run
```bash
# Quick smoke test (random tokens)
python train.py --iters=500

# On real text corpus
python train.py --corpus=data/corpus.txt --iters=600000
```

## Files
- `model.py` — GPT Flax module (causal attention, MLP, positional embeddings)
- `train.py` — training loop
