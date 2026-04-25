"""
05 – GPT Pre-training
Trains a GPT-2-small-scale model from scratch on a text corpus.
Run: python train.py --corpus=data/corpus.txt
"""

import argparse
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from model import GPT, GPT_SMALL_CONFIG


SEQ_LEN    = 1024
BATCH_SIZE = 8    # per device
LR         = 3e-4
GRAD_CLIP  = 1.0
TOTAL_ITERS = 600_000


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_token_array(corpus_path: str, vocab_size: int = 50257) -> np.ndarray:
    """Tokenize corpus with tiktoken if available, else fake data."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        text = Path(corpus_path).read_text()
        return np.array(enc.encode(text), dtype=np.uint16)
    except (ImportError, FileNotFoundError):
        print("Warning: using random tokens (no corpus / tiktoken not installed)")
        return np.random.randint(0, vocab_size, size=(1_000_000,), dtype=np.uint16)


def random_batch(tokens: np.ndarray, seq_len: int, batch_size: int, rng: np.random.Generator):
    idx = rng.integers(0, len(tokens) - seq_len - 1, size=batch_size)
    x = np.stack([tokens[i:i + seq_len] for i in idx]).astype(np.int32)
    y = np.stack([tokens[i + 1:i + seq_len + 1] for i in idx]).astype(np.int32)
    return x, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
def train_step(state, x, y):
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params}, x, train=True,
            rngs={"dropout": jax.random.PRNGKey(state.step[0] if hasattr(state.step, '__len__') else state.step)},
        )
        B, T, V = logits.shape
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(B * T, V), y.reshape(B * T)
            )
        )
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss  = jax.lax.pmean(loss,  axis_name="batch")
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--iters", type=int, default=10_000)
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    n = jax.device_count()
    rng_np = np.random.default_rng(42)
    tokens = build_token_array(args.corpus or "", GPT_SMALL_CONFIG["vocab_size"])

    # Build model + optimizer
    model = GPT(**GPT_SMALL_CONFIG)
    dummy_x = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), dummy_x, train=False)["params"]

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LR,
        warmup_steps=2000,
        decay_steps=args.iters,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adamw(schedule, weight_decay=0.1),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state = jax.device_put_replicated(state, jax.devices())

    local_batch = BATCH_SIZE * n

    for step in range(1, args.iters + 1):
        x, y = random_batch(tokens, SEQ_LEN, local_batch, rng_np)
        x = x.reshape(n, BATCH_SIZE, SEQ_LEN)
        y = y.reshape(n, BATCH_SIZE, SEQ_LEN)
        state, loss = train_step(state, x, y)

        if step % args.log_every == 0:
            print(f"step {step:6d}/{args.iters}  loss={float(loss[0]):.4f}")


if __name__ == "__main__":
    main()
