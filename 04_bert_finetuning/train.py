"""
04 – BERT Fine-tuning on GLUE (SST-2)
Loads a pretrained BERT from HuggingFace and fine-tunes on SST-2 sentiment.
Run: python train.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from functools import partial
from flax.training import train_state
from transformers import (
    AutoTokenizer,
    FlaxAutoModelForSequenceClassification,
)


TASK        = "sst2"
MODEL_NAME  = "bert-base-uncased"
MAX_LEN     = 128
BATCH_SIZE  = 32
EPOCHS      = 3
LR          = 2e-5
NUM_LABELS  = 2


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def tokenize(examples, tokenizer):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )


def build_loaders(tokenizer):
    raw = load_dataset("glue", TASK)
    cols = ["input_ids", "attention_mask", "token_type_ids", "label"]

    train_ds = raw["train"].map(
        partial(tokenize, tokenizer=tokenizer), batched=True, remove_columns=["sentence", "idx"]
    ).with_format("numpy", columns=cols)

    val_ds = raw["validation"].map(
        partial(tokenize, tokenizer=tokenizer), batched=True, remove_columns=["sentence", "idx"]
    ).with_format("numpy", columns=cols)

    return train_ds, val_ds


def batch_iter(ds, batch_size: int, shuffle: bool = False):
    indices = np.arange(len(ds))
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(indices) - batch_size + 1, batch_size):
        idx = indices[start:start + batch_size]
        yield {k: ds[idx][k] for k in ["input_ids", "attention_mask", "token_type_ids", "label"]}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@partial(jax.pmap, axis_name="batch")
def train_step(state, batch):
    labels = batch.pop("label")

    def loss_fn(params):
        out = state.apply_fn(**batch, params=params, train=True, dropout_rng=jax.random.PRNGKey(0))
        logits = out.logits
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss  = jax.lax.pmean(loss,  axis_name="batch")
    new_state = state.apply_gradients(grads=grads)
    acc = jax.lax.pmean(jnp.mean(jnp.argmax(logits, -1) == labels), axis_name="batch")
    return new_state, {"loss": loss, "accuracy": acc}


@partial(jax.pmap, axis_name="batch")
def eval_step(state, batch):
    labels = batch.pop("label")
    out = state.apply_fn(**batch, params=state.params, train=False)
    logits = out.logits
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    acc  = jnp.mean(jnp.argmax(logits, -1) == labels)
    return jax.lax.pmean({"loss": loss, "accuracy": acc}, axis_name="batch")


def shard(batch, n):
    return jax.tree_util.tree_map(lambda x: x.reshape((n, -1) + x.shape[1:]), batch)


def main():
    n = jax.device_count()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = FlaxAutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    total_steps = (EPOCHS * 67349) // BATCH_SIZE  # SST-2 train size
    schedule = optax.linear_schedule(LR, 0.0, total_steps)
    tx = optax.adamw(schedule, weight_decay=0.01)
    state = train_state.TrainState.create(
        apply_fn=model.__call__, params=model.params, tx=tx
    )
    state = jax.device_put_replicated(state, jax.devices())

    train_ds, val_ds = build_loaders(tokenizer)

    for epoch in range(1, EPOCHS + 1):
        for batch in batch_iter(train_ds, BATCH_SIZE * n, shuffle=True):
            state, _ = train_step(state, shard(batch, n))

        val_metrics = []
        for batch in batch_iter(val_ds, BATCH_SIZE * n):
            m = eval_step(state, shard(batch, n))
            val_metrics.append({k: float(v[0]) for k, v in m.items()})

        avg = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}
        print(f"Epoch {epoch}/{EPOCHS}  "
              f"val_loss={avg['loss']:.4f}  val_acc={avg['accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
