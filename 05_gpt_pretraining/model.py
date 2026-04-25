"""Minimal GPT in Flax."""

from dataclasses import field
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct


class CausalSelfAttention(nn.Module):
    n_heads: int
    d_model: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        B, T, C = x.shape
        head_dim = C // self.n_heads

        qkv = nn.Dense(3 * C, use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(t):
            return t.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        q, k, v = map(split_heads, (q, k, v))

        scale = head_dim ** -0.5
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # Causal mask
        mask = jnp.tril(jnp.ones((T, T)))
        attn = jnp.where(mask == 0, jnp.finfo(attn.dtype).min, attn)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.dropout_rate, deterministic=not train)(attn)

        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return nn.Dense(C)(out)


class MLP(nn.Module):
    d_model: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = nn.Dense(4 * self.d_model)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model)(x)
        return nn.Dropout(self.dropout_rate, deterministic=not train)(x)


class TransformerBlock(nn.Module):
    n_heads: int
    d_model: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = x + CausalSelfAttention(self.n_heads, self.d_model, self.dropout_rate)(
            nn.LayerNorm()(x), train
        )
        x = x + MLP(self.d_model, self.dropout_rate)(nn.LayerNorm()(x), train)
        return x


class GPT(nn.Module):
    vocab_size: int
    max_seq_len: int
    n_layers: int
    n_heads: int
    d_model: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        B, T = tokens.shape
        pos = jnp.arange(T)[None]

        tok_emb = nn.Embed(self.vocab_size, self.d_model)(tokens)
        pos_emb = nn.Embed(self.max_seq_len, self.d_model)(pos)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(tok_emb + pos_emb)

        for _ in range(self.n_layers):
            x = TransformerBlock(self.n_heads, self.d_model, self.dropout_rate)(x, train)

        x = nn.LayerNorm()(x)
        return nn.Dense(self.vocab_size, use_bias=False)(x)  # logits


# GPT-2 small
GPT_SMALL_CONFIG = dict(
    vocab_size=50257,
    max_seq_len=1024,
    n_layers=12,
    n_heads=12,
    d_model=768,
)
