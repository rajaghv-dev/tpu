"""ResNet-50 in Flax."""

from typing import Sequence
import jax.numpy as jnp
from flax import linen as nn


class ResNetBlock(nn.Module):
    filters: int
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        y = nn.Conv(self.filters, (3, 3), self.strides, padding="SAME", use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(self.filters, (3, 3), padding="SAME", use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)

        if residual.shape != y.shape:
            residual = nn.Conv(self.filters, (1, 1), self.strides, use_bias=False)(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        return nn.relu(y + residual)


class BottleneckBlock(nn.Module):
    filters: int
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        y = nn.Conv(self.filters, (1, 1), use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(self.filters, (3, 3), self.strides, padding="SAME", use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(self.filters * 4, (1, 1), use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)

        if residual.shape != y.shape:
            residual = nn.Conv(self.filters * 4, (1, 1), self.strides, use_bias=False)(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        return nn.relu(y + residual)


class ResNet50(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        for filters, n_blocks, stride in [
            (64,  3, (1, 1)),
            (128, 4, (2, 2)),
            (256, 6, (2, 2)),
            (512, 3, (2, 2)),
        ]:
            for i in range(n_blocks):
                x = BottleneckBlock(filters, strides=stride if i == 0 else (1, 1))(x, train)

        x = jnp.mean(x, axis=(1, 2))  # global average pool
        x = nn.Dense(self.num_classes)(x)
        return x
