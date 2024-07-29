from __future__ import annotations

import dataclasses
from functools import partial

import serde
import jax
import optax
from jax import numpy as jnp
from flax import linen as nn
from train_state import TrainStateBase


def pos_to_board(
    pos1: jnp.ndarray,
    pos2: jnp.ndarray,
    color1: jnp.ndarray,
    color2: jnp.ndarray
) -> jnp.ndarray:
    batch_shape = pos1.shape[:-1]

    pos1 = pos1.reshape(-1, 8)
    pos2 = pos2.reshape(-1, 8)
    color1 = color1.reshape(-1, 8)
    color2 = color2.reshape(-1, 8)

    def scan_f(x_i) -> jnp.ndarray:
        p1, p2, c1, c2 = [x_i[i*8: (i+1)*8] for i in range(4)]

        board = jnp.zeros((37, 4), dtype=jnp.uint8)
        board = board.at[p1, 0].set(c1)
        board = board.at[p1, 1].set(255 - c1)
        board = board.at[p2, 2].set(c2)
        board = board.at[p2, 3].set(255 - c2)

        return None, board

    xs = jnp.concatenate([pos1, pos2, color1, color2], axis=-1, dtype=jnp.uint8)

    _, board = jnp.apply_along_axis(scan_f, axis=-1, arr=xs)

    board = board[..., :36, :].reshape((*batch_shape, 6, 6, 4))

    return board


@serde.serde
@dataclasses.dataclass(frozen=True)
class CNNConfig:
    num_filters: list[int]

    def create_model(self):
        return CNN(self)


resnet_kernel_init = nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal')

class ResNetBlock(nn.Module):
    act_fn : callable  # Activation function
    c_out : int   # Output feature size
    subsample : bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, eval=False):
        # Network representing F
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            strides=(1, 1) if not self.subsample else (2, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False
        )(x)

        z = nn.LayerNorm()(z)
        z = self.act_fn(z)
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            kernel_init=resnet_kernel_init,
            use_bias=False
        )(z)

        z = nn.LayerNorm()(z)

        if self.subsample:
            x = nn.Conv(self.c_out, kernel_size=(1, 1), strides=(2, 2), kernel_init=resnet_kernel_init)(x)

        x_out = self.act_fn(z + x)
        return x_out


class ResNet(nn.Module):
    act_fn : callable = jax.nn.relu
    num_blocks : tuple = (3, 3)
    c_hidden : tuple = (32, 64)

    @nn.compact
    def __call__(self, x: jnp.ndarray, concat: jnp.ndarray, eval=False):
        x = x.astype(dtype=jnp.float16) / 255.0

        a = jnp.zeros((*x.shape[:-1], x.shape[-1] + concat.shape[-1]))

        a = a.at[..., :x.shape[-1]].set(x)
        a = a.at[..., x.shape[-1]:].set(concat[..., jnp.newaxis, jnp.newaxis, :])

        x = a.reshape((-1, *a.shape[-3:]))

        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(self.c_hidden[0], kernel_size=(3, 3), kernel_init=resnet_kernel_init, use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = ResNetBlock(
                    c_out=self.c_hidden[block_idx],
                    act_fn=self.act_fn,
                    subsample=subsample
                )(x, eval=eval)

        # Mapping to classification output
        x = x.mean(axis=(1, 2))

        v = nn.Dense(features=7)(x)
        p = nn.Dense(features=144)(x)

        v = v.reshape((*a.shape[:-3], 7))
        p = p.reshape((*a.shape[:-3], 144))

        return p, v


class CNN(nn.Module):
    config: CNNConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        concat: jnp.ndarray,
        eval=True
    ):
        batch_shape = x.shape[:-3]

        x = x.reshape((-1, *x.shape[-3:]))
        x = x.astype(dtype=jnp.float16) / 255.0

        concat = concat.reshape((*x.shape[:-3], concat.shape[-1]))

        for n in self.config.num_filters:
            x = nn.Conv(features=n, kernel_size=(3, 3))(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((*x.shape[:-3], -1))

        x = jnp.concatenate([x, concat], axis=-1)

        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        v = nn.Dense(features=7)(x)
        p = nn.Dense(features=144)(x)

        v = v.reshape((*batch_shape, 7))
        p = p.reshape((*batch_shape, 144))

        return p, v


@jax.jit
def calc_loss(
    p_pred: jnp.ndarray, v_pred: jnp.ndarray,
    p_true: jnp.ndarray, v_true: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # [Batch, SeqLen, 32]
    p_true = p_true.reshape(-1)
    v_true = v_true.reshape(-1)

    loss_p = optax.softmax_cross_entropy_with_integer_labels(p_pred, p_true)
    loss_v = optax.softmax_cross_entropy_with_integer_labels(v_pred, v_true)

    loss_p = jnp.average(loss_p)
    loss_v = jnp.average(loss_v)

    loss = loss_p + loss_v
    losses = jnp.array([loss_p, loss_v])

    return loss, losses


@partial(jax.jit, static_argnames=['eval'])
def loss_fn(
    params,
    state: TrainStateCNN,
    x: jnp.ndarray,
    n_cap: jnp.ndarray,
    p_true: jnp.ndarray,
    v_true: jnp.ndarray,
    eval: bool
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    # x: [Batch, 6, 6, layers]
    p_pred, v_pred = state.apply_fn({'params': params}, x, n_cap, eval=eval)
    loss, losses = calc_loss(p_pred, v_pred, p_true, v_true)
    return loss, losses


class TrainStateCNN(TrainStateBase):
    @partial(jax.jit, static_argnames=['eval'])
    def train_step(
        self, x: jnp.ndarray, eval: bool
    ) -> tuple[TrainStateCNN, jnp.ndarray, jnp.ndarray]:
        
        n_cap = x[..., 4*36: 4*36 + 8]
        p_true = x[..., 4*36 + 8].astype(jnp.int16)
        v_true = x[..., 4*36 + 9].astype(jnp.int16)
        x = x[..., :4*36].reshape(-1, 6, 6, 4)

        if not eval:
            (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                self.params, self, x, n_cap, p_true, v_true, eval=eval
            )
            state = self.apply_gradients(grads=grads)
        else:
            loss, losses = loss_fn(self.params, self, x, n_cap, p_true, v_true, eval=eval)
            state = self

        return state, loss, losses

    def get_head_names(self) -> list[str]:
        return ['P', 'V']
