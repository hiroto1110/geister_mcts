from __future__ import annotations

import dataclasses
from functools import partial

import serde
import numpy as np
import jax
import optax
from jax import numpy as jnp
from flax import linen as nn
from train_state import TrainStateBase


@serde.serde
@dataclasses.dataclass(frozen=True)
class CNNConfig:
    num_filters: list[int]

    def create_model(self):
        return CNN(self)


class CNN(nn.Module):
    config: CNNConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        eval=True
    ):
        x = nn.Conv(features=self.config.num_filters[0], kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=self.config.num_filters[1], kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        logits_pi = nn.Dense(features=32)(x)
        logits_v = nn.Dense(features=7)(x)

        return logits_pi, logits_v


@jax.jit
def calc_loss(
    p_pred: jnp.ndarray, v_pred: jnp.ndarray,
    p_true: jnp.ndarray, v_true: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # [Batch, SeqLen, 32]
    p_true = p_true.reshape(-1, 32)
    v_true = v_true.reshape(-1, 1)

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
    p_true: jnp.ndarray,
    v_true: jnp.ndarray,
    eval: bool
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    # x: [Batch, 6, 6, layers]
    p_pred, v_pred = state.apply_fn({'params': params}, x, eval=eval)
    loss, losses = calc_loss(p_pred, v_pred, p_true, v_true)
    return loss, losses


class TrainStateCNN(TrainStateBase):
    @partial(jax.jit, static_argnames=['eval'])
    def train_step(
        self, x: jnp.ndarray, eval: bool
    ) -> tuple[TrainStateCNN, jnp.ndarray, jnp.ndarray]:
        
        p_true = x[:, 12 * 36]
        v_true = x[:, 12 * 36 + 1]
        x = x[:, :12 * 36].reshape(-1, 6, 6, 12)

        if not eval:
            (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                self.params, state, x, p_true, v_true, eval=eval
            )
            state = self.apply_gradients(grads=grads)
        else:
            loss, losses = loss_fn(self.params, state, x, p_true, v_true, eval=eval)
            state = self

        return state, loss, losses

    def get_head_names(self) -> list[str]:
        return ['P', 'V']


def create_pos_history_from_tokens(tokens: np.ndarray) -> np.ndarray:
    pos_history = np.zeros((tokens.shape[0], 16), dtype=np.uint8)

    if tokens[0, 3] < 3:
        pos = np.array([1, 2, 3, 4, 7, 8, 9, 10, 25, 26, 27, 28, 31, 32, 33, 34])
    else:
        pos = np.array([25, 26, 27, 28, 31, 32, 33, 34, 1, 2, 3, 4, 7, 8, 9, 10])

    for t, (c, id, x, y, n) in enumerate(tokens):
        if np.all(tokens[t] == 0):
            break

        if x < 6 and y < 6:
            pos[id] = x + 6 * y
        else:
            pos[id] = 36

        pos_history[t] = pos

    return pos_history
