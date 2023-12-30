from typing import Any
from functools import partial
import time
import dataclasses
import itertools

from tqdm import tqdm

import jax
from jax import random, numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from flax.training import orbax_utils
import orbax.checkpoint
import optax

import wandb

from buffer import Batch
from network.transformer import Transformer, TransformerWithCache


class TrainState(train_state.TrainState):
    epoch: int
    dropout_rng: Any


@dataclasses.dataclass
class Checkpoint:
    step: int
    params: FrozenDict
    model: Transformer

    def asdict(self):
        return {
            'step': self.step,
            'params': self.params,
            'model': dataclasses.asdict(self.model),
        }

    @classmethod
    def from_dict(
        cls,
        ckpt_dict: dict,
        is_caching_model: bool = False
    ) -> 'Checkpoint':
        if not is_caching_model:
            model = Transformer(**ckpt_dict['model'])
        else:
            model = TransformerWithCache(**ckpt_dict['model'])

        step = ckpt_dict['step']
        params = ckpt_dict['params']

        return Checkpoint(step, params, model)

    @classmethod
    def load(
        cls,
        checkpoint_manager: orbax.checkpoint.CheckpointManager,
        step: int,
        is_caching_model: bool = False
    ) -> 'Checkpoint':
        ckpt = checkpoint_manager.restore(step)
        return Checkpoint.from_dict(ckpt, is_caching_model=is_caching_model)

    def save(self, checkpoint_manager: orbax.checkpoint.CheckpointManager):
        ckpt = self.asdict()
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(self.step, ckpt, save_kwargs={'save_args': save_args})


@jax.jit
def calc_loss(
    x: jnp.ndarray,
    p_pred: jnp.ndarray, v_pred: jnp.ndarray, c_pred: jnp.ndarray,
    p_true: jnp.ndarray, v_true: jnp.ndarray, c_true: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    mask = jnp.any(x != 0, axis=2)

    # [Batch, SeqLen, 32]
    p_true = p_true.reshape(-1, x.shape[1])
    v_true = v_true.reshape(-1, 1)
    c_true = c_true.reshape(-1, 1, 8)

    loss_p = optax.softmax_cross_entropy_with_integer_labels(p_pred, p_true)
    loss_v = optax.softmax_cross_entropy_with_integer_labels(v_pred, v_true)
    loss_c = optax.sigmoid_binary_cross_entropy(c_pred, c_true).mean(axis=2)

    loss_p = jnp.average(loss_p, weights=mask)
    loss_v = jnp.average(loss_v, weights=mask)
    loss_c = jnp.average(loss_c, weights=mask)

    loss = loss_p + loss_v + loss_c
    losses = jnp.array([loss_p, loss_v, loss_c])

    return loss, losses


@partial(jax.jit, static_argnames=['eval'])
def loss_fn(
    params,
    state: TrainState,
    x: jnp.ndarray,
    p_true: jnp.ndarray,
    v_true: jnp.ndarray,
    c_true: jnp.ndarray,
    dropout_rng,
    eval: bool
) -> tuple[jnp.ndarray, jnp.ndarray]:
    p, v, c = state.apply_fn(
        {'params': params},
        x, eval=eval,
        rngs={'dropout': dropout_rng}
    )

    return calc_loss(x, p, v, c, p_true, v_true, c_true)


@partial(jax.jit, static_argnames=['eval'])
def loss_fn_rmt(
    params,
    state: TrainState,
    x: jnp.ndarray,
    p_true: jnp.ndarray,
    v_true: jnp.ndarray,
    c_true: jnp.ndarray,
    dropout_rng,
    eval: bool
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # x: [Batch, Segments, Seq Len, 5]
    seg_len = x.shape[1]

    loss = jnp.zeros(seg_len, dtype=jnp.float32)
    losses = jnp.zeros((seg_len, 3))

    memory = None

    for i in range(seg_len):
        p, v, c, memory = state.apply_fn(
            {'params': params},
            x[:, i], read_memory=memory, eval=eval,
            rngs={'dropout': dropout_rng}
        )

        loss_i, losses_i = calc_loss(x[:, i], p, v, c, p_true[:, i], v_true[:, i], c_true[:, i])

        loss = loss.at[i].set(loss_i)
        losses = losses.at[i].set(losses_i)

    return loss.mean(), losses


@partial(jax.jit, static_argnames=['eval', 'is_rmt'])
def train_step(
    state: TrainState,
    x: jnp.ndarray, p_true: jnp.ndarray, v_true: jnp.ndarray, c_true: jnp.ndarray,
    eval: bool, is_rmt: bool = False
) -> tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    if is_rmt:
        fun = loss_fn_rmt
    else:
        fun = loss_fn

    if not eval:
        new_dropout_rng, dropout_rng = random.split(state.dropout_rng)
        (loss, losses), grads = jax.value_and_grad(fun, has_aux=True)(
            state.params, state, x, p_true, v_true, c_true, dropout_rng, eval
        )

        new_state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)
    else:
        loss, losses = fun(state.params, state, x, p_true, v_true, c_true, random.PRNGKey(0), eval)
        new_state = state

    return new_state, loss, losses


def train_epoch(state: TrainState, batches: list[Batch], eval: bool, is_rmt: bool):
    loss_history, info_history = [], []

    for batch in tqdm(batches):
        state, loss, info = train_step(state, *batch.astuple(), eval, is_rmt)
        loss_history.append(jax.device_get(loss))
        info_history.append(jax.device_get(info))

    return state, jnp.mean(jnp.array(loss_history)), jnp.mean(jnp.array(info_history), axis=0)


def fit(state: TrainState,
        model: Transformer,
        checkpoint_manager: orbax.checkpoint.CheckpointManager,
        train_batch: Batch,
        test_batch: Batch,
        epochs: int,
        batch_size: int,
        log_wandb: bool
        ):
    is_rmt = model.has_memory_block()

    train_batches = train_batch.divide(batch_size)
    test_batches = test_batch.divide(batch_size)

    for epoch in range(state.epoch + 1, state.epoch + 1 + epochs):
        start = time.perf_counter()

        state, loss_train, info_train = train_epoch(state, train_batches, eval=False, is_rmt=is_rmt)
        _, loss_test, info_test = train_epoch(state, test_batches, eval=True, is_rmt=is_rmt)

        elapsed_time = time.perf_counter() - start

        print(f'Epoch: {epoch}')
        print(f'Loss: ({loss_train:.3f}, {loss_test:.3f})')
        print(f'P: ({info_train[0, 0]:.3f}_{info_train[-1, 0]:.3f}, {info_test[0, 0]:.3f}_{info_test[-1, 0]:.3f})')
        print(f'V: ({info_train[0, 1]:.3f}_{info_train[-1, 1]:.3f}, {info_test[0, 1]:.3f}_{info_test[-1, 1]:.3f})')
        print(f'C: ({info_train[0, 2]:.3f}_{info_train[-1, 2]:.3f}, {info_test[0, 2]:.3f}_{info_test[-1, 2]:.3f})')

        log_dict = {
            'epoch': epoch,
            'elapsed time': elapsed_time,
            'train/loss': loss_train,
            'train/loss policy': info_train[0],
            'train/loss value': info_train[1],
            'train/loss color': info_train[2],
            'test/loss': loss_test,
            'test/loss policy': info_test[0],
            'test/loss value': info_test[1],
            'test/loss color': info_test[2],
        }
        if log_wandb:
            wandb.log(log_dict)

        state = state.replace(epoch=state.epoch + 1)
        Checkpoint(state.epoch, state.params, model).save(checkpoint_manager)

    return state


def main_train(batch: Batch, log_wandb=False):
    train_batch, test_batch = batch.split(0.8)

    heads = 4,
    dims = 256,
    num_layers = 4,

    for h, d, n in itertools.product(heads, dims, num_layers):
        if log_wandb:
            name = f'h={h}, d={d}, n={n}'
            run_config = {
                'num heads': h,
                'embed dim': d,
                'num layers': n,
            }
            run = wandb.init(project='network benchmark', config=run_config, name=name)

        model = Transformer(num_heads=h, embed_dim=d, num_hidden_layers=n, length_memory_block=8)

        variables = model.init(random.PRNGKey(0), jnp.zeros((1, 200, 5), dtype=jnp.uint8))
        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optax.adam(learning_rate=0.0005),
            dropout_rng=random.PRNGKey(0),
            epoch=0)

        ckpt_dir = f'./data/checkpoints/rmt_{h}_{d}_{n}'

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(create=True)
        checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer, options)

        Checkpoint(state.epoch, state.params, model).save(checkpoint_manager)

        state = fit(state, model, checkpoint_manager,
                    train_batch=train_batch,
                    test_batch=test_batch,
                    epochs=16, batch_size=64,
                    log_wandb=log_wandb)

        if log_wandb:
            run.finish()


def main():
    batch = Batch.from_npz('./data/replay_buffer/run-2.npz', shuffle=True)

    batch = batch.create_batch_from_indices(jnp.arange((len(batch) // 16) * 16))
    batch = batch.reshape((-1, 16))

    main_train(batch)


if __name__ == "__main__":
    main()
