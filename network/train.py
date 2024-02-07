from typing import Any
from functools import partial
import time
import itertools

from tqdm import tqdm

import jax
from jax import random, numpy as jnp
import optax
from flax.training import train_state

import wandb

from network.checkpoints import Checkpoint, CheckpointManager
from network.transformer import Transformer, TransformerConfig
from batch import load


class TrainState(train_state.TrainState):
    epoch: int
    dropout_rng: Any
    init_memory: jnp.ndarray

    @classmethod
    def create_init_memory(cls, model: Transformer) -> jnp.ndarray:
        return jnp.zeros((model.config.length_memory_block, model.config.embed_dim))


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


@partial(jax.jit, static_argnames=['eval', 'segment_offset', 'segment_length'])
def loss_fn(
    params,
    state: TrainState,
    x: jnp.ndarray,
    p_true: jnp.ndarray,
    v_true: jnp.ndarray,
    c_true: jnp.ndarray,
    init_memory: jnp.ndarray,
    segment_offset: int,
    segment_length: int,
    dropout_rng,
    eval: bool
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    # x: [Batch, Segments, Seq Len, 5]
    batch_size = x.shape[0]

    @jax.checkpoint
    def apply(_x, read_memory):
        return state.apply_fn(
            {'params': params},
            _x, read_memory=read_memory, eval=eval,
            rngs={'dropout': dropout_rng}
        )

    def scan_f(carry, i):
        memory_c, loss_c = carry

        p, v, c, memory_c = apply(x[:, i], memory_c)
        loss_i, losses_i = calc_loss(x[:, i], p, v, c, p_true[:, i], v_true[:, i], c_true[:, i])

        loss_c += loss_i

        return (memory_c, loss_c), losses_i

    init_memory = jnp.zeros((batch_size, *state.init_memory.shape))
    indices = segment_offset + jnp.arange(segment_length)

    (memory, loss), losses = jax.lax.scan(scan_f, init=(init_memory, 0), xs=indices)

    loss /= segment_length

    return loss, (losses, memory)


@partial(jax.jit, static_argnames=['eval', 'num_division_of_segment'])
def train_step(
    state: TrainState,
    x: jnp.ndarray, p_true: jnp.ndarray, v_true: jnp.ndarray, c_true: jnp.ndarray,
    num_division_of_segment: int, eval: bool
) -> tuple[TrainState, jnp.ndarray, jnp.ndarray]:

    batch_size = x.shape[0]
    segment_length = x.shape[1]
    sub_segment_length = segment_length // num_division_of_segment

    def scan_f(carry, i):
        current_state: TrainState = carry[0]
        current_memory = carry[1]
        current_loss = carry[2]

        offset = i * sub_segment_length

        if not eval:
            (loss_i, (losses, next_memory)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                current_state.params, current_state,
                x, p_true, v_true, c_true, current_memory, offset, sub_segment_length,
                current_state.dropout_rng, eval=eval
            )
            next_state = current_state.apply_gradients(grads=grads, dropout_rng=random.PRNGKey(current_state.epoch))
        else:
            loss_i, (losses, next_memory) = loss_fn(
                current_state.params, current_state,
                x, p_true, v_true, c_true, current_memory, offset, sub_segment_length,
                current_state.dropout_rng, eval=eval
            )
            next_state = current_state

        next_loss = current_loss + loss_i

        return (next_state, next_memory, next_loss), losses

    init_memory = jnp.zeros((batch_size, *state.init_memory.shape))
    indices = jnp.arange(num_division_of_segment)

    (state, _, loss), losses = jax.lax.scan(scan_f, init=(state, init_memory, 0), xs=indices)

    losses = losses.reshape((segment_length, *losses.shape[2:]))

    return state, loss, losses


def train_epoch(state: TrainState, batches: list[jnp.ndarray], num_division_of_segment: int, eval: bool):
    loss_history, info_history = [], []

    for batch in tqdm(batches):
        state, loss, info = train_step(state, *batch.astuple(), num_division_of_segment, eval)
        loss_history.append(jax.device_get(loss))
        info_history.append(jax.device_get(info))

    return state, jnp.mean(jnp.array(loss_history)), jnp.mean(jnp.array(info_history), axis=0)


def fit(
    state: TrainState,
    model_config: TransformerConfig,
    checkpoint_manager: CheckpointManager,
    train_batch: jnp.ndarray,
    test_batch: jnp.ndarray,
    epochs: int,
    batch_size: int,
    num_division_of_segment: int,
    log_wandb: bool
):
    train_batches = train_batch.divide(batch_size)
    test_batches = test_batch.divide(batch_size)

    for epoch in range(state.epoch + 1, state.epoch + 1 + epochs):
        start = time.perf_counter()

        state, loss_train, info_train = train_epoch(state, train_batches, num_division_of_segment, eval=False)
        _, loss_test, info_test = train_epoch(state, test_batches, num_division_of_segment, eval=True)

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
        checkpoint_manager.save(Checkpoint(state.epoch, model_config, state.params))

    return state


def main_train(batch: jnp.ndarray, log_wandb=False):
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

        model_config = TransformerConfig(num_heads=h, embed_dim=d, num_hidden_layers=n, length_memory_block=8)
        model = model_config.create_model()

        variables = model.init(random.PRNGKey(0), jnp.zeros((1, 200, 5), dtype=jnp.uint8))
        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optax.adam(learning_rate=0.0005),
            dropout_rng=random.PRNGKey(0),
            epoch=0,
            init_memory=model.create_zero_memory())

        ckpt_dir = f'./data/checkpoints/rmt_{h}_{d}_{n}'

        checkpoint_manager = CheckpointManager(ckpt_dir)
        checkpoint_manager.save(Checkpoint(state.epoch, model, state.params))

        state = fit(
            state, model, checkpoint_manager,
            train_batch=train_batch,
            test_batch=test_batch,
            epochs=16, batch_size=4, num_division_of_segment=4,
            log_wandb=log_wandb
        )

        if log_wandb:
            run.finish()


def main():
    batch = load('./data/replay_buffer/189.npz', shuffle=True)

    main_train(batch)


if __name__ == "__main__":
    main()
