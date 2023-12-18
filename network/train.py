from typing import Any
from functools import partial
import time
import dataclasses
import itertools

from tqdm import tqdm

import numpy as np
import jax
from jax import random, numpy as jnp
from flax.training import train_state
import orbax.checkpoint
from flax.training import orbax_utils
import optax

import wandb

from buffer import load_batch
import transformer


class TrainState(train_state.TrainState):
    dropout_rng: Any


@dataclasses.dataclass
class Checkpoint:
    state: TrainState
    model: transformer.TransformerDecoder
    is_league_member: bool = False

    def asdict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, ckpt_dict: dict, is_caching_model: bool = False) -> 'Checkpoint':
        state = TrainState(**ckpt_dict['state'])

        if is_caching_model:
            model = transformer.TransformerDecoder(**ckpt_dict['model'])
        else:
            model = transformer.TransformerDecoderWithCache(**ckpt_dict['model'])

        is_league_member = ckpt_dict['is_league_member']

        return Checkpoint(state, model, is_league_member)

    def save(self, checkpoint_manager: orbax.checkpoint.CheckpointManager):
        ckpt = self.asdict()
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(self.state.step, ckpt, save_kwargs={'save_args': save_args})

    @classmethod
    def load(
        cls,
        checkpoint_manager: orbax.checkpoint.CheckpointManager,
        step: int,
        is_caching_model: bool = False
    ) -> 'Checkpoint':
        ckpt = checkpoint_manager.restore(step)
        return Checkpoint.from_dict(ckpt, is_caching_model=is_caching_model)


@partial(jax.jit, static_argnames=['eval'])
def loss_fn(params,
            state: TrainState,
            x: jnp.ndarray,
            y_pi: jnp.ndarray,
            y_v: jnp.ndarray,
            y_color: jnp.ndarray,
            dropout_rng,
            eval: bool):
    pi, v, color = state.apply_fn({'params': params}, x, eval=eval,
                                  rngs={'dropout': dropout_rng})

    mask = jnp.any(x != 0, axis=2)

    # [Batch, SeqLen, 32]
    y_pi = y_pi.reshape(-1, x.shape[1])
    y_v = y_v.reshape(-1, 1)
    y_color = y_color.reshape(-1, 1, 8)

    loss_pi = optax.softmax_cross_entropy_with_integer_labels(pi, y_pi)
    loss_v = optax.softmax_cross_entropy_with_integer_labels(v, y_v)
    loss_color = optax.sigmoid_binary_cross_entropy(color, y_color).mean(axis=2)

    loss_pi = jnp.average(loss_pi, weights=mask)
    loss_v = jnp.average(loss_v, weights=mask)
    loss_color = jnp.average(loss_color, weights=mask)

    loss = loss_pi + loss_v + loss_color

    info = jnp.array([loss_pi, loss_v, loss_color])

    return loss, info


@partial(jax.jit, static_argnames=['eval'])
def train_step(state: TrainState, x, y_pi, y_v, y_color, eval):
    if not eval:
        new_dropout_rng, dropout_rng = random.split(state.dropout_rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state, x, y_pi, y_v, y_color, dropout_rng, eval)

        new_state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)
    else:
        loss, info = loss_fn(
            state.params, state, x, y_pi, y_v, y_color, random.PRNGKey(0), eval)
        new_state = state

    return new_state, loss, info


def train_epoch(state, data_batched, eval):
    loss_history, info_history = [], []
    for x, mask, y_pi, y_v, y_color in tqdm(zip(*data_batched), total=len(data_batched[0])):
        state, loss, info = train_step(state, x, mask, y_pi, y_v, y_color, eval)
        loss_history.append(jax.device_get(loss))
        info_history.append(jax.device_get(info))
    return state, jnp.mean(jnp.array(loss_history)), jnp.mean(jnp.array(info_history), axis=0)


def create_batches(data, batch_size):
    n = len(data) // batch_size
    data_batched = data[:n*batch_size].reshape(n, batch_size, *data.shape[1:])
    return data_batched


def fit(state, model, checkpoint_manager, train_data, test_data, epochs, batch_size, log_wandb):
    train_data_batched = [create_batches(data, batch_size) for data in train_data]
    test_data_batched = [create_batches(data, batch_size) for data in test_data]

    for epoch in range(state.epoch + 1, state.epoch + 1 + epochs):
        start = time.perf_counter()

        state, loss_train, info_train = train_epoch(state, train_data_batched, eval=False)
        _, loss_test, info_test = train_epoch(state, test_data_batched, eval=True)

        elapsed_time = time.perf_counter() - start

        print(f'Epoch: {epoch}, ', end='')
        print(f'Loss: ({loss_train:.3f}, {loss_test:.3f}), ', end='')
        print(f'P: ({info_train[0]:.3f}, {info_test[0]:.3f}), ', end='')
        print(f'V: ({info_train[1]:.3f}, {info_test[1]:.3f}), ', end='')
        print(f'C: ({info_train[2]:.3f}, {info_test[2]:.3f})')

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
        Checkpoint(state, model).save(checkpoint_manager)

    return state


def main_train(data, log_wandb=False):
    train_n = int(len(data[0]) * 0.8)

    train_data = [d[:train_n] for d in data]
    test_data = [d[train_n:] for d in data]

    key, key1, key2 = random.split(random.PRNGKey(0), 3)

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

        model = transformer.TransformerDecoder(num_heads=h, embed_dim=d, num_hidden_layers=n, is_linear_attention=False)

        variables = model.init(key1, data[0][:1])
        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optax.adam(learning_rate=0.0005),
            dropout_rng=key2,
            epoch=0)

        ckpt_dir = f'./checkpoints/{h}_{d}_{n}_run-2'

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(create=True)
        checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer, options)

        Checkpoint(state, model).save(checkpoint_manager)

        state = fit(state, model, checkpoint_manager,
                    train_data=train_data,
                    test_data=test_data,
                    epochs=8, batch_size=64,
                    log_wandb=log_wandb)

        if log_wandb:
            run.finish()


def main():
    batch = load_batch(['replay_buffer/run-2-e.npz'], shuffle=False)
    print(np.mean(np.sum(batch.mask, axis=1)))
    data = batch.astuple()

    print(data[0].shape)
    # main_train(data)


if __name__ == "__main__":
    main()
