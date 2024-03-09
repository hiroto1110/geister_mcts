from typing import Any
from functools import partial
from dataclasses import dataclass
import time
import itertools

from tqdm import tqdm

import jax
from jax import random, numpy as jnp
import optax
from flax.training import train_state

from network.checkpoints import Checkpoint, CheckpointManager
from network.transformer import Transformer, TransformerConfig
from batch import load, astuple, get_tokens


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

    def scan_f(carry: tuple[jnp.ndarray, jnp.ndarray], i):
        memory_prev, loss_prev = carry

        p, v, c, memory_next = apply(x[:, i], memory_prev)
        loss_i, losses_i = calc_loss(x[:, i], p, v, c, p_true[:, i], v_true[:, i], c_true[:, i])

        memory_next = memory_next.reshape(memory_prev.shape)
        loss_next = loss_prev + loss_i

        return (memory_next, loss_next), losses_i

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


class MinibatchProducer:
    def num_minibatch(self, num_batches: int) -> int:
        pass

    def next_minibatch(self, step: int) -> jnp.ndarray:
        pass


@dataclass
class MinibatchProducerSimple(MinibatchProducer):
    batch_size: int

    def num_minibatch(self, num_batches: int) -> int:
        return num_batches // self.batch_size

    def next_minibatch(self, step: int) -> jnp.ndarray:
        return jnp.arange(self.batch_size) + (self.batch_size * step)


@dataclass
class MinibatchProducerRL(MinibatchProducer):
    replay_buffer_size: int
    update_period: int
    batch_size: int
    num_batches: int
    rng_key: jax.Array = random.PRNGKey(0)

    def num_minibatch(self, num_batches: int) -> int:
        return self.num_batches * (num_batches - self.replay_buffer_size) // self.update_period

    def next_minibatch(self, step: int) -> jnp.ndarray:
        sub_step = step // self.num_batches

        indices = random.choice(random.PRNGKey(step), jnp.arange(self.replay_buffer_size), shape=(self.batch_size,))
        indices += sub_step * self.update_period

        return indices


def train_epoch(
    state: TrainState,
    batches: list[jnp.ndarray],
    minibatch_producer: MinibatchProducer,
    num_division_of_segment: int,
    eval: bool
):
    loss_history, info_history = [], []

    num_steps = minibatch_producer.num_minibatch(len(batches))

    with tqdm(range(num_steps)) as pbar:
        for i in pbar:
            indices = minibatch_producer.next_minibatch(i)

            state, loss, info = train_step(state, *astuple(batches[indices]), num_division_of_segment, eval)
            loss_history.append(jax.device_get(loss))
            info_history.append(jax.device_get(info))

            pbar.set_postfix({"loss": f"{float(loss):.3f}"})

    return state, jnp.mean(jnp.array(loss_history)), jnp.mean(jnp.array(info_history), axis=0)


def fit(
    state: TrainState,
    model_config: TransformerConfig,
    checkpoint_manager: CheckpointManager,
    train_batches: jnp.ndarray,
    test_batches: jnp.ndarray,
    minibatch_producer: MinibatchProducer,
    epochs: int,
    num_division_of_segment: int,
    log_wandb: bool
):
    import wandb

    for epoch in range(state.epoch + 1, state.epoch + 1 + epochs):
        start = time.perf_counter()

        state, loss_train, info_train = train_epoch(
            state, train_batches, minibatch_producer, num_division_of_segment, eval=False
        )
        _, loss_test, info_test = train_epoch(
            state, test_batches, minibatch_producer, num_division_of_segment, eval=True
        )

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
        checkpoint_manager.save(Checkpoint(int(state.epoch), model_config, state.params))

    return state


def main_train(batch: jnp.ndarray, log_wandb=False):
    import wandb

    n_train = int(batch.shape[0] * 0.8)
    train_batch = batch[:n_train]
    test_batch = batch[n_train:]

    minibatch_producer = MinibatchProducerRL(
        replay_buffer_size=2048,
        update_period=64,
        batch_size=16,
        num_batches=32
    )
    # minibatch_producer = MinibatchProducerSimple(batch_size=16)

    heads = 8,
    dims = 512,
    num_layers = 8,
    memory_length = 16,

    for h, d, n, m in itertools.product(heads, dims, num_layers, memory_length):
        if log_wandb:
            name = f'h={h}, d={d}, n={n}'
            run_config = {
                'num heads': h,
                'embed dim': d,
                'num layers': n,
                'memory length': m,
            }
            run = wandb.init(project='network benchmark', config=run_config, name=name)

        model_config = TransformerConfig(
            num_heads=h,
            embed_dim=d,
            num_hidden_layers=n,
            length_memory_block=m,
        )
        model = model_config.create_model()

        init_x = get_tokens(train_batch[0, :1])

        variables = model.init(random.PRNGKey(0), init_x)
        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optax.adam(learning_rate=0.0005),
            dropout_rng=random.PRNGKey(0),
            epoch=0,
            init_memory=model.create_zero_memory()
        )

        ckpt_dir = f'./data/checkpoints/rmt_{h}_{d}_{n}_{m}'

        checkpoint_manager = CheckpointManager(ckpt_dir)
        checkpoint_manager.save(Checkpoint(state.epoch, model_config, state.params))

        state = fit(
            state, model_config, checkpoint_manager,
            train_batches=train_batch,
            test_batches=test_batch,
            minibatch_producer=minibatch_producer,
            epochs=4, num_division_of_segment=8,
            log_wandb=log_wandb
        )

        if log_wandb:
            run.finish()


def main():
    batch = load("./data/replay_buffer/run-7-new.npy")
    print(batch.shape)

    # indices = jnp.arange(batch.shape[0])
    # indices = random.shuffle(random.PRNGKey(0), indices)

    # batch = batch[indices]

    main_train(batch)


if __name__ == "__main__":
    main()
