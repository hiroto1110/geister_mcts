from __future__ import annotations

from dataclasses import dataclass, field
import time
import itertools

from tqdm import tqdm

import jax
from jax import random, numpy as jnp
import optax

from network.checkpoints import Checkpoint, CheckpointManager, NetworkConfig
from train_state import TrainStateBase
from batch import load, get_tokens


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
    rng_key: jax.Array = field(default_factory=lambda: random.PRNGKey(0))

    def num_minibatch(self, num_batches: int) -> int:
        return self.num_batches * (num_batches - self.replay_buffer_size) // self.update_period

    def next_minibatch(self, step: int) -> jnp.ndarray:
        sub_step = step // self.num_batches

        indices = random.choice(random.PRNGKey(step), jnp.arange(self.replay_buffer_size), shape=(self.batch_size,))
        indices += sub_step * self.update_period

        return indices


def train_epoch(
    state: TrainStateBase,
    batches: list[jnp.ndarray],
    minibatch_producer: MinibatchProducer,
    eval: bool
):
    losses_history = []

    num_steps = minibatch_producer.num_minibatch(len(batches))

    with tqdm(range(num_steps)) as pbar:
        for i in pbar:
            indices = minibatch_producer.next_minibatch(i)

            state, loss, losses = state.train_step(batches[indices], eval)
            losses_history.append(jax.device_get(losses))

            pbar.set_postfix({"loss": f"{float(loss):.3f}"})

    return state, jnp.mean(jnp.array(losses_history), axis=0)


def fit(
    state: TrainStateBase,
    model_config: NetworkConfig,
    checkpoint_manager: CheckpointManager,
    train_batches: jnp.ndarray,
    test_batches: jnp.ndarray,
    minibatch_producer: MinibatchProducer,
    epochs: int,
    log_wandb: bool
):
    import wandb

    for epoch in range(state.epoch + 1, state.epoch + 1 + epochs):
        start = time.perf_counter()

        state, losses_train = train_epoch(
            state, train_batches, minibatch_producer, eval=False
        )
        _, losses_test = train_epoch(
            state, test_batches, minibatch_producer, eval=True
        )

        elapsed_time = time.perf_counter() - start

        msg = f'Epoch: {epoch}, Loss: ({losses_train.sum():.3f}, {losses_test.sum():.3f})'

        for i, name in enumerate(state.get_head_names()):
            msg += f', {name}: ({losses_train[i]:.3f}, {losses_test[i]:.3f})'

        print(msg)

        log_dict = {
            'epoch': epoch,
            'elapsed time': elapsed_time,
            'train/loss': losses_train.sum(),
            'train/loss policy': losses_train[0],
            'train/loss value': losses_train[1],
            'train/loss color': losses_train[2],
            'test/loss': losses_test.sum(),
            'test/loss policy': losses_test[0],
            'test/loss value': losses_test[1],
            'test/loss color': losses_test[2],
        }
        if log_wandb:
            wandb.log(log_dict)

        state = state.replace(epoch=state.epoch + 1)
        checkpoint_manager.save(Checkpoint(int(state.epoch), model_config, state.params))

    return state


def main_train(batch: jnp.ndarray, log_wandb=False):
    import wandb
    from transformer import TrainStateTransformer

    n_train = int(batch.shape[0] * 0.8)
    train_batch = batch[:n_train]
    test_batch = batch[n_train:]

    minibatch_producer = MinibatchProducerRL(
        replay_buffer_size=2048,
        update_period=64,
        batch_size=16,
        num_batches=32
    )
    minibatch_producer = MinibatchProducerSimple(batch_size=16)

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

        model_config = TransformerConfig(
            num_heads=h,
            embed_dim=d,
            num_hidden_layers=n,
        )
        model = model_config.create_model()

        init_x = get_tokens(train_batch[0, :1])

        variables = model.init(random.PRNGKey(0), init_x)
        state = TrainStateTransformer.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optax.adam(learning_rate=0.0005),
            dropout_rng=random.PRNGKey(0),
            epoch=0,
        )

        ckpt_dir = f'./data/checkpoints/{h}_{d}_{n}'

        checkpoint_manager = CheckpointManager(ckpt_dir)
        checkpoint_manager.save(Checkpoint(state.epoch, model_config, state.params))

        state = fit(
            state, model_config, checkpoint_manager,
            train_batches=train_batch,
            test_batches=test_batch,
            minibatch_producer=minibatch_producer,
            epochs=4,
            log_wandb=log_wandb
        )

        if log_wandb:
            run.finish()


def main():
    batch = load("./data/replay_buffer/run-7.npy")
    batch = batch.reshape(-1, batch.shape[-1])
    print(batch.shape)

    # indices = jnp.arange(batch.shape[0])
    # indices = random.shuffle(random.PRNGKey(0), indices)

    # batch = batch[indices]

    main_train(batch)


if __name__ == "__main__":
    main()
