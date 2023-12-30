import multiprocessing

from tqdm import tqdm
import click

import numpy as np
import jax
import optax
import orbax.checkpoint
import wandb

from network.train import Checkpoint, TrainState, train_step
from network.transformer import Transformer
import buffer

from config import RunConfig
import collector


@click.command()
@click.argument('config_path', type=str)
@click.argument('host', type=str)
@click.argument('port', type=int)
def main(
        config_path: str,
        host: str,
        port: int
):
    config = RunConfig.from_json_file(config_path)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    if not config.init_checkpoint_config.init_random:
        checkpoint_manager = orbax.checkpoint.CheckpointManager(config.init_checkpoint_config.dir_name, checkpointer)
        ckpt = Checkpoint.load(
            checkpoint_manager,
            step=config.init_checkpoint_config.step
        )

        model = ckpt.model
        params = ckpt.params
    else:
        model = Transformer(
            num_heads=4,
            embed_dim=256,
            num_hidden_layers=4,
            length_memory_block=8
        )

        init_data = np.zeros((1, 200, 5), dtype=np.uint8)
        variables = model.init(jax.random.PRNGKey(0), init_data)
        params = variables['params']

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(learning_rate=0.0005),
        dropout_rng=jax.random.PRNGKey(0),
        epoch=0
    )

    wandb.init(project="geister-s")

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=25, keep_period=50, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(config.ckpt_dir, checkpointer, options)

    Checkpoint(state.epoch, state.params, model).save(checkpoint_manager)

    ctx = multiprocessing.get_context('spawn')
    learner_update_queue = ctx.Queue(100)
    learner_request_queue = ctx.Queue(100)

    collecor_process = ctx.Process(target=collector.main, args=(
        host, port, config,
        learner_update_queue,
        learner_request_queue,
    ))
    collecor_process.start()

    while True:
        log_dict: dict = learner_request_queue.get()
        minibatch = buffer.Batch.from_npz(config.minibatch_temp_path)

        state, train_log_dict = train(state, minibatch, config.batch_size)

        log_dict.update(train_log_dict)

        Checkpoint(state.epoch, state.params, model).save(checkpoint_manager)

        learner_update_queue.put(state.epoch)

        wandb.log(log_dict)


def train(
        state: TrainState,
        train_batch: buffer.Batch,
        batch_size: int
        ) -> tuple[TrainState, dict]:
    train_batches = train_batch.divide(batch_size)
    num_batches = len(train_batches)

    loss = 0
    losses = []

    for i in tqdm(range(num_batches), desc=' Training '):
        state, loss_i, losses_i = train_step(state, *train_batches[i].astuple(), eval=False, is_rmt=True)

        loss += loss_i
        losses.append(losses_i)

    loss /= num_batches

    num_division = 2
    losses = np.reshape(losses, (num_batches, num_division, -1, 3))
    losses = np.mean(losses, axis=(0, 2))

    log_dict = {'train/loss': loss}

    for i in range(num_division):
        for j, name in enumerate(['policy', 'value', 'color']):
            log_dict[f'train/loss {name} {i}'] = losses[i, j]

    print(log_dict)

    return state.replace(epoch=state.epoch + 1), log_dict


if __name__ == "__main__":
    try:
        main()

    except Exception:
        import traceback
        traceback.print_exc()

        with open('error.log', 'w') as f:
            traceback.print_exc(file=f)