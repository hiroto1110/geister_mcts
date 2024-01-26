import multiprocessing

from tqdm import tqdm
import click

import numpy as np
import jax
import optax
import orbax.checkpoint
import wandb

from network.train import Checkpoint, TrainState, train_step
from batch import load, astuple

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

    model, params = config.init_params.create_model_and_params()

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(learning_rate=config.learning_rate),
        dropout_rng=jax.random.PRNGKey(0),
        epoch=0,
        init_memory=TrainState.create_init_memory(model)
    )

    wandb.init(project=config.project_name)

    checkpoint_manager = orbax.checkpoint.CheckpointManager(config.ckpt_dir, checkpointer, options=config.ckpt_options)

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
        minibatch = load(config.minibatch_temp_path)

        state, train_log_dict = train(state, minibatch, config.batch_size)

        log_dict.update(train_log_dict)

        Checkpoint(state.epoch, state.params, model).save(checkpoint_manager)

        learner_update_queue.put(state.epoch)

        wandb.log(log_dict)


def train(
    state: TrainState,
    train_batch: np.ndarray,
    batch_size: int
) -> tuple[TrainState, dict]:
    train_batches = np.split(train_batch, len(train_batch) // batch_size)
    num_batches = len(train_batches)

    loss = 0
    losses = []

    for i in tqdm(range(num_batches), desc=' Training '):
        state, loss_i, losses_i = train_step(state, *astuple(train_batches[i]), num_division_of_segment=4, eval=False)

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
