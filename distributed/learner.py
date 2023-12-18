from typing import Union
import os
import multiprocessing

from tqdm import tqdm
import glob
import click

import numpy as np
import jax
import optax
import orbax.checkpoint
# import wandb

from network.train import Checkpoint, TrainState, train_step
from network.transformer import TransformerDecoder
import buffer
import match_makers
import mcts

import collector


@click.command()
@click.argument('host', type=str)
@click.argument('port', type=int)
def main(
        host: str,
        port: int,
        buffer_size=400000,
        batch_size=64,
        num_batches=32,
        update_period=200,
        fsp_threshold=0.6,
        prev_run_dir: Union[str, None] = None,
        prev_run_step: Union[int, None] = None,
        minibatch_temp_path='replay_buffer/minibatch.npz',
):
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    if prev_run_dir is not None:
        checkpoint_manager = orbax.checkpoint.CheckpointManager(prev_run_dir, checkpointer)
        ckpt = Checkpoint.load(checkpoint_manager, prev_run_step, tx=optax.adam(learning_rate=0.0005))

        model = ckpt.model
        state = ckpt.state
    else:
        model = TransformerDecoder(
            num_heads=4,
            embed_dim=256,
            num_hidden_layers=4
        )

        init_data = np.zeros((1, 200, 5), dtype=np.uint8)
        variables = model.init(jax.random.PRNGKey(0), init_data)

        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optax.adam(learning_rate=0.0005),
            dropout_rng=jax.random.PRNGKey(0),
            epoch=0
        )

    run_list = glob.glob("./checkpoints/run-*/")
    max_run_number = max([int(os.path.dirname(run)[-1]) for run in run_list])
    run_name = f'run-{max_run_number + 1}'
    # run_name = 'run-3'

    ckpt_dir = f'./checkpoints/{run_name}/'

    """run_config = {
        "batch_size": batch_size,
        "num_batches": num_batches,
    }"""

    # wandb.init(project="geister-zero", name=run_name, config=run_config)

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=25, keep_period=50, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, checkpointer, options)

    Checkpoint(state, model).save(checkpoint_manager)

    mcts_params = mcts.SearchParameters(
        num_simulations=25,
        dirichlet_alpha=0.3,
        c_base=25
    )

    ctx = multiprocessing.get_context('spawn')
    learner_update_queue = ctx.Queue(100)
    learner_request_queue = ctx.Queue(100)

    collecor_process = ctx.Process(target=collector.main, args=(
        host, port,
        buffer_size,
        batch_size * num_batches,
        update_period,
        match_makers.MatchMakerFSP(n_agents=1, selfplay_p=0.3, match_buffer_size=2000, p=6),
        fsp_threshold,
        mcts_params,
        ckpt_dir,
        minibatch_temp_path,
        learner_update_queue,
        learner_request_queue,
    ))
    collecor_process.start()

    while True:
        log_dict: dict = learner_request_queue.get()
        minibatch = buffer.Batch.from_npz(minibatch_temp_path)

        state = train_and_log(state, minibatch, num_batches, log_dict)

        Checkpoint(state, model).save(checkpoint_manager)

        learner_update_queue.put(state.epoch)

        # wandb.log(log_dict)


def train_and_log(state: TrainState,
                  train_batch: buffer.Batch,
                  num_batches: int,
                  log_dict: dict):
    info = np.zeros((num_batches, 3))
    loss = 0

    train_batches = train_batch.divide(num_batches)

    for i in tqdm(range(num_batches), desc=' Training '):
        state, loss_i, info_i = train_step(state, *train_batches[i].astuple(), eval=False)

        loss += loss_i
        info[i] = info_i

    info = info.mean(axis=0)
    loss /= num_batches

    log_dict["train/loss"] = loss
    log_dict["train/loss policy"] = info[0]
    log_dict["train/loss value"] = info[1]
    log_dict["train/loss color"] = info[2]

    return state.replace(epoch=state.epoch + 1)


if __name__ == "__main__":
    try:
        main()

    except Exception:
        import traceback
        traceback.print_exc()

        with open('error.log', 'w') as f:
            traceback.print_exc(file=f)
