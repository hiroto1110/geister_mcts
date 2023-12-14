import os
import multiprocessing

from tqdm import tqdm
import glob

import numpy as np
import wandb

import optax
import orbax.checkpoint
import network_transformer as network
import buffer
import fsp

import collector


def main(
        host: str,
        port: int,
        buffer_size=400000,
        batch_size=64,
        num_batches=32,
        update_period=200,
        num_mcts_sim=50,
        dirichlet_alpha=0.3,
        fsp_threshold=0.6,
        minibatch_temp_path='replay_buffer/minibatch.npz'
):
    run_list = glob.glob("./checkpoints/run-*/")
    max_run_number = max([int(os.path.dirname(run)[-1]) for run in run_list])
    run_name = f'run-{max_run_number + 1}'
    run_name = 'run-3'
    prev_run_name = 'fresh-terrain-288'

    ckpt_dir = f'./checkpoints/{run_name}/'
    prev_ckpt_dir = f'./checkpoints/{prev_run_name}/'

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(prev_ckpt_dir, checkpointer)

    ckpt = checkpoint_manager.restore(769)

    model = network.TransformerDecoder(**ckpt['model'])

    state = network.TrainState.create(
        apply_fn=model.apply,
        params=ckpt['state']['params'],
        tx=optax.adam(learning_rate=0.0005),
        dropout_rng=ckpt['state']['dropout_rng'],
        epoch=ckpt['state']['epoch'])

    run_config = {
        "batch_size": batch_size,
        "num_batches": num_batches,
        "num_mcts_sim": num_mcts_sim,
        "dirichlet_alpha": dirichlet_alpha,
        }
    run_config.update(ckpt['model'])

    # wandb.init(project="geister-zero", name=run_name, config=run_config)

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=25, keep_period=50, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, checkpointer, options)

    network.save_checkpoint(state, model, checkpoint_manager)

    ctx = multiprocessing.get_context('spawn')
    learner_update_queue = ctx.Queue(100)
    learner_request_queue = ctx.Queue(100)

    collecor_process = ctx.Process(target=collector.main, args=(
        host, port,
        buffer_size,
        batch_size * num_batches,
        update_period,
        fsp_threshold,
        fsp.FSP(n_agents=1, selfplay_p=0.3, match_buffer_size=2000, p=6),
        ckpt_dir,
        minibatch_temp_path,
        learner_update_queue,
        learner_request_queue,
    ))
    collecor_process.start()

    while True:
        log_dict: dict = learner_request_queue.get()
        minibatch = buffer.Batch.from_npz(minibatch_temp_path)

        log_dict['step'] = state.epoch

        state = train_and_log(state, minibatch, num_batches, log_dict)

        network.save_checkpoint(state, model, checkpoint_manager)

        learner_update_queue.put(state.epoch)

        # wandb.log(log_dict)


def train_and_log(state,
                  train_batch: buffer.Batch,
                  num_batches: int,
                  log_dict: dict):
    import network_transformer as network

    info = np.zeros((num_batches, 3))
    loss = 0

    for i in tqdm(range(num_batches), desc=' Training '):
        state, loss_i, info_i = network.train_step(state, *train_batch.astuple(), eval=False)

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
        main(host='localhost', port=23001)

    except Exception:
        import traceback
        traceback.print_exc()

        with open('error.log', 'w') as f:
            traceback.print_exc(file=f)
