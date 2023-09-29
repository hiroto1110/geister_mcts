import os
import multiprocessing as mp

from tqdm import tqdm

import numpy as np
import wandb

from buffer import ReplayBuffer
import geister as game


def start_selfplay_process(queue: mp.Queue, ckpt_dir: str, seed: int, num_mcts_sim: int, dirichlet_alpha: float):
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import jax
    import orbax.checkpoint

    import network_transformer as network
    import mcts

    jax.config.update('jax_platform_name', 'cpu')

    np.random.seed(seed)

    with jax.default_device(jax.devices("cpu")[0]):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer)

        ckpt = checkpoint_manager.restore(checkpoint_manager.latest_step())

        model = network.TransformerDecoderWithCache(**ckpt['model'])

        last_n_updates = checkpoint_manager.latest_step()

        mcts_params = mcts.SearchParameters(num_mcts_sim,
                                            dirichlet_alpha=dirichlet_alpha,
                                            n_ply_to_apply_noise=20,
                                            max_duplicates=3)

        player1 = mcts.PlayerMCTS(ckpt['state']['params'], model, mcts_params)
        player2 = mcts.PlayerMCTS(ckpt['state']['params'], model, mcts_params)

        while True:
            actions, color1, color2 = mcts.play_game(player1, player2)

            sample1 = player1.create_sample(actions, color2)
            sample2 = player2.create_sample(actions, color1)

            queue.put(sample1)
            queue.put(sample2)

            if last_n_updates != checkpoint_manager.latest_step():
                ckpt = checkpoint_manager.restore(checkpoint_manager.latest_step())

                player1.update_params(ckpt['state']['params'])
                player2.update_params(ckpt['state']['params'])

                last_n_updates = checkpoint_manager.latest_step()


def main(n_clients=30,
         buffer_size=200000,
         batch_size=256,
         num_batches=16,
         update_period=200,
         num_mcts_sim=50,
         dirichlet_alpha=0.3):

    import optax
    import orbax.checkpoint

    from buffer import ReplayBuffer
    import network_transformer as network

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager('./checkpoints/4_256_4', checkpointer)
    ckpt = checkpoint_manager.restore(checkpoint_manager.latest_step())

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

    run = wandb.init(project="geister-zero", config=run_config)
    name = run.name

    replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                 seq_length=game.MAX_TOKEN_LENGTH,
                                 file_name=f'replay_buffer/{name}.npz')
    replay_buffer.load('replay_buffer/189.npz')

    ckpt_dir = f'./checkpoints/{name}/'

    options = orbax.checkpoint.CheckpointManagerOptions(save_interval_steps=50, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, checkpointer, options)

    network.save_checkpoint(state, model, checkpoint_manager)

    ctx = mp.get_context('spawn')
    sample_queue = ctx.Queue()

    for i in range(n_clients):
        seed = np.random.randint(0, 10000)
        args = sample_queue, ckpt_dir, seed, num_mcts_sim, dirichlet_alpha

        process = ctx.Process(target=start_selfplay_process, args=args)
        process.start()

    while True:
        for i in tqdm(range(update_period)):
            while sample_queue.empty():
                pass

            sample = sample_queue.get()
            replay_buffer.add_sample(sample)

        log_dict = {"step": state.epoch}

        log_games(replay_buffer, update_period, log_dict)
        state = train_and_log(state, replay_buffer, batch_size, num_batches, log_dict)

        wandb.log(log_dict)

        network.save_checkpoint(state, model, checkpoint_manager)


def log_games(buffer: ReplayBuffer, num_games: int, log_dict):
    batch = buffer.get_last__minibatch(batch_size=num_games)

    n_ply = batch.tokens[:, :, game.Token.T].max(axis=1)

    log_dict["n_ply"] = n_ply.mean()
    log_dict["n_ply histgram"] = wandb.Histogram(n_ply, num_bins=50)

    value = batch.reward.flatten()
    value_count = np.bincount(np.abs(value - 3), minlength=4)

    for i in range(4):
        log_dict[f'value/{i}'] = value_count[i] / num_games


def train_and_log(state,
                  buffer: ReplayBuffer,
                  train_batch_size: int,
                  num_batches: int,
                  log_dict: dict):
    import network_transformer as network

    if len(buffer) < num_batches * train_batch_size:
        return state

    info = np.zeros((num_batches, 3))
    loss = 0

    for i in range(num_batches):
        train_batch = buffer.get_minibatch(batch_size=train_batch_size)
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
        main()

    except Exception:
        import traceback
        traceback.print_exc()

        with open('error.log', 'w') as f:
            traceback.print_exc(file=f)
