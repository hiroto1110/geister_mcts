import os
import multiprocessing
from dataclasses import dataclass

from tqdm import tqdm

import numpy as np
import wandb

from buffer import ReplayBuffer, Sample
import geister as game


@dataclass
class MatchResult:
    sample1: Sample
    sample2: Sample
    agent_id: int

    def is_winning(self):
        return self.sample1.reward > 3


def start_selfplay_process(match_request_queue: multiprocessing.Queue,
                           match_result_queue: multiprocessing.Queue,
                           ckpt_queue: multiprocessing.Queue,
                           ckpt_dir: str,
                           ckpt_init_step: int,
                           seed: int,
                           num_mcts_sim: int,
                           dirichlet_alpha: float):
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

        ckpt = checkpoint_manager.restore(ckpt_init_step)

        model = network.TransformerDecoderWithCache(**ckpt['model'])
        current_params = ckpt['state']['params']

        params_checkpoints = [current_params]

        mcts_params = mcts.SearchParameters(num_mcts_sim,
                                            dirichlet_alpha=dirichlet_alpha,
                                            n_ply_to_apply_noise=20,
                                            max_duplicates=3,
                                            depth_search_checkmate_leaf=4,
                                            depth_search_checkmate_root=7,
                                            should_do_visibilize_node_graph=False)

        while True:
            agent_id = match_request_queue.get()
            opponent_params = params_checkpoints[agent_id]

            player1 = mcts.PlayerMCTS(current_params, model, mcts_params)
            player2 = mcts.PlayerMCTS(opponent_params, model, mcts_params)

            if np.random.random() > 0.5:
                actions, color1, color2 = mcts.play_game(player1, player2, game_length=180)
            else:
                actions, color2, color1 = mcts.play_game(player2, player1, game_length=180)
                actions = actions[::-1]

            sample1 = player1.create_sample(actions[0], color2)
            sample2 = player2.create_sample(actions[1], color1)

            match_result_queue.put(MatchResult(sample1, sample2, agent_id))

            if not ckpt_queue.empty():
                step, is_league_member = ckpt_queue.get()
                if is_league_member:
                    params_checkpoints.append(current_params)

                # print(f'update: {step}')
                ckpt = checkpoint_manager.restore(step)
                current_params = ckpt['state']['params']


def main(n_clients=30,
         buffer_size=200000,
         batch_size=256,
         num_batches=16,
         update_period=200,
         num_mcts_sim=50,
         dirichlet_alpha=0.3,
         fsp_threshold=0.6,
         log_wandb=True):

    import optax
    import orbax.checkpoint

    from buffer import ReplayBuffer
    from fsp import FSP
    import network_transformer as network

    from jax import config
    config.update("jax_debug_nans", True)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager('./checkpoints/4_256_2', checkpointer)
    # ckpt = checkpoint_manager.restore(checkpoint_manager.latest_step())
    ckpt = checkpoint_manager.restore(0)

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

    if log_wandb:
        run = wandb.init(project="geister-zero", config=run_config)
        name = run.name
    else:
        name = "test"

    replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                 seq_length=game.MAX_TOKEN_LENGTH,
                                 file_name=f'replay_buffer/{name}.npz')
    # replay_buffer.load('replay_buffer/189.npz')

    ckpt_dir = f'./checkpoints/{name}/'

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=50, keep_period=50, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, checkpointer, options)

    network.save_checkpoint(state, model, checkpoint_manager)

    ctx = multiprocessing.get_context('spawn')
    match_request_queue = ctx.Queue()
    match_result_queue = ctx.Queue()
    ckpt_queues = [ctx.Queue() for _ in range(n_clients)]

    for _ in range(n_clients * 2):
        match_request_queue.put(0)

    for i in range(n_clients):
        seed = np.random.randint(0, 10000)
        args = (match_request_queue,
                match_result_queue,
                ckpt_queues[i],
                ckpt_dir,
                0,
                seed,
                num_mcts_sim,
                dirichlet_alpha)

        process = ctx.Process(target=start_selfplay_process, args=args)
        process.start()

    fsp = FSP(n_agents=1, match_buffer_size=2000, p=5)

    while True:
        for i in tqdm(range(update_period)):
            while match_result_queue.empty():
                pass

            match_result = match_result_queue.get()
            replay_buffer.add_sample(match_result.sample1)
            # replay_buffer.add_sample(match_result.sample2)
            fsp.apply_match_result(match_result.agent_id, match_result.is_winning())

            match_request_queue.put(fsp.next_match())

        log_dict = {'step': state.epoch}

        log_games(replay_buffer, update_period, log_dict)
        state = train_and_log(state, replay_buffer, batch_size, num_batches, log_dict)

        network.save_checkpoint(state, model, checkpoint_manager)

        is_league_member, win_rate = fsp.is_winning_all_agents(fsp_threshold)
        if is_league_member:
            fsp.add_agent()

        print(win_rate)

        for i in range(len(win_rate)):
            log_dict[f'fsp/win_rate_{i}'] = win_rate[i]

        log_dict['fsp/n_agents'] = fsp.n_agents

        for ckpt_queue in ckpt_queues:
            ckpt_queue.put((state.epoch, is_league_member))

        if log_wandb:
            wandb.log(log_dict)


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

        try:
            state, loss_i, info_i = network.train_step(state, *train_batch.create_inputs(False), eval=False)
        except Exception as e:
            print(train_batch.tokens.min(axis=(0, 1)), train_batch.tokens.max(axis=(0, 1)))
            raise e

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
