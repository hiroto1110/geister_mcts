import time
from pathlib import Path
import shutil
import warnings

import tensorflow as tf
import jax
import optax
from flax.training import checkpoints

import numpy as np
import ray
from tqdm import tqdm

from network_transformer import TrainState, TransformerDecoder, train_step
from buffer import ReplayBuffer, Sample
import geister as game
import mcts


@ray.remote(num_cpus=1, num_gpus=0)
def selfplay(train_state: TrainState, model, params, num_mcts_simulations: int, dirichlet_alpha):
    state = game.get_initial_state()

    player = 1

    node1 = mcts.create_root_node(state, train_state, model, 1)
    node2 = mcts.create_root_node(state, train_state, model, -1)

    policies = np.zeros((200, 144))

    for i in range(200):
        policy, node1, node2 = mcts.step(node1, node2, state, player, train_state, num_mcts_simulations)

        policies[i + 7] = policy

        if game.is_done(state, player):
            break

        player = -player

    record_player = np.random.choice([1, -1])

    reward = game.get_result(state, record_player)
    tokens = game.get_tokens(state, record_player, 200)
    color = state.color_o if record_player == 1 else state.color_p

    return Sample(tokens, policies, record_player, reward, color)


@ray.remote(num_cpus=1, num_gpus=0)
def testplay(train_state, params, num_mcts_simulations, dirichlet_alpha=None, n_testplay=24):
    states = [game.get_initial_state() for _ in range(n_testplay)]
    undone_states = list(states)

    alphazero = np.random.choice([1, -1])
    player = 1

    for i in range(200):
        if player == alphazero:
            step_batch(train_state, params, undone_states, player, num_sim=num_mcts_simulations)
        else:
            for state in undone_states:
                game.greedy_action(state, player, epsilon=1)

        undone_states = [state for state in undone_states if not state.is_done]
        player = -player

        if len(undone_states) == 0:
            break
    reward = [game.get_result(state, player) for state in states]
    reward = np.mean(reward)

    return reward


def main(num_cpus, n_episodes=20000, buffer_size=100000,
         batch_size=256, epochs_per_update=4,
         num_mcts_simulations=25,
         update_period=25, test_period=100,
         n_testplay=5,
         save_period=25,
         dirichlet_alpha=0.35):

    ray.init(num_cpus=num_cpus, num_gpus=1, local_mode=False)
    warnings.filterwarnings('ignore')

    logdir = Path(__file__).parent / "log"
    if logdir.exists():
        shutil.rmtree(logdir)
    summary_writer = tf.summary.create_file_writer(str(logdir))

    model = TransformerDecoder(num_heads=8, embed_dim=128, num_hidden_layers=2)

    key, key1, key2 = jax.random.split(jax.random.PRNGKey(0), 3)

    dummy_tokens = game.get_tokens(game.get_initial_state(), 1, 200)

    variables = model.init(key1, dummy_tokens[np.newaxis, ...])
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=0.00005),
        dropout_rng=key2,
        epoch=0)

    ckpt_dir = './checkpoints/'
    prefix = 'geister_'

    state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, prefix=prefix, target=state)

    current_weights = ray.put(state.params)

    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir, prefix=prefix,
        target=state, step=state.epoch, overwrite=True)

    replay = ReplayBuffer(buffer_size=buffer_size)

    work_in_progresses = [
        selfplay.remote(state, current_weights, 32, num_mcts_simulations, dirichlet_alpha)
        for _ in range(num_cpus - 4)]

    test_in_progress = testplay.remote(state, current_weights, num_mcts_simulations, n_testplay=n_testplay)

    n_updates = 0
    n = 0
    while n <= n_episodes:
        for _ in tqdm(range(update_period)):
            #: selfplayが終わったプロセスを一つ取得
            finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            replay.add_record(ray.get(finished[0]))
            work_in_progresses.extend([
                selfplay.remote(state, current_weights, 32, num_mcts_simulations, dirichlet_alpha)
            ])
            n += 1

        num_iters = epochs_per_update * (len(replay) // batch_size)
        for i in range(num_iters):
            batch = replay.get_minibatch(batch_size=batch_size)

            state, loss, info = train_step(state, *batch, eval=False)

            n_updates += 1

            if i % 100 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar("loss", loss, step=n_updates)
                    tf.summary.scalar("loss policy", info[0], step=n_updates)
                    tf.summary.scalar("loss reward", info[1], step=n_updates)
                    tf.summary.scalar("loss pieces", info[2], step=n_updates)
                    tf.summary.scalar("acc pieces", info[3], step=n_updates)

        current_weights = ray.put(state.params)

        if n % test_period == 0:
            print(f"{n - test_period}: TEST")
            win_ratio = ray.get(test_in_progress)
            print(f"SCORE: {win_ratio}")
            test_in_progress = testplay.remote(state, current_weights, num_mcts_simulations, n_testplay=n_testplay)

            with summary_writer.as_default():
                tf.summary.scalar("win_ratio", win_ratio, step=n-test_period)
                tf.summary.scalar("buffer_size", len(replay), step=n)

        if n % save_period == 0:
            checkpoints.save_checkpoint(
                ckpt_dir=ckpt_dir, prefix=prefix,
                target=state, step=n//save_period, overwrite=True, keep=5)


if __name__ == "__main__":
    try:
        main(num_cpus=31)

    except Exception:
        import traceback
        with open('error.log', 'a') as f:
            traceback.print_exc(file=f)
