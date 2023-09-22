from dataclasses import dataclass, replace
import multiprocessing as mp
import itertools

from flax.training import checkpoints
import jax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from multiprocessing_util import MultiSenderPipe
import network_transformer as network
import mcts
from main import PREFIX, CKPT_BACKUP_DIR


@dataclass
class GameResult:
    agent1: int
    agent2: int
    reward: int


def load_params(steps):
    params_list = []

    for ckpt_dir, step in steps:
        ckpt = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, prefix=PREFIX, step=step, target=None)
        params_list.append(ckpt['params'])

    return params_list


def start_league_process(league_queue: mp.Queue, result_sender, seed: int,
                         num_mcts_sim: int, dirichlet_alpha: float):
    np.random.seed(seed)

    with jax.default_device(jax.devices("cpu")[0]):
        model = network.TransformerDecoderWithCache(num_heads=8, embed_dim=128, num_hidden_layers=2)

        params_list = load_params(PARAMS_STEPS)

        def params_to_player(params):
            return mcts.PlayerMCTS(params, model, num_mcts_sim, dirichlet_alpha)

        players = [params_to_player(params) for params in params_list]

        for player in players:
            player.n_ply_to_apply_noise = 0

        while not league_queue.empty():
            next_game = league_queue.get()

            actions, _, color2 = mcts.play_game(players[next_game.agent1], players[next_game.agent2])
            sample = players[next_game.agent1].create_sample(actions, color2)

            result_sender.send(replace(next_game, reward=sample.reward))


PARAMS_STEPS = [(CKPT_BACKUP_DIR, i * 100) for i in range(1, 31, 3)]
# PARAMS_STEPS = [(CKPT_BACKUP_DIR, 1200), (CKPT_BACKUP_DIR, 1300), (CKPT_BACKUP_DIR, 1700)]
# PARAMS_STEPS += [('checkpoints_100/', 1100), ('checkpoints_100/', 1400), ('checkpoints_100/', 1600)]


def main(n_clients=14,
         num_games_per_combination=10,
         num_mcts_sim=50,
         dirichlet_alpha=0.1):

    num_agents = len(PARAMS_STEPS)

    league_queue = mp.Queue()

    np.random.seed(1000)

    combinations = itertools.combinations(range(num_agents), 2)
    # combinations = [(len(PARAMS_STEPS) - 2, i) for i in range(len(PARAMS_STEPS) - 1)]
    combinations = np.array(list(combinations))

    for i in range(num_games_per_combination):
        np.random.shuffle(combinations)

        for agent1, agent2 in combinations:
            if i % 2 == 0:
                league_queue.put(GameResult(agent1, agent2, 0))
            else:
                league_queue.put(GameResult(agent2, agent1, 0))

    pipe = MultiSenderPipe(n_clients)

    for i in range(n_clients):
        sender = pipe.get_sender(i)
        seed = np.random.randint(0, 10000)
        args = league_queue, sender, seed, num_mcts_sim, dirichlet_alpha

        process = mp.Process(target=start_league_process, args=args)
        process.start()

    result_table = np.zeros((num_agents, num_agents, 7))

    n_loops = len(combinations) * num_games_per_combination

    for _ in tqdm(range(n_loops)):
        while not pipe.poll():
            pass

        result = pipe.recv()

        result_table[result.agent1, result.agent2, result.reward] += 1
        result_table[result.agent2, result.agent1, 6 - result.reward] += 1

    np.save("league_result.npy", result_table)

    n_games = result_table.sum(axis=(2))
    n_games[n_games == 0] = 1

    win_ratio = result_table * np.array([0, 0, 0, 0.5, 1, 1, 1]).reshape(1, 1, -1)
    win_ratio = win_ratio.sum(axis=(2)) / n_games

    win_ratio[range(win_ratio.shape[0]), range(win_ratio.shape[1])] = 0.5

    win_ratio_mean = win_ratio.mean(axis=1).reshape(-1, 1)

    win_ratio = np.concatenate([win_ratio, win_ratio_mean], axis=1)

    y_label = [i // 100 for _, i in PARAMS_STEPS]
    x_label = y_label + ['total']

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(win_ratio, annot=True, fmt='.2f', cmap='bwr', square=True,
                     vmin=0, vmax=1,
                     xticklabels=x_label,
                     yticklabels=y_label)
    ax.xaxis.tick_top()

    plt.savefig("league_result.png")
    plt.show()


def visiblize():
    result_table = np.load('league_result.npy')

    print(result_table[0])
    return

    n_games = result_table.sum(axis=(2))
    n_games[n_games == 0] = 1

    win_ratio = result_table * np.array([0, 0, 0, 0.5, 1, 1, 1]).reshape(1, 1, -1)
    win_ratio = win_ratio.sum(axis=(2)) / n_games

    win_ratio[range(win_ratio.shape[0]), range(win_ratio.shape[1])] = 0.5

    win_ratio_mean = win_ratio.mean(axis=1).reshape(-1, 1)

    win_ratio = np.concatenate([win_ratio, win_ratio_mean], axis=1)

    y_label = [i // 100 for i in PARAMS_STEPS]
    x_label = y_label + ['total']

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(win_ratio, annot=True, fmt='.2f', cmap='bwr', square=True,
                     vmin=0, vmax=1,
                     xticklabels=x_label,
                     yticklabels=y_label)
    ax.xaxis.tick_top()
    plt.show()


if __name__ == "__main__":
    # visiblize()
    main()
