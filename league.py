from dataclasses import dataclass, replace
import multiprocessing as mp
import itertools

from flax.training import checkpoints
import jax
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from multiprocessing_util import MultiSenderPipe
import network_transformer as network
import mcts
from main import PREFIX


@dataclass
class GameResult:
    agent1: int
    agent2: int
    reward: int


def load_params(steps):
    params_list = []

    for step in steps:
        ckpt = checkpoints.restore_checkpoint(ckpt_dir='./checkpoints_100/', prefix=PREFIX, step=step, target=None)
        params_list.append(ckpt['params'])

    return params_list


def start_league_process(league_queue: mp.Queue, result_sender, seed: int,
                         num_mcts_sim: int, dirichlet_alpha: float):
    np.random.seed(seed)

    with jax.default_device(jax.devices("cpu")[0]):
        model = network.TransformerDecoderWithCache(num_heads=8, embed_dim=128, num_hidden_layers=2)

        params_list = load_params(PARAMS_STEPS)

        while not league_queue.empty():
            next_game = league_queue.get()

            _, _, reward, _ = mcts.play_game(model,
                                             params_list[next_game.agent1],
                                             params_list[next_game.agent2],
                                             num_mcts_sim, num_mcts_sim,
                                             dirichlet_alpha, record_player=1)

            result_sender.send(replace(next_game, reward=reward))


PARAMS_STEPS = [i * 100 for i in range(17)]
# PARAMS_STEPS = [100, 200, 300, 400, 500, 600, 700, 3010]


def main(n_clients=30,
         num_games_per_combination=100,
         num_mcts_sim=50,
         dirichlet_alpha=0.3):

    num_agents = len(PARAMS_STEPS)

    league_queue = mp.Queue()

    combinations = itertools.combinations(range(num_agents), 2)
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

        result_table[result.agent1, result.agent2, result.reward + 3] += 1
        result_table[result.agent2, result.agent1, -result.reward + 3] += 1

    np.save("league_result.npy", result_table)

    n_games = result_table.sum(axis=(2))
    n_games[n_games == 0] = 1

    win_ratio = result_table * np.array([0, 0, 0, 0.5, 1, 1, 1]).reshape(1, 1, -1)
    win_ratio = win_ratio.sum(axis=(2)) / n_games

    win_ratio[range(win_ratio.shape[0]), range(win_ratio.shape[1])] = 0.5

    print(win_ratio)
    print()
    print(win_ratio.mean(axis=1))

    fig, ax = plt.subplots()
    ax.pcolor(win_ratio, cmap='bwr')

    plt.savefig("league_result.png")
    # plt.show()


if __name__ == "__main__":
    main()
