import multiprocessing

import numpy as np
import jax

import actor
from messages import MatchInfo, MatchResult
from constants import SearchParameters

from batch import get_reward


def main(
    n_clients: int = 12,
    ckpt_dir: str = './data/checkpoints/run-3',
    series_length=8,
    tokens_length=220,
):
    mcts_params_min = SearchParameters(
        num_simulations=20,
        dirichlet_alpha=0.1,
        n_ply_to_apply_noise=0,
        max_duplicates=1,
        depth_search_checkmate_leaf=4,
        depth_search_checkmate_root=7,
        visibilize_node_graph=False,
        c_base=25,
    )

    mcts_params_max = SearchParameters(
        num_simulations=40,
        dirichlet_alpha=0.2,
        n_ply_to_apply_noise=0,
        max_duplicates=3,
        depth_search_checkmate_leaf=4,
        depth_search_checkmate_root=7,
        visibilize_node_graph=False,
        c_base=40,
    )

    ctx = multiprocessing.get_context('spawn')
    match_request_queue = ctx.Queue(100)
    match_result_queue = ctx.Queue(100)
    ckpt_queues = [ctx.Queue(100) for _ in range(n_clients)]

    for i in range(n_clients * 2):
        match_request_queue.put(MatchInfo(-2))

    for i in range(n_clients):
        seed = np.random.randint(0, 10000)
        args = (match_request_queue,
                match_result_queue,
                ckpt_queues[i],
                ckpt_dir,
                seed,
                mcts_params_min,
                mcts_params_max,
                series_length,
                tokens_length)

        process = ctx.Process(target=actor.start_selfplay_process, args=args)
        process.start()

    win_count = np.zeros((series_length, 2))

    for count in range(10000):
        result: MatchResult = match_result_queue.get()
        match_request_queue.put(MatchInfo(-2))

        for i, sample in enumerate(result.samples):
            if get_reward(sample) > 3:
                win_count[i, 0] += 1
            elif get_reward(sample) < 3:
                win_count[i, 1] += 1

        n = win_count.sum(axis=1)
        n[n == 0] = 1

        win_rate = win_count[:, 0] / n

        print(count, [f'{w:.3f}' for w in win_rate])


if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')

    with jax.default_device(jax.devices("cpu")[0]):
        main()
