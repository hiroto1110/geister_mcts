import multiprocessing

import numpy as np
import jax

import actor
from messages import MatchInfo, MessageMatchResult, SnapshotInfo
from constants import SearchParametersRange, IntRange, FloatRange

from batch import get_reward


def main(
    n_clients: int = 12,
    ckpt_dir: str = './data/checkpoints/run-3',
    series_length=12,
    tokens_length=220,
):
    mcts_params = SearchParametersRange(
        num_simulations=IntRange(20, 40),
        dirichlet_alpha=FloatRange(0.1, 0.2),
        n_ply_to_apply_noise=IntRange(0, 10),
        max_duplicates=IntRange(1, 3),
        depth_search_checkmate_leaf=IntRange(4, 4),
        depth_search_checkmate_root=IntRange(7, 7),
        c_base=IntRange(25, 40)
    )

    ctx = multiprocessing.get_context('spawn')
    match_request_queue = ctx.Queue(100)
    match_result_queue = ctx.Queue(100)

    for i in range(n_clients * 2):
        match_request_queue.put(MatchInfo(SnapshotInfo("main", -1), SnapshotInfo("NAOTTI2020", -1)))

    for i in range(n_clients):
        seed = np.random.randint(0, 10000)
        args = (match_request_queue,
                match_result_queue,
                ckpt_dir,
                {"main": mcts_params},
                series_length,
                tokens_length,
                seed)

        process = ctx.Process(target=actor.start_selfplay_process, args=args)
        process.start()

    win_count = np.zeros((series_length, 2))

    for count in range(10000):
        result: MessageMatchResult = match_result_queue.get()
        match_request_queue.put(MatchInfo(SnapshotInfo("main", -1), SnapshotInfo("NAOTTI2020", -1)))

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
