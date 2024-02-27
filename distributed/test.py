import multiprocessing

import numpy as np
import jax
from sklearn.linear_model import LinearRegression

import actor
from messages import MatchInfo, MessageMatchResult, SnapshotInfo, SNAPSHOT_INFO_NAOTTI
from constants import SearchParametersRange, IntRange, FloatRange

from batch import get_reward


def main(
    n_clients=4,
    ckpt_dir='./data/projects/run-7',
    aganet=SnapshotInfo("main", step=300),
    series_length=8,
    tokens_length=220,
):
    mcts_params = SearchParametersRange(
        num_simulations=IntRange(400, 400),
        dirichlet_alpha=FloatRange(0.1, 0.1),
        n_ply_to_apply_noise=IntRange(0, 10),
        max_duplicates=IntRange(1, 3),
        depth_search_checkmate_leaf=IntRange(6, 6),
        depth_search_checkmate_root=IntRange(7, 7),
        c_base=IntRange(25, 40)
    )

    ctx = multiprocessing.get_context('spawn')
    match_request_queue = ctx.Queue(100)
    match_result_queue = ctx.Queue(100)

    for i in range(n_clients * 2):
        match_request_queue.put(MatchInfo(aganet, SNAPSHOT_INFO_NAOTTI))

    for i in range(n_clients):
        seed = np.random.randint(0, 10000)
        args = (match_request_queue,
                match_result_queue,
                ckpt_dir,
                {aganet.name: mcts_params},
                series_length,
                tokens_length,
                seed)

        process = ctx.Process(target=actor.start_selfplay_process, args=args)
        process.start()

    win_count = np.zeros((series_length, 2))

    for count in range(10000):
        result: MessageMatchResult = match_result_queue.get()
        match_request_queue.put(MatchInfo(aganet, SNAPSHOT_INFO_NAOTTI))

        for i, sample in enumerate(result.samples):
            if get_reward(sample) > 3:
                win_count[i, 0] += 1
            elif get_reward(sample) < 3:
                win_count[i, 1] += 1

        n = win_count.sum(axis=1)
        n[n == 0] = 1

        win_rate = win_count[:, 0] / n

        lr = LinearRegression()
        lr.fit(np.arange(len(win_rate)).reshape(-1, 1), win_rate)

        print(f"{count}, {lr.intercept_:.4f}, {lr.coef_[0]:.4f}")
        print(win_rate)


if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')

    with jax.default_device(jax.devices("cpu")[0]):
        main()
