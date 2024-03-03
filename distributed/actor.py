import time
import multiprocessing

import numpy as np
import jax

from messages import MatchInfo, MessageMatchResult
from players.factory import create_player
from players.base import play_game


def start_selfplay_process(
    match_request_queue: multiprocessing.Queue,
    match_result_queue: multiprocessing.Queue,
    project_dir: str,
    series_length: int,
    tokens_length: int,
    seed: int
):
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_debug_nans", True)

    np.random.seed(seed)

    with jax.default_device(jax.devices("cpu")[0]):
        while True:
            start_t = time.perf_counter()
            match: MatchInfo = match_request_queue.get()
            elapsed_t = time.perf_counter() - start_t

            name_p = match.player.get_name()
            name_o = match.opponent.get_name()
            print(f"Assigned: (elapsed={elapsed_t:.3f}s, agent={name_p}, opponent={name_o})")

            samples = play_games(match, project_dir, series_length, tokens_length)
            match_result_queue.put(MessageMatchResult(match, samples))


def play_games(
    match: MatchInfo,
    project_dir: str,
    series_length: int,
    tokens_length: int,
):
    player1 = create_player(match.player, project_dir)
    player2 = create_player(match.opponent, project_dir)

    samples = []

    for i in range(series_length):
        if np.random.random() > 0.5:
            result = play_game(player1, player2)
            sample = result.create_sample_p(tokens_length)
        else:
            result = play_game(player2, player1)
            sample = result.create_sample_o(tokens_length)

        samples.append(sample)

    return np.stack(samples, dtype=np.uint8)
