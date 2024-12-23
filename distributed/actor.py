import time
import multiprocessing

import numpy as np
import jax

from distributed.messages import MatchInfo, MessageMatchResult
from players.base import play_games
from players.strategy import StrategyTokenProducer


def start_selfplay_process(
    match_request_queue: multiprocessing.Queue,
    match_result_queue: multiprocessing.Queue,
    project_dir: str,
    series_length: int,
    tokens_length: int,
    seed: int
):
    jax.config.update('jax_platform_name', 'cpu')
    # jax.config.update("jax_debug_nans", True)

    np.random.seed(seed)

    with jax.default_device(jax.devices("cpu")[0]):
        while True:
            start_t = time.perf_counter()
            match: MatchInfo = match_request_queue.get()
            elapsed_t = time.perf_counter() - start_t

            name_p = match.player.name
            name_o = match.opponent.name
            print(f"Assigned: (elapsed={elapsed_t:.3f}s, agent={name_p}, opponent={name_o})")

            while True:
                try:
                    player1 = match.player.create_player(project_dir)
                    player2 = match.opponent.create_player(project_dir)

                    samples_list = play_games(
                        player1, player2,
                        num_games=series_length,
                        tokens_length=tokens_length,
                        token_producer=StrategyTokenProducer()
                    )
                    break

                except Exception as e:
                    print(f"Error playing games: {e}")
                    continue

            samples = np.stack(samples_list, dtype=np.uint8)

            match_result_queue.put(MessageMatchResult(match, samples))
