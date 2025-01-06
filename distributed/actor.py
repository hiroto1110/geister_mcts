import time
import multiprocessing

import numpy as np
import jax

from distributed.messages import MatchInfo, MessageMatchResult
from players.base import play_game


def start_selfplay_process(
    match_request_queue: multiprocessing.Queue,
    match_result_queue: multiprocessing.Queue,
    project_dir: str,
    tokens_length: int,
    seed: int
):
    jax.config.update('jax_platform_name', 'cpu')
    # jax.config.update("jax_debug_nans", True)

    np.random.seed(seed)

    with jax.default_device(jax.devices("cpu")[0]):
        prev_start_t = time.perf_counter()

        while True:
            start_t = time.perf_counter()
            match: MatchInfo = match_request_queue.get()
            time_taken_to_get_match = time.perf_counter() - start_t

            time_taken_to_play_game = start_t - prev_start_t
            prev_start_t = start_t

            msg_dict = {
                "get": f"{time_taken_to_get_match:.4f}",
                "prev": f"{time_taken_to_play_game:.4f}",
                "agent": match.player.name,
                "opponent": match.opponent.name,
            }
            print(f"Assigned: ({[key + msg_dict for key in msg_dict]})")

            while True:
                try:
                    result_msg = play_game(
                        match,
                        project_dir=project_dir,
                        tokens_length=tokens_length,
                    )
                    break

                except Exception as e:
                    print(f"Error playing games: {e}")
                    continue

            match_result_queue.put(result_msg)


def play_games(
    match: MatchInfo,
    project_dir: str,
    num_turns: int = 200,
    tokens_length: int = 240,
) -> MessageMatchResult:

    player_1 = match.player.create_player(project_dir)
    player_2 = match.opponent.create_player(project_dir)

    if np.random.random() > 0.5:
        result_1, result_2 = play_game(player_1, player_2, num_turns=num_turns)
    else:
        result_2, result_1 = play_game(player_2, player_1, num_turns=num_turns)

    sample_1 = player_1.create_sample(result_1, result_2, tokens_length)
    sample_2 = player_2.create_sample(result_2, result_1, tokens_length)

    return MessageMatchResult(match, sample_1, sample_2)