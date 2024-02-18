import time
import multiprocessing

import numpy as np
import jax

from network.checkpoints import CheckpointManager
from network.transformer import Transformer
from messages import SnapshotInfo, MatchInfo, MessageMatchResult, SNAPSHOT_INFO_NAOTTI
from constants import SearchParametersRange
import mcts


def start_selfplay_process(
    match_request_queue: multiprocessing.Queue,
    match_result_queue: multiprocessing.Queue,
    ckpt_dir: str,
    mcts_params_ranges: dict[str, SearchParametersRange],
    series_length: int,
    tokens_length: int,
    seed: int
):
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_debug_nans", True)

    np.random.seed(seed)

    with jax.default_device(jax.devices("cpu")[0]):
        models = {}

        while True:
            start_t = time.perf_counter()
            match: MatchInfo = match_request_queue.get()
            elapsed_t = time.perf_counter() - start_t
            print(
                f"Assigned: (elapsed={elapsed_t:.3f}s, \
                agent={match.player.name}-{match.player.step}, \
                opponent={match.opponent.name}-{match.opponent.step})"
            )

            samples = play_games(match, models, ckpt_dir, mcts_params_ranges, series_length, tokens_length)
            match_result_queue.put(MessageMatchResult(match, samples))


def play_games(
    match: MatchInfo,
    models: dict[str, Transformer],
    ckpt_dir: str,
    mcts_params_ranges: dict[str, SearchParametersRange],
    series_length: int,
    tokens_length: int,
):
    def create_player(info: SnapshotInfo) -> mcts.PlayerMCTS:
        ckpt = CheckpointManager(f'{ckpt_dir}/{info.name}').load(info.step)

        if info.name not in models:
            models[info.name] = ckpt.model.create_caching_model()

        return mcts.PlayerMCTS(
            ckpt.params,
            models[info.name],
            mcts_params_ranges[info.name].sample(),
            tokens_length
        )

    player1 = create_player(match.player)

    if match.opponent.name == SNAPSHOT_INFO_NAOTTI.name:
        player2 = mcts.PlayerNaotti2020(depth_min=3, depth_max=6, random_n_ply=8)
    else:
        player2 = create_player(match.opponent)

    samples = []

    for i in range(series_length):
        if np.random.random() > 0.5:
            actions, color1, color2 = mcts.play_game(player1, player2)
        else:
            actions, color2, color1 = mcts.play_game(player2, player1)

        sample = player1.create_sample(actions, color2)
        samples.append(sample)

    return np.stack(samples, dtype=np.uint8)
