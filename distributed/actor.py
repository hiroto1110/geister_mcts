import time
import multiprocessing

import numpy as np
import jax

from network.checkpoints import Checkpoint, CheckpointManager
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
        main(
            match_request_queue,
            match_result_queue,
            ckpt_dir,
            mcts_params_ranges,
            series_length,
            tokens_length
        )


class Agent:
    def __init__(self, name: str, mcts_params_range: SearchParametersRange, ckpt_dir: str) -> None:
        self.name = name
        self.mcts_params_range = mcts_params_range

        self.checkpoint_manager = CheckpointManager(f'{ckpt_dir}/{name}')

        self.current_ckpt = self.checkpoint_manager.load(step=-1)

        self.snapshots: dict[int, Checkpoint] = {}

    def get_snapshot_ckpt(self, step: int) -> Checkpoint:
        if step not in self.snapshots:
            self.snapshots[step] = self.checkpoint_manager.load(step)

        return self.snapshots[step]

    def update(self):
        if self.checkpoint_manager.lastest_step() == self.current_ckpt.step:
            return
        self.current_ckpt = self.checkpoint_manager.load(step=-1)


def main(
    match_request_queue: multiprocessing.Queue,
    match_result_queue: multiprocessing.Queue,
    ckpt_dir: str,
    mcts_params_ranges: dict[str, SearchParametersRange],
    series_length: int,
    tokens_length: int,
):
    agents = {
        name: Agent(name, mcts_params_ranges[name], ckpt_dir)
        for name in mcts_params_ranges
    }

    def create_player(agent: Agent, ckpt: Checkpoint) -> mcts.PlayerMCTS:
        params = ckpt.params
        model = ckpt.model.create_caching_model()

        mcts_params = agent.mcts_params_range.sample()

        return mcts.PlayerMCTS(params, model, mcts_params, tokens_length)

    def create_snapshot_player(info: SnapshotInfo) -> mcts.PlayerMCTS:
        agent = agents[info.name]
        return create_player(agent, agent.get_snapshot_ckpt(info.step))

    def cretae_current_player(info: SnapshotInfo) -> mcts.PlayerMCTS:
        agent = agents[info.name]
        return create_player(agent, agent.current_ckpt)

    while True:
        for agent in agents.values():
            agent.update()

        start_t = time.perf_counter()
        match: MatchInfo = match_request_queue.get()
        elapsed_t = time.perf_counter() - start_t
        print(f"Assigned: (elapsed={elapsed_t:.3f}s, \
              agent={match.player.name}-{match.player.step}, \
              opponent={match.opponent.name}-{match.opponent.step})")

        player1 = cretae_current_player(match.player)

        if match.opponent.name == SNAPSHOT_INFO_NAOTTI.name:
            player2 = mcts.PlayerNaotti2020(depth_min=3, depth_max=6, random_n_ply=12)
        else:
            player2 = create_snapshot_player(match.opponent)

        samples = []

        for i in range(series_length):
            if np.random.random() > 0.5:
                actions, color1, color2 = mcts.play_game(player1, player2)
            else:
                actions, color2, color1 = mcts.play_game(player2, player1)

            sample = player1.create_sample(actions, color2)
            samples.append(sample)

        match_result_queue.put(MessageMatchResult(match, samples))
