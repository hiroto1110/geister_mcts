from dataclasses import dataclass

import numpy as np

from players.base import PlayerBase, PlayerState, ActionSelectionResult, PlayerConfig
from env.state import State
import env.lib.naotti2020 as naotti2020
import gat.server_util


class PlayerNaotti2020(PlayerBase):
    def __init__(self, depth_min: int, depth_max: int, num_random_ply: int, print_log=False) -> None:
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.num_random_ply = num_random_ply
        self.print_log = print_log
    
    def init_state(
        self,
        state: State,
        prev_state: PlayerState | None = None
    ) -> tuple[State, PlayerState, list[list[int]]]:

        self.turn_count = 0

        depth = np.random.randint(self.depth_min, self.depth_max + 1)
        naotti2020.initGame(depth, self.print_log)

        return None, state.create_init_tokens()

    def select_next_action(self, state: State, player_state: PlayerState) -> ActionSelectionResult:
        board_msg = gat.server_util.encode_board_str(state)
        naotti2020.recvBoard(board_msg)

        action_msg = naotti2020.solve(self.turn_count, state.n_ply <= self.num_random_ply)
        action = gat.server_util.decode_action_message(action_msg)

        p_id = action // 4
        d_id = action % 4

        action = p_id * 4 + (3 - d_id)

        self.turn_count += 2

        return ActionSelectionResult(action)


@dataclass(frozen=True)
class PlayerNaotti2020Config(PlayerConfig[PlayerNaotti2020]):
    depth_min: int
    depth_max: int
    num_random_ply: int
    print_log: bool = False

    @property
    def name(self) -> str:
        return "naotti"

    def create_player(self, project_dir: str) -> PlayerNaotti2020:
        return PlayerNaotti2020(
            depth_min=self.depth_min,
            depth_max=self.depth_max,
            num_random_ply=self.num_random_ply,
            print_log=self.print_log,
        )
