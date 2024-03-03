import numpy as np

from players.config import PlayerNaotti2020Config
from players.base import PlayerBase
from env.state import SimulationState
import env.naotti2020 as naotti2020
import gat.server_util


class PlayerNaotti2020(PlayerBase):
    def __init__(self, depth_min: int, depth_max: int, num_random_ply: int, print_log=False) -> None:
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.num_random_ply = num_random_ply
        self.print_log = print_log

    @classmethod
    def from_config(cls, config: PlayerNaotti2020Config, project_dir: str) -> "PlayerNaotti2020":
        return PlayerNaotti2020(
            depth_min=config.depth_min,
            depth_max=config.depth_max,
            num_random_ply=config.num_random_ply,
            print_log=config.print_log,
        )

    def init_state(self, state: SimulationState):
        self.state = state
        self.turn_count = 0

        depth = np.random.randint(self.depth_min, self.depth_max + 1)
        naotti2020.initGame(depth, self.print_log)

        return state.create_init_tokens()

    def select_next_action(self) -> int:
        board_msg = gat.server_util.encode_board_str(self.state)
        naotti2020.recvBoard(board_msg)

        action_msg = naotti2020.solve(self.turn_count, self.state.n_ply <= self.num_random_ply)
        action = gat.server_util.decode_action_message(action_msg)

        if self.state.root_player == 1:
            p_id = action // 4
            d_id = action % 4

            action = p_id * 4 + (3 - d_id)

        self.turn_count += 2

        return action
