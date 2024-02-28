import numpy as np

from env.state import get_valid_actions
from players.config import PlayerRandomConfig
from players.base import PlayerBase


class PlayerRandom(PlayerBase):
    def __init__(self) -> None:
        pass

    @classmethod
    def from_config(cls, config: PlayerRandomConfig) -> "PlayerRandom":
        return PlayerRandom()

    def select_next_action(self) -> int:
        actions = get_valid_actions(self.state, 1)
        return np.random.choice(actions)


class PlayerSimple(PlayerBase):
    def __init__(self, depth_min: int, depth_max: int, num_random_ply: int, print_log=False) -> None:
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.num_random_ply = num_random_ply
        self.print_log = print_log

    @classmethod
    def from_config(cls, config: PlayerRandomConfig) -> "PlayerSimple":
        return PlayerSimple(
            depth_min=config.depth_min,
            depth_max=config.depth_max,
            num_random_ply=config.num_random_ply,
            print_log=config.print_log,
        )

    def select_next_action(self) -> int:
        action = 0

        return action
