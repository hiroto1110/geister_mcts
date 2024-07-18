import numpy as np

from env.state import State, get_valid_actions
from players.base import PlayerBase, PlayerState, ActionSelectionResult, PlayerConfig


class PlayerRandom(PlayerBase):
    def select_next_action(
        self,
        state: State,
        player_state: PlayerState,
        actions: list[int] = None
    ) -> ActionSelectionResult:
        if actions is None:
            actions = get_valid_actions(state, 1)

        return ActionSelectionResult(np.random.choice(actions))


class PlayerRandomConfig(PlayerConfig[PlayerRandom]):
    def create_player(self, project_dir: str) -> PlayerRandom:
        return PlayerRandomConfig()
