from dataclasses import dataclass

import numpy as np

from env.state import State, get_valid_actions
from players.base import PlayerBase, PlayerState, ActionSelectionResult, PlayerConfig


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class PlayerRandomConfig(PlayerConfig[PlayerRandom]):
    def create_player(self, project_dir: str) -> PlayerRandom:
        return PlayerRandom()


@dataclass(frozen=True)
class PlayerTracing(PlayerBase):
    tracing_actions: list[int]

    def select_next_action(
        self,
        state: State,
        player_state: PlayerState,
        actions: list[int] = None
    ) -> ActionSelectionResult:
        if actions is None:
            actions = get_valid_actions(state, 1)

        if len(self.tracing_actions) <= state.n_ply:
            print("not enough length", state.n_ply)
            action = np.random.choice(actions)
        else:
            action = self.tracing_actions[state.n_ply]

        if action not in actions:
            print("invalid action", action, actions)
            action = np.random.choice(actions)

        # assert action in actions, f"invalid action {action}, valid actions {actions}"

        return ActionSelectionResult(action)


@dataclass(frozen=True)
class PlayerTracingConfig(PlayerConfig[PlayerTracing]):
    tracing_actions: list[int]

    def create_player(self, project_dir: str) -> PlayerTracing:
        return PlayerTracing(self.tracing_actions)