from __future__ import annotations

import enum
import dataclasses
import itertools

import numpy as np

from players.base import PlayerBase, PlayerState, ActionSelectionResult, PlayerConfig
from env.state import State
from env.checkmate import find_checkmate
import env.state as game


class GameResult(enum.IntFlag):
    NONE = 0
    WON_E = 2**0
    WON_B = 2**1
    WON_R = 2**2
    LST_E = 2**3
    LST_B = 2**4
    LST_R = 2**5

    @classmethod
    def from_step_result(cls, result: game.StepResult) -> "GameResult":
        if result.winner > 0:
            if result.win_type == game.WinType.BLUE_4:
                return GameResult.WON_B
            if result.win_type == game.WinType.RED_4:
                return GameResult.WON_R
            if result.win_type == game.WinType.ESCAPE:
                return GameResult.WON_E

        if result.winner < 0:
            if result.win_type == game.WinType.BLUE_4:
                return GameResult.LST_B
            if result.win_type == game.WinType.RED_4:
                return GameResult.LST_R
            if result.win_type == game.WinType.ESCAPE:
                return GameResult.LST_E

        return GameResult.NONE


@dataclasses.dataclass(frozen=True)
class PlayerStatisticsZ(PlayerBase):
    z: np.ndarray
    player: PlayerBase
    checkmate_depth: int

    def init_state(self, state: game.State, prev_state: PlayerState = None) -> tuple[PlayerState, list[list[int]]]:
        return self.player.init_state(state, prev_state)

    def apply_action(
        self,
        state: game.State,
        player_state: PlayerState,
        action: int, player: int,
        true_color_o: np.ndarray
    ) -> tuple[PlayerState, game.State, list[list[int]], game.StepResult]:
        return self.player.apply_action(state, player_state, action, player, true_color_o)

    def select_next_action(self, state: State, player_state: PlayerState) -> ActionSelectionResult:
        action_dict = self.create_action_dict(state)

        if len(action_dict.keys()) == 1:
            return self.player.select_next_action(state, player_state)

        flags = [int(flag) for flag in action_dict.keys()]

        p = self.z[flags]
        p = p / np.sum(p)
        flag = np.random.choice(flags, p=p)

        actions = action_dict[flag]

        return self.player.select_next_action(state, player_state, actions)

    def create_action_dict(self, state: State):
        action_dict: dict[GameResult, list[int]] = {}

        for action in game.get_valid_actions(state, player=1):
            flag = self.search(state, player=1, action=action, depth=2)

            if flag not in action_dict:
                action_dict[flag] = []

            action_dict[flag].append(action)

        return action_dict

    def search(self, state: State, player: int, action: int, depth: int) -> GameResult:
        if depth == 0:
            return GameResult.NONE

        next_state, a_result = state.step(action, player)

        next_states: list[tuple[State, game.StepResult]] = []

        if len(a_result.afterstates) == 0:
            next_states.append((next_state, a_result))

        else:
            for colors in itertools.product([game.BLUE, game.RED], repeat=len(a_result.afterstates)):
                for i, color in enumerate(colors):
                    next_state_i, a_result_i = next_state.step_afterstate(a_result.afterstates[i], color)

                    if a_result_i.winner != 0:
                        break
                next_states.append((next_state_i, a_result_i))

        result = GameResult.NONE

        for next_state_i, a_result_i in next_states:
            if a_result_i.winner != 0:
                result |= GameResult.from_step_result(a_result_i)
            else:
                c_result = find_checkmate(next_state_i, -player, depth=self.checkmate_depth)

                if c_result.eval > 0:
                    result |= GameResult.WON_E
                    continue
                if c_result.eval < 0:
                    result |= GameResult.LST_E
                    continue

                if depth > 1:
                    for next_action in game.get_valid_actions(next_state_i, -player):
                        result |= self.search(next_state_i, -player, next_action, depth - 1)

        return result


@dataclasses.dataclass
class StatisticsZFactory:
    def create_z(self) -> np.ndarray:
        pass


@dataclasses.dataclass
class StatisticsZFactoryRandom:
    def create_z(self) -> np.ndarray:
        return np.random.random(2**6)


@dataclasses.dataclass
class PlayerStatisticsZConfig(PlayerConfig[PlayerStatisticsZ]):
    z: StatisticsZFactory
    player: PlayerConfig
    checkmate_depth: int

    def get_name(self) -> str:
        return "Statistics"

    def create_player(self, project_dir: str) -> PlayerStatisticsZ:
        return PlayerStatisticsZ(
            z=self.z.create_z(),
            player=self.player.create_player(project_dir),
            checkmate_depth=self.checkmate_depth
        )


def test_game():
    from players.base import play_game
    from players.mcts import SearchParameters, PlayerMCTS

    np.random.seed(3)

    mcts_params = SearchParameters(
        num_simulations=100,
        time_limit=20
    )
    from network.checkpoints import Checkpoint
    ckpt = Checkpoint.from_json_file("./data/projects/run-7/main/600.json")

    player_mcts = PlayerMCTS(ckpt.params, ckpt.model.create_caching_model(), mcts_params)

    player = player_mcts

    play_game(player, player, print_board=True)


if __name__ == "__main__":
    test_game()
