import enum
import numpy as np

from base import PlayerBase
from env.state import State
from env.checkmate import find_checkmate
import env.state as game


class ResultType(enum.Flag):
    NONE = enum.auto()
    ESC = enum.auto()
    BLUE = enum.auto()
    RED = enum.auto()

    @classmethod
    def from_win_type(t: game.WinType) -> "ResultType":
        if t == game.WinType.ESCAPE:
            return ResultType.ESC
        if t == game.WinType.BLUE_4:
            return ResultType.BLUE
        if t == game.WinType.RED_4:
            return ResultType.RED


class StrategyPlayer(PlayerBase):
    strategy: np.ndarray

    def search(self, state: State, player: int, depth: int) -> ResultType:
        if depth == 0:
            return ResultType.NONE

        actions = game.get_valid_actions(state, player)

        for action in actions:
            next_state, result = state.step(action, player)

            if result.winner < 0:
                return ResultType.from_win_type(result.win_type)

            result = find_checkmate(next_state, -player, depth=5)

            if result.eval < 0:
                return ResultType.ESC

            return self.search(next_state, -player, depth - 1)


    def select_action(self):
        pass
