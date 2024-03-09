from typing import Generic, TypeVar
from dataclasses import dataclass

import numpy as np

from env.state import SimulationState, WinType, get_initial_state_pair
import batch


@dataclass
class ActionSelectionResult:
    action: int


T = TypeVar("T", bound=ActionSelectionResult)


class PlayerBase(Generic[T]):
    def __init__(self) -> None:
        self.state: SimulationState = None

    def init_state(self, state: SimulationState) -> list[list[int]]:
        self.state = state
        return state.create_init_tokens()

    def select_next_action(self) -> T:
        pass

    def apply_action(self, action: int, player: int, true_color_o: np.ndarray) -> list[list[int]]:
        tokens, afterstates = self.state.step(action, player)

        for i in range(len(afterstates)):
            tokens += self.state.step_afterstate(afterstates[i], true_color_o[afterstates[i].piece_id])

        return tokens


@dataclass
class GameResult:
    actions: np.ndarray
    winner: int
    win_type: WinType
    color1: np.ndarray
    color2: np.ndarray
    tokens1: list[list[int]]
    tokens2: list[list[int]]

    @staticmethod
    def create_sample(
        tokens_list: list[list[int]],
        actions: np.ndarray,
        color_o: np.ndarray,
        reward: int,
        token_length: int
    ) -> np.ndarray:
        tokens = np.zeros((token_length, 5), dtype=np.uint8)
        tokens[:min(token_length, len(tokens_list))] = tokens_list[:token_length]

        actions = actions[tokens[:, 4]]
        reward = np.array([reward], dtype=np.uint8)

        return batch.create_batch(
            tokens.astype(np.uint8),
            actions.astype(np.uint8),
            reward.astype(np.uint8),
            color_o.astype(np.uint8)
        )

    def create_sample_p(self, token_length: int) -> np.ndarray:
        reward = 3 + int(self.winner * self.win_type.value)

        return GameResult.create_sample(
            self.tokens1, self.actions, self.color2, reward, token_length
        )

    def create_sample_o(self, token_length: int) -> np.ndarray:
        reward = 3 + -1 * int(self.winner * self.win_type.value)

        return GameResult.create_sample(
            self.tokens2, self.actions, self.color1, reward, token_length
        )


def play_game(player1: PlayerBase[T], player2: PlayerBase[T], game_length=200, print_board=False) -> GameResult:
    state1, state2 = get_initial_state_pair()
    tokens1 = player1.init_state(state1)
    tokens2 = player2.init_state(state2)

    action_history = np.zeros(game_length + 20, dtype=np.int16)

    player = 1

    for i in range(game_length):
        if player == 1:
            action = player1.select_next_action().action
        else:
            action = player2.select_next_action().action

        tokens1 += player1.apply_action(action, player, state2.color_p)
        tokens2 += player2.apply_action(action, -player, state1.color_p)

        action_history[i] = action

        if print_board:
            board = np.zeros(36, dtype=np.int8)

            board[state1.pieces_p[(state1.pieces_p >= 0) & (state1.color_p == 1)]] = 1
            board[state1.pieces_p[(state1.pieces_p >= 0) & (state1.color_p == 0)]] = 2
            board[state2.pieces_p[(state2.pieces_p >= 0) & (state2.color_p == 1)]] = -1
            board[state2.pieces_p[(state2.pieces_p >= 0) & (state2.color_p == 0)]] = -2

            print(str(board.reshape((6, 6))).replace('0', ' '))
            print(i)

        if state1.is_done or state2.is_done:
            break

        player = -player

    return GameResult(
        actions=action_history,
        winner=state1.winner,
        win_type=state1.win_type,
        color1=state1.color_p,
        color2=state2.color_p,
        tokens1=tokens1,
        tokens2=tokens2
    )
