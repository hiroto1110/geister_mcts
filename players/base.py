from typing import Generic, TypeVar
from dataclasses import dataclass

import numpy as np

from env.state import SimulationState, WinType, get_initial_state_pair, get_valid_actions
from game_analytics import states_to_str
import batch


@dataclass
class ActionSelectionResult:
    action: int


T = TypeVar("T", bound=ActionSelectionResult)


class PlayerBase(Generic[T]):
    def __init__(self) -> None:
        self.state: SimulationState = None

    def select_first_action(self, state: SimulationState) -> int:
        return np.random.choice(get_valid_actions(state, player=1))

    def init_state(self, state: SimulationState, first_action: int = None) -> list[list[int]]:
        self.state = state
        return state.create_init_tokens()

    def select_next_action(self) -> T:
        pass

    def apply_action(self, action: int, player: int, true_color_o: np.ndarray) -> list[list[int]]:
        if player == -1:
            action = 31 - action
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
            self.tokens1, self.actions, self.color2[::-1], reward, token_length
        )

    def create_sample_o(self, token_length: int) -> np.ndarray:
        reward = 3 + -1 * int(self.winner * self.win_type.value)

        return GameResult.create_sample(
            self.tokens2, self.actions, self.color1[::-1], reward, token_length
        )


def play_game(player1: PlayerBase[T], player2: PlayerBase[T], game_length=200, print_board=False) -> GameResult:
    action_history = np.zeros(game_length + 20, dtype=np.int16)

    state1, state2 = get_initial_state_pair()

    tokens1 = player1.init_state(state1)
    tokens2 = player2.init_state(state2)

    player = 1

    for i in range(game_length):
        if player == 1:
            action = player1.select_next_action().action
        else:
            action = player2.select_next_action().action

        tokens1 += player1.apply_action(action, player, state2.color_p[::-1])
        tokens2 += player2.apply_action(action, -player, state1.color_p[::-1])

        action_history[i] = action

        if print_board:
            s = states_to_str(
                states=[state1, state2],
                predicted_colors=[[0.5]*8, [0.5]*8],
                true_colors=[None, None],
                colored=True
            )
            print(s)
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
