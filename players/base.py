from typing import Generic, TypeVar
from dataclasses import dataclass

import numpy as np

import env.state as game
from env.state import State, WinType, get_initial_state_pair, get_valid_actions
from game_analytics import states_to_str
import batch


@dataclass
class ActionSelectionResult:
    action: int


@dataclass
class PlayerState:
    pass


T_Result = TypeVar("T_Result", bound=ActionSelectionResult)
T_State = TypeVar("T_State", bound=PlayerState)


@dataclass(frozen=True)
class PlayerBase(Generic[T_State, T_Result]):
    def select_first_action(self, state: State) -> int:
        return np.random.choice(get_valid_actions(state, player=1))

    def init_state(self, state: game.State, prev_state: T_State = None) -> tuple[T_State, list[list[int]]]:
        return T_State(), state.create_init_tokens()

    def select_next_action(self, state: State, player_state: T_State) -> T_Result:
        pass

    def apply_action(
        self,
        state: game.State,
        player_state: T_State,
        action: int, player: int,
        true_color_o: np.ndarray
    ) -> tuple[T_State, game.State, list[list[int]], game.StepResult]:

        if player == -1:
            action = 31 - action

        state, result = state.step(action, player)
        tokens = result.tokens

        for afterstate in result.afterstates:
            state, result = state.step_afterstate(afterstate, true_color_o[afterstate.piece_id])
            tokens += result.tokens

        return PlayerState(), state, tokens, result

    def visualize_state(self, player_state: T_State, output_path: str):
        pass


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


def play_game(
    player1: PlayerBase[T_State, T_Result],
    player2: PlayerBase[T_State, T_Result],
    player_state1: T_State = None,
    player_state2: T_State = None,
    visualization_directory: str = None,
    game_length=200,
    print_board=False
) -> GameResult:
    action_history = np.zeros(game_length + 20, dtype=np.int16)

    state1, state2 = get_initial_state_pair()

    player_state1, tokens1 = player1.init_state(state1, player_state1)
    player_state2, tokens2 = player2.init_state(state2, player_state2)

    player = 1

    for i in range(game_length):
        if player == 1:
            action = player1.select_next_action(state1, player_state1).action

            if visualization_directory is not None:
                player1.visualize_state(player_state1, f"{visualization_directory}/{i}")
        else:
            action = player2.select_next_action(state2, player_state2).action

            if visualization_directory is not None:
                player2.visualize_state(player_state2, f"{visualization_directory}/{i}")

        player_state1, state1, tokens1_i, result1 = player1.apply_action(
            state1, player_state1, action, player, state2.board[game.COL_P, ::-1]
        )
        player_state2, state2, tokens2_i, result2 = player2.apply_action(
            state2, player_state2, action, -player, state1.board[game.COL_P, ::-1]
        )

        tokens1 += tokens1_i
        tokens2 += tokens2_i

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

        if result1.winner != 0 or result2.winner != 0:
            break

        player = -player

    return GameResult(
        actions=action_history,
        winner=result1.winner,
        win_type=result1.win_type,
        color1=state1.board[game.COL_P],
        color2=state2.board[game.COL_P],
        tokens1=tokens1,
        tokens2=tokens2
    )
