from __future__ import annotations

from typing import Literal

from dataclasses import dataclass

import numpy as np

import env.state as game
from env.state import State, WinType, get_initial_state_pair, get_valid_actions
from game_analytics import states_to_str
from distributed.communication import SerdeJsonSerializable
from network.checkpoints import Checkpoint
import batch


@dataclass
class ActionSelectionResult:
    action: int


@dataclass
class PlayerState:
    pass


@dataclass(frozen=True)
class PlayerBase[T: ActionSelectionResult, S: PlayerState]:
    def select_first_action(self, state: State) -> int:
        return np.random.choice(get_valid_actions(state, player=1))

    def init_state(self, state: game.State, prev_state: S = None) -> tuple[game.State, S, list[list[int]]]:
        return state, PlayerState(), state.create_init_tokens()

    def select_next_action(self, state: State, player_state: S) -> T:
        pass

    def apply_action(
        self,
        state: game.State,
        player_state: S,
        action: int, player: int,
        true_color_o: np.ndarray
    ) -> tuple[S, game.State, list[list[int]], game.StepResult]:

        if player == -1:
            action = 31 - action

        state, result = state.step(action, player)
        tokens = result.tokens

        for afterstate in result.afterstates:
            state, result = state.step_afterstate(afterstate, true_color_o[afterstate.piece_id])
            tokens += result.tokens

        return player_state, state, tokens, result

    def apply_game_result(self, player_state: S, result: GameResult, player_id: int) -> S:
        pass

    def create_sample_p(self, player_state: S, result: GameResult, token_length: int) -> np.ndarray:
        return self.create_sample(
            player_state, result, player_id=0, token_length=token_length
        )

    def create_sample_o(self, player_state: S, result: GameResult, token_length: int) -> np.ndarray:
        return self.create_sample(
            player_state, result, player_id=1, token_length=token_length
        )

    def create_sample(
        self,
        player_state: S,
        result: GameResult,
        player_id: Literal[0, 1],
        token_length: int
    ) -> np.ndarray:
        opponent_id = 1 - player_id

        tokens = GameResult.adjust_tokens_length(
            result.tokens[player_id],
            token_length
        )
        actions = result.actions[tokens[:, 4]]
        color_o = result.states[opponent_id].col_p[::-1]

        reward_int = 3 + int(result.winner[player_id] * result.win_type.value)
        reward = np.array([reward_int], dtype=np.uint8)

        return self._create_sample(
            tokens.astype(np.uint8),
            actions.astype(np.uint8),
            reward.astype(np.uint8),
            color_o.astype(np.uint8),
            state=player_state
        )

    def _create_sample(
        self,
        x: np.ndarray, p: np.ndarray, v: np.ndarray, c: np.ndarray,
        state: S
    ) -> np.ndarray:
        return batch.FORMAT_XARC.from_tuple(x, p, v, c)

    def visualize_state(self, player_state: S, output_path: str):
        pass


@dataclass(frozen=True)
class PlayerConfig[T: PlayerBase](SerdeJsonSerializable):
    @property
    def necessary_checkpoint_step(self) -> int | None:
        return None

    @property
    def name(self) -> str:
        pass

    def create_player(self, project_dir: str) -> T:
        pass

    def get_checkpoint(self, project_dir: str) -> Checkpoint:
        return None


@dataclass(frozen=True)
class GameResult:
    actions: np.ndarray
    win_type: WinType
    winner: tuple[int, int]
    states: tuple[game.State, game.State]
    tokens: tuple[np.ndarray, np.ndarray]
    player_states: tuple[PlayerState, PlayerState]

    @staticmethod
    def adjust_tokens_length(tokens: np.ndarray, length: int) -> np.ndarray:
        if tokens.shape[0] == length:
            return tokens

        if tokens.shape[0] > length:
            return tokens[:length]

        if tokens.shape[0] < length:
            tokens_new = np.zeros((length, tokens.shape[-1]), dtype=tokens.dtype)
            tokens_new[:tokens.shape[0]] = tokens

            return tokens_new

    def invert(self) -> "GameResult":
        return GameResult(
            actions=self.actions,
            win_type=self.win_type,
            winner=self.winner[::-1],
            states=self.states[::-1],
            tokens=self.tokens[::-1],
            player_states=self.player_states[::-1],
        )


class TokenProducer:
    def __init__(self):
        self.tokens: np.ndarray = None
    
    @property
    def token_dtype(self):
        return np.uint8

    def init_game(self, game_length: int):
        self.tokens = np.zeros((2, game_length + 40, 5), dtype=self.token_dtype)

    def on_step(self, state: game.State, action: int, player: int):
        pass

    def add_tokens(self, state: game.State, tokens_in_step: list[list[int]], player_id: int):
        empty_mask = np.all(self.tokens[player_id] == 0, axis=-1)

        if not np.any(empty_mask):
            return

        idx = np.arange(len(empty_mask))[empty_mask].min()
        self.tokens[player_id, idx: idx + len(tokens_in_step)] = [token[:5] for token in tokens_in_step]


def play_game[T: ActionSelectionResult, S: PlayerState](
    player1: PlayerBase[T, S],
    player2: PlayerBase[T, S],
    player_state1: S = None,
    player_state2: S = None,
    color1: np.ndarray = None,
    color2: np.ndarray = None,
    token_producer: TokenProducer = TokenProducer(),
    visualization_directory: str = None,
    num_turns=200,
    print_board=False
) -> GameResult:
    action_history = np.zeros(num_turns + 20, dtype=np.int16)

    players = player1, player2
    player_states = [player_state1, player_state2]

    states = get_initial_state_pair(color1, color2)
    states = list(states)

    token_producer.init_game(num_turns)

    for i in range(2):
        states[i], player_states[i], init_tokens = players[i].init_state(states[i], player_states[i])
        token_producer.add_tokens(states[i], init_tokens, player_id=i)

    turn_player = 1

    for i in range(num_turns):
        p = 0 if turn_player == 1 else 1

        action = players[p].select_next_action(states[p], player_states[p]).action

        if visualization_directory is not None:
            players[p].visualize_state(player_states[p], f"{visualization_directory}/{i}")

        token_producer.on_step(states[p], action, turn_player)

        results: list[game.StepResult] = [None, None]

        for j in range(2):
            turn_player_j = turn_player * (1 if j == 0 else -1)
            col_o = states[1 - j].board[game.COL_P, ::-1]

            player_states[j], states[j], tokens_i, results[j] = players[j].apply_action(
                states[j], player_states[j], action, turn_player_j, col_o
            )

            token_producer.add_tokens(states[j], tokens_i, player_id=j)

        action_history[i] = action

        if print_board:
            s = states_to_str(
                states=states,
                predicted_colors=[[0.5]*8, [0.5]*8],
                true_colors=[None, None],
                colored=True
            )
            print(s)
            print(i)

        if results[0].winner != 0:
            winner = results[0].winner
            win_type = results[0].win_type
            break

        if results[1].winner != 0:
            winner = -results[1].winner
            win_type = results[1].win_type
            break

        turn_player = -turn_player
    else:
        winner = 0
        win_type = WinType.DRAW

    return GameResult(
        actions=action_history,
        win_type=win_type,
        winner=(winner, 1 - winner),
        states=tuple(states),
        tokens=(token_producer.tokens[0], token_producer.tokens[1]),
        player_states=tuple(player_states),
    )


def play_games[T: ActionSelectionResult, S: PlayerState](
    player1: PlayerBase[T, S],
    player2: PlayerBase[T, S],
    num_games: int,
    token_producer: TokenProducer = TokenProducer(),
    num_turns: int = 200,
    tokens_length: int = 240,
) -> GameResult:
    samples = []

    state1 = None
    state2 = None

    for i in range(num_games):
        if np.random.random() > 0.5:
            result = play_game(
                player1, player2,
                player_state1=state1, player_state2=state2,
                token_producer=token_producer,
                num_turns=num_turns
            )
        else:
            result = play_game(
                player2, player1,
                player_state1=state2, player_state2=state1,
                token_producer=token_producer,
                num_turns=num_turns
            )
            result = result.invert()

        state1 = player1.apply_game_result(result, player_id=0)
        state2 = player2.apply_game_result(result, player_id=1)

        sample = player1.create_sample_p(state1, result, tokens_length)

        samples.append(sample)
    
    return samples
