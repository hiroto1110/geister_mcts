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

    def init_state(self, state: game.State, prev_state: S = None) -> tuple[S, list[list[int]]]:
        return PlayerState(), state.create_init_tokens()

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

    def visualize_state(self, player_state: S, output_path: str):
        pass


@dataclass(frozen=True)
class PlayerConfig[T: PlayerBase](SerdeJsonSerializable):
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
    winner: int
    win_type: WinType
    color1: np.ndarray
    color2: np.ndarray
    tokens1: np.ndarray
    tokens2: np.ndarray

    @classmethod
    def get_batch_format(cls) -> batch.BatchFormat:
        return batch.FORMAT_XARC

    @classmethod
    def create_sample(
        cls,
        tokens_list: np.ndarray,
        actions: np.ndarray,
        color_o: np.ndarray,
        reward: int,
        token_length: int
    ) -> np.ndarray:
        tokens = np.zeros((token_length, tokens_list.shape[-1]), dtype=np.uint8)
        tokens[:min(token_length, len(tokens_list))] = tokens_list[:token_length]

        actions = actions[tokens[:, 4]]
        reward = np.array([reward], dtype=np.uint8)

        return cls.get_batch_format().from_tuple(
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

class TokenProducer:
    def __init__(self):
        self.tokens: np.ndarray = None

    def init_game(self, game_length: int):
        self.tokens = np.zeros((2, game_length + 40, 5), dtype=np.uint8)

    def on_step(self, state: game.State, action: int, player: int):
        pass

    def add_tokens(self, state: game.State, tokens_in_step: list[list[int]], player_id: int):
        empty_mask = np.all(self.tokens[player_id] == 0, axis=-1)

        if not np.any(empty_mask):
            return

        idx = np.arange(len(empty_mask))[empty_mask].min()
        self.tokens[player_id, idx: idx + len(tokens_in_step)] = tokens_in_step


def play_game[T: ActionSelectionResult, S: PlayerState](
    player1: PlayerBase[T, S],
    player2: PlayerBase[T, S],
    player_state1: S = None,
    player_state2: S = None,
    color1: np.ndarray = None,
    color2: np.ndarray = None,
    token_producer: TokenProducer = TokenProducer(),
    visualization_directory: str = None,
    game_length=200,
    print_board=False
) -> GameResult:
    action_history = np.zeros(game_length + 20, dtype=np.int16)

    players = player1, player2
    player_states = [player_state1, player_state2]

    states = get_initial_state_pair(color1, color2)
    states = list(states)

    token_producer.init_game(game_length)

    for i in range(2):
        player_states[i], init_tokens = players[i].init_state(states[i], player_states[i])
        token_producer.add_tokens(states[i], init_tokens, player_id=i)

    turn_player = 1

    for i in range(game_length):
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
        winner=winner,
        win_type=win_type,
        color1=states[0].col_p,
        color2=states[1].col_p,
        tokens1=token_producer.tokens[0],
        tokens2=token_producer.tokens[1]
    )
