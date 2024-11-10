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


@dataclass
class PlayerConfig[T: PlayerBase](SerdeJsonSerializable):
    @property
    def name(self) -> str:
        pass

    def create_player(self, project_dir: str) -> T:
        pass

    def get_checkpoint(self, project_dir: str) -> Checkpoint:
        return None


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

        return batch.FORMAT_XARC.from_tuple(
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


def find_closest_pieses(pieses_pos: np.ndarray, target_pos: int) -> list[int]:
    mask = pieses_pos == game.CAPTURED

    x = (pieses_pos % 6) - (target_pos % 6)
    y = (pieses_pos // 6) - (target_pos // 6)

    d = np.abs(x) + np.abs(y)
    d[mask] = 100

    closest_mask = d == np.min(d)
    pieses_id = np.arange(len(pieses_pos))[closest_mask]

    return list(pieses_id)


def is_action_to_enter_deadlock(state: game.State, action: int, player: int) -> bool:
    if player == 1:
        pos_p, pos_o = state.board[game.POS_P], state.board[game.POS_O]
        escaping_pos = game.ESCAPE_POS_P
    else:
        pos_p, pos_o = state.board[game.POS_O], state.board[game.POS_P]
        escaping_pos = game.ESCAPE_POS_O

    defenders_0 = find_closest_pieses(pos_o, target_pos=escaping_pos[0])
    defenders_1 = find_closest_pieses(pos_o, target_pos=escaping_pos[1])
    defenders = defenders_0 + defenders_1

    defender_pos = pos_o[defenders]
    defender_pos = np.stack([defender_pos] * 4, axis=0)
    defender_pos[0] -= 6
    defender_pos[1] -= 1
    defender_pos[2] += 1
    defender_pos[3] += 6

    defender_pos[1, defender_pos[1] % 6 == 5] = -1
    defender_pos[2, defender_pos[2] % 6 == 0] = -1

    defender_pos = defender_pos.flatten()

    p_id, d = game.action_to_id(action)
    action_pos = pos_p[p_id] + d

    if action_pos not in defender_pos:
        return False

    return True


def play_game[T: ActionSelectionResult, S: PlayerState](
    player1: PlayerBase[T, S],
    player2: PlayerBase[T, S],
    player_state1: S = None,
    player_state2: S = None,
    visualization_directory: str = None,
    game_length=200,
    print_board=False
) -> GameResult:
    action_history = np.zeros(game_length + 20, dtype=np.int16)

    players = player1, player2
    player_states = [player_state1, player_state2]

    states = get_initial_state_pair()
    states = list(states)

    tokens: list[list[list[int]]] = [[], []]

    for i in range(2):
        player_states[i], tokens[i] = players[i].init_state(states[i], player_states[i])

    turn_player = 1

    attacked_count = np.zeros((2, 2))

    for i in range(game_length):
        p = 0 if turn_player == 1 else 1

        action = players[p].select_next_action(states[p], player_states[p]).action

        if visualization_directory is not None:
            players[p].visualize_state(player_states[p], f"{visualization_directory}/{i}")

        if is_action_to_enter_deadlock(states[p], action, 1):
            p_id, _ = game.action_to_id(action)
            color = states[p].board[game.COL_P, p_id]
            attacked_count[p, color] += 1
            print("t!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        results: list[game.StepResult] = [None, None]

        for j in range(2):
            player_states[j], states[j], tokens_i, result = players[j].apply_action(
                states[j], player_states[j], action, turn_player * (1 if j == 0 else -1), states[1 - j].board[game.COL_P, ::-1]
            )
            tokens[j] += tokens_i
            results[j] = result

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

        if results[0].winner != 0 or results[1].winner != 0:
            break

        turn_player = -turn_player
    
    print(attacked_count)

    return GameResult(
        actions=action_history,
        winner=results[0].winner,
        win_type=results[0].win_type,
        color1=states[0].board[game.COL_P],
        color2=states[1].board[game.COL_P],
        tokens1=tokens[0],
        tokens2=tokens[1]
    )
