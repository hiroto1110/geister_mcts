import time
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import List, Tuple

import numpy as np


N_COLS = N_ROWS = 6
DIRECTIONS = -6, -1, 1, 6

ACTION_SPACE = 8 * 4

UNCERTAIN_PIECE = 2
BLUE = 1
RED = 0

ESCAPE_POS_P = 30, 35
ESCAPE_POS_O = 0, 5

CAPTURED = -1

TOKEN_SIZE = 5
MAX_TOKEN_LENGTH = 200


class Token(IntEnum):
    COLOR = 0
    ID = 1
    X = 2
    Y = 3
    T = 4


class WinType(Enum):
    DRAW = 0
    ESCAPE = 1
    BLUE_4 = 2
    RED_4 = 3


class AfterstateType(Enum):
    NONE = -1
    ESCAPING = 0
    CAPTURING = 1


@dataclass
class Afterstate:
    type: AfterstateType
    piece_id: int


class SimulationState:
    def __init__(self, color, root_player: int):
        self.root_player = root_player

        self.pieces_p = np.array([1, 2, 3, 4, 7, 8, 9, 10], dtype=np.int16)
        self.pieces_o = np.array([25, 26, 27, 28, 31, 32, 33, 34], dtype=np.int16)

        self.color_p = color
        self.color_o = np.array([UNCERTAIN_PIECE]*8, dtype=np.int16)

        self.escape_pos_p = ESCAPE_POS_P
        self.escape_pos_o = ESCAPE_POS_O

        if root_player == -1:
            self.pieces_p, self.pieces_o = self.pieces_o, self.pieces_p
            self.escape_pos_p, self.escape_pos_o = self.escape_pos_o, self.escape_pos_p

        self.is_done = False
        self.winner = 0
        self.win_type = WinType.DRAW
        self.n_ply = 0

    def create_init_tokens(self):
        return [[self.color_p[i], i, self.pieces_p[i] % 6, self.pieces_p[i] // 6, 0] for i in range(8)]

    def step_afterstate(self, afterstate: Afterstate, color: int) -> List[List[int]]:
        self.color_o[afterstate.piece_id] = color

        if afterstate.type == AfterstateType.CAPTURING:
            self.update_is_done_caused_by_capturing()

            return [(
                color + 2,
                afterstate.piece_id + 8,
                6, 6, self.n_ply
            )]

        elif afterstate.type == AfterstateType.ESCAPING:
            if color == BLUE:
                self.is_done = True
                self.winner = -1
                self.win_type = WinType.ESCAPE

            pos = self.pieces_o[afterstate.piece_id]

            return [(
                color + 2,
                afterstate.piece_id + 8,
                pos % 6,
                pos // 6,
                self.n_ply
            )]

    def undo_step_afterstate(self, afterstate: Afterstate):
        self.color_o[afterstate.piece_id] = UNCERTAIN_PIECE

        self.is_done = False
        self.win_type = WinType.DRAW
        self.winner = 0

    def step(self, action: int, player: int) -> Tuple[List[List[int]], List[Afterstate]]:
        if player == 1:
            return self.step_p(action)
        else:
            return self.step_o(action)

    def step_p(self, action: int) -> Tuple[List[List[int]], List[Afterstate]]:
        self.n_ply += 1

        p_id, d = action_to_id(action)

        pos = self.pieces_p[p_id]
        pos_next = pos + d

        p_cap_id = np.where(self.pieces_o == pos_next)[0]

        afterstates = []

        tokens = [[
            self.color_p[p_id],
            p_id,
            pos_next % 6,
            pos_next // 6,
            self.n_ply]]

        if len(p_cap_id) > 0:
            p_cap_id = p_cap_id[0]
            self.pieces_o[p_cap_id] = CAPTURED
            color = self.color_o[p_cap_id]

            if color == UNCERTAIN_PIECE:
                afterstates.append(Afterstate(AfterstateType.CAPTURING, p_cap_id))

            elif color == RED or color == BLUE:
                tokens.append([
                    color + 2, p_cap_id + 8,
                    6, 6, self.n_ply])

        self.pieces_p[p_id] = pos_next

        self.update_is_done_caused_by_capturing()

        escaped = (self.pieces_o == self.escape_pos_o[0]) | (self.pieces_o == self.escape_pos_o[1])
        escaped_u = escaped & (self.color_o == UNCERTAIN_PIECE)

        if np.any(escaped_u):
            escaped_u_id = np.where(escaped_u)[0][0]
            afterstates.append(Afterstate(AfterstateType.ESCAPING, escaped_u_id))

        escaped_b = escaped & (self.color_o == BLUE)
        if np.any(escaped_b):
            self.is_done = True
            self.winner = -1
            self.win_type = WinType.ESCAPE

        return tokens, afterstates

    def step_o(self, action: int) -> Tuple[List[List[int]], List[Afterstate]]:
        self.n_ply += 1

        p_id, d = action_to_id(action)

        pos = self.pieces_o[p_id]
        pos_next = pos + d

        p_cap_id = np.where(self.pieces_p == pos_next)[0]

        tokens = [[
            4, p_id + 8,
            pos_next % 6,
            pos_next // 6,
            self.n_ply]]

        if len(p_cap_id) > 0:
            p_cap_id = p_cap_id[0]
            self.pieces_p[p_cap_id] = CAPTURED

            tokens.append([
                self.color_p[p_cap_id], p_cap_id,
                6, 6, self.n_ply])

        self.pieces_o[p_id] = pos_next

        self.update_is_done_caused_by_capturing()

        escaped = (self.pieces_p == self.escape_pos_p[0]) | (self.pieces_p == self.escape_pos_p[1])
        escaped = escaped & (self.color_p == BLUE)

        if np.any(escaped):
            self.is_done = True
            self.winner = 1
            self.win_type = WinType.ESCAPE

        return tokens, []

    def update_is_done_caused_by_capturing(self):
        if 4 <= np.sum(self.pieces_p[self.color_p == BLUE] == CAPTURED):
            self.is_done = True
            self.win_type = WinType.BLUE_4
            self.winner = -1
            return

        if 4 <= np.sum(self.pieces_p[self.color_p == RED] == CAPTURED):
            self.is_done = True
            self.win_type = WinType.RED_4
            self.winner = 1
            return

        if 4 <= np.sum(self.pieces_o[self.color_o == BLUE] == CAPTURED):
            self.is_done = True
            self.win_type = WinType.BLUE_4
            self.winner = 1
            return

        if 4 <= np.sum(self.pieces_o[self.color_o == RED] == CAPTURED):
            self.is_done = True
            self.win_type = WinType.RED_4
            self.winner = -1
            return

        self.is_done = False
        self.win_type = WinType.DRAW
        self.winner = 0

    def undo_step(self, action: int, player: int,
                  tokens: List[List[int]],
                  info: Afterstate):

        if player == 1:
            self.undo_step_p(action, tokens, info)
        else:
            self.undo_step_o(action, tokens)

    def undo_step_p(self, action: int, tokens: List[List[int]], info: List[Afterstate]):
        p_id, d = action_to_id(action)

        pos_next = self.pieces_p[p_id]
        self.pieces_p[p_id] = pos_next - d

        if len(info) > 0 and info[0].type == AfterstateType.CAPTURING:
            self.pieces_o[info[0].piece_id] = pos_next

        if len(tokens) == 2:
            p_cap_id = tokens[1][Token.ID] - 8
            self.pieces_o[p_cap_id] = pos_next

        self.n_ply -= 1
        self.is_done = False
        self.winner = 0

    def undo_step_o(self, action: int, tokens: List[List[int]]):
        p_id, d = action_to_id(action)

        pos_next = self.pieces_o[p_id]
        self.pieces_o[p_id] = pos_next - d

        if len(tokens) == 2:
            p_cap_id = tokens[1][Token.ID]
            self.pieces_p[p_cap_id] = pos_next

        self.n_ply -= 1
        self.is_done = False
        self.winner = 0


def create_random_color() -> np.ndarray:
    blue = np.random.choice(np.arange(8), 4, replace=False)
    color = np.zeros(8, dtype=np.int16)
    color[blue] = BLUE

    return color


def get_initial_state_pair() -> Tuple[SimulationState, SimulationState]:
    color_p = create_random_color()
    color_o = create_random_color()

    state_p = SimulationState(color_p, 1)
    state_o = SimulationState(color_o, -1)

    return state_p, state_o


def action_to_id(action):
    p_id = action // 4
    d_i = action % 4
    d = DIRECTIONS[d_i]

    return p_id, d


def action_to_pos(action):
    pos = action // 4
    d_i = action % 4
    d = DIRECTIONS[d_i]

    return pos, pos + d


def get_valid_actions_mask(state: SimulationState, player: int):
    if player == 1:
        pieces = state.pieces_p
    else:
        pieces = state.pieces_o

    pos = np.stack([pieces]*4, axis=1)

    pos[pos == CAPTURED] = -1000
    pos[pos[:, 1] % 6 == 0, 1] = -1000
    pos[pos[:, 2] % 6 == 5, 2] = -1000

    next_pos = (pos + DIRECTIONS).flatten()

    within_board = (0 <= next_pos) & (next_pos < 36)

    is_not_player_piece = pieces != np.stack([next_pos]*len(pieces), axis=1)
    is_not_player_piece = np.all(is_not_player_piece, axis=1)

    return within_board & is_not_player_piece


def get_valid_actions(state: SimulationState, player: int):
    mask = get_valid_actions_mask(state, player)
    moves = np.where(mask)[0]

    return moves


def test_moves():
    state1, state2 = get_initial_state_pair()

    player = 1

    while not state1.is_done:
        moves = get_valid_actions(state1, player)

        move = np.random.choice(moves)

        state1.step(move, player)
        state2.step(move, -player)

        board = np.zeros(36, dtype=np.int8)

        board[state1.pieces_p[(state1.pieces_p >= 0) & (state1.color_p == 1)]] = 1
        board[state1.pieces_p[(state1.pieces_p >= 0) & (state1.color_p == 0)]] = 2
        board[state2.pieces_p[(state2.pieces_p >= 0) & (state2.color_p == 1)]] = 1
        board[state2.pieces_p[(state2.pieces_p >= 0) & (state2.color_p == 0)]] = 2

        print(str(board.reshape((6, 6))).replace('0', ' '))
        print(state1.n_ply)

        player = -player

    print(f"winner: {state1.winner}")


if __name__ == "__main__":
    np.random.seed(12)

    start = time.perf_counter()
    for i in range(1):
        test_moves()
    end = time.perf_counter()

    print(f"time: {end - start}")
