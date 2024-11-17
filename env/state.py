from dataclasses import dataclass
from enum import Enum, IntEnum

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


@dataclass(frozen=True)
class Afterstate:
    type: AfterstateType
    piece_id: int
    player_id: int = 1


@dataclass(frozen=True)
class StepResult:
    tokens: list[list[int]]
    afterstates: list[Afterstate]
    winner: int
    win_type: WinType

    def is_captured(self) -> bool:
        return any([token[Token.X] == 6 for token in self.tokens])


POS_P = 0
POS_O = 1
COL_P = 2
COL_O = 3


@dataclass
class State:
    # shape: [4, 8]
    board: np.ndarray
    n_ply: int

    @property
    def pos_p(self) -> np.ndarray:
        return self.board[POS_P]

    @property
    def pos_o(self) -> np.ndarray:
        return self.board[POS_O]

    @property
    def col_p(self) -> np.ndarray:
        return self.board[COL_P]

    @property
    def col_o(self) -> np.ndarray:
        return self.board[COL_O]

    @staticmethod
    def create(color: np.ndarray) -> "State":
        pieces_p = np.array([1, 2, 3, 4, 7, 8, 9, 10], dtype=np.int16)
        pieces_o = np.array([25, 26, 27, 28, 31, 32, 33, 34], dtype=np.int16)

        color_p = color.astype(np.int16)
        color_o = np.array([UNCERTAIN_PIECE]*8, dtype=np.int16)

        board = np.stack([pieces_p, pieces_o, color_p, color_o])

        return State(board, 0)

    def create_init_tokens(self):
        return [[self.board[COL_P, i], i, self.board[POS_P, i] % 6, self.board[POS_P, i] // 6, 0] for i in range(8)]

    def num_captured_pieces(self, player: int, color: int):
        if player == 1:
            return np.sum(self.board[POS_P][self.board[COL_P] == color] == CAPTURED)
        else:
            return np.sum(self.board[POS_O][self.board[COL_O] == color] == CAPTURED)

    @staticmethod
    def is_done_caused_by_capturing(board: np.ndarray) -> tuple[int, WinType]:
        if 4 <= np.sum(board[POS_P][board[COL_P] == BLUE] == CAPTURED):
            return -1, WinType.BLUE_4

        if 4 <= np.sum(board[POS_P][board[COL_P] == RED] == CAPTURED):
            return 1, WinType.RED_4

        if 4 <= np.sum(board[POS_O][board[COL_O] == BLUE] == CAPTURED):
            return 1, WinType.BLUE_4

        if 4 <= np.sum(board[POS_O][board[COL_O] == RED] == CAPTURED):
            return -1, WinType.RED_4

        return 0, WinType.DRAW

    def step_afterstate(self, afterstate: Afterstate, color: int) -> tuple["State", StepResult]:
        next_board = self.board.copy()

        next_board[COL_P + afterstate.player_id, afterstate.piece_id] = color

        if afterstate.type == AfterstateType.CAPTURING:
            winner, win_type = State.is_done_caused_by_capturing(next_board)

            tokens = [(
                color + 2,
                afterstate.piece_id + 8 * afterstate.player_id,
                6, 6, self.n_ply
            )]

            return (
                State(next_board, self.n_ply),
                StepResult(tokens, [], winner, win_type)
            )

        elif afterstate.type == AfterstateType.ESCAPING:
            if color == BLUE:
                if afterstate.player_id == 1:
                    winner = -1
                else:
                    winner = 1
                win_type = WinType.ESCAPE
            else:
                winner = 0
                win_type = WinType.DRAW

            pos = next_board[POS_P + afterstate.player_id, afterstate.piece_id]

            tokens = [(
                4,
                afterstate.piece_id + 8 * afterstate.player_id,
                pos % 6,
                pos // 6,
                self.n_ply
            )]
            # tokens = []

            return (
                State(next_board, self.n_ply),
                StepResult(tokens, [], winner, win_type)
            )

    def step(self, action: int, player: int) -> tuple["State", StepResult]:
        if player == 1:
            return self.step_p(action)
        else:
            return self.step_o(action)

    def step_p(self, action: int) -> tuple["State", StepResult]:
        next_board = self.board.copy()

        p_id, d = action_to_id(action)

        pos = next_board[POS_P, p_id]
        pos_next = pos + d

        p_cap_id = np.where(next_board[POS_O] == pos_next)[0]

        afterstates = []

        tokens = [[
            next_board[COL_P, p_id],
            p_id,
            pos_next % 6,
            pos_next // 6,
            self.n_ply + 1]]

        if len(p_cap_id) > 0:
            p_cap_id = p_cap_id[0]
            next_board[POS_O, p_cap_id] = CAPTURED
            color = next_board[COL_O, p_cap_id]

            if color == UNCERTAIN_PIECE:
                afterstates.append(Afterstate(AfterstateType.CAPTURING, p_cap_id))

            elif color == RED or color == BLUE:
                tokens.append([
                    color + 2, p_cap_id + 8,
                    6, 6, self.n_ply + 1])
            else:
                RuntimeError(f"Unknown color: {self.board}")

        next_board[POS_P, p_id] = pos_next

        winner, win_type = State.is_done_caused_by_capturing(next_board)

        if winner == 0:
            escaped = (next_board[POS_O] == ESCAPE_POS_O[0]) | (next_board[POS_O] == ESCAPE_POS_O[1])
            escaped_u = escaped & (next_board[COL_O] == UNCERTAIN_PIECE)

            if np.any(escaped_u):
                escaped_u_id = np.where(escaped_u)[0][0]
                afterstates.append(Afterstate(AfterstateType.ESCAPING, escaped_u_id))

            escaped_b = escaped & (next_board[COL_O] == BLUE)

            if np.any(escaped_b):
                winner = -1
                win_type = WinType.ESCAPE

        return (
            State(next_board, self.n_ply + 1),
            StepResult(tokens, afterstates, winner, win_type)
        )

    def step_o(self, action: int) -> tuple["State", StepResult]:
        next_board = self.board.copy()

        p_id, d = action_to_id(action)

        pos = next_board[POS_O, p_id]
        pos_next = pos + d

        p_cap_id = np.where(next_board[POS_P] == pos_next)[0]

        afterstates = []

        tokens = [[
            4, p_id + 8,
            pos_next % 6,
            pos_next // 6,
            self.n_ply + 1]]

        if len(p_cap_id) > 0:
            p_cap_id = p_cap_id[0]
            next_board[POS_P, p_cap_id] = CAPTURED
            color = next_board[COL_P, p_cap_id]

            if color == UNCERTAIN_PIECE:
                afterstates.append(Afterstate(AfterstateType.CAPTURING, p_cap_id, player_id=0))
            elif color == RED or color == BLUE:
                tokens.append([color, p_cap_id, 6, 6, self.n_ply + 1])
            else:
                RuntimeError(f"Unknown color: {self.board}")

        next_board[POS_O, p_id] = pos_next

        winner, win_type = State.is_done_caused_by_capturing(next_board)

        if winner != 0:
            escaped = (next_board[POS_P] == ESCAPE_POS_P[0]) | (next_board[POS_P] == ESCAPE_POS_P[1])
            escaped_u = escaped & (next_board[COL_P] == UNCERTAIN_PIECE)

            if np.any(escaped_u):
                escaped_u_id = np.where(escaped_u)[0][0]
                afterstates.append(Afterstate(AfterstateType.ESCAPING, escaped_u_id, player_id=0))

            escaped_b = escaped & (next_board[COL_O] == BLUE)

            if np.any(escaped_b):
                winner = 1
                win_type = WinType.ESCAPE

        return (
            State(next_board, self.n_ply + 1),
            StepResult(tokens, afterstates, winner, win_type)
        )


def create_random_color() -> np.ndarray:
    blue = np.random.choice(np.arange(8), 4, replace=False)
    color = np.zeros(8, dtype=np.int16)
    color[blue] = BLUE

    return color


def get_initial_state_pair(color_p: np.ndarray = None, color_o: np.ndarray = None) -> tuple[State, State]:
    if color_p is None:
        color_p = create_random_color()
    
    if color_o is None:
        color_o = create_random_color()

    state_p = State.create(color_p)
    state_o = State.create(color_o)

    return state_p, state_o


def action_to_id(action: int) -> tuple[int, int]:
    p_id = action // 4
    d_i = action % 4
    d = DIRECTIONS[d_i]

    return p_id, d


def get_valid_actions_mask(state: State, player: int):
    if player == 1:
        pieces = state.board[POS_P]
    else:
        pieces = state.board[POS_O]

    pos = np.stack([pieces]*4, axis=1)

    pos[pos == CAPTURED] = -1000
    pos[pos[:, 1] % 6 == 0, 1] = -1000
    pos[pos[:, 2] % 6 == 5, 2] = -1000

    next_pos = (pos + DIRECTIONS).flatten()

    within_board = (0 <= next_pos) & (next_pos < 36)

    is_not_player_piece = pieces != np.stack([next_pos]*len(pieces), axis=1)
    is_not_player_piece = np.all(is_not_player_piece, axis=1)

    return within_board & is_not_player_piece


def get_valid_actions(state: State, player: int):
    mask = get_valid_actions_mask(state, player)
    moves = np.where(mask)[0]

    return moves
