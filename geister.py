import random
import time
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np


N_COLS = N_ROWS = 6
DIRECTIONS = -6, -1, 1, 6

ACTION_SPACE = N_ROWS * N_COLS * len(DIRECTIONS)

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


class State:
    def __init__(self, color_p, color_o):
        self.pieces_p = np.array([1, 2, 3, 4, 7, 8, 9, 10], dtype=np.int16)
        self.pieces_o = np.array([25, 26, 27, 28, 31, 32, 33, 34], dtype=np.int16)

        self.color_p = color_p
        self.color_o = color_o

        self.tokens_p = []
        self.tokens_o = []

        for p_id in range(8):
            pos = self.pieces_p[p_id]
            self.tokens_p.append([self.color_p[p_id], p_id, pos % 6, pos // 6, 0])

            pos = self.pieces_o[p_id]
            self.tokens_o.append([self.color_o[p_id], p_id, pos % 6, pos // 6, 0])

        self.is_done = False
        self.winner = 0
        self.win_type = WinType.DRAW
        self.n_ply = 0

    def undo_step(self, action: int, player: int):
        pieces_p = self.pieces_p
        pieces_o = self.pieces_o
        tokens_p = self.tokens_p
        tokens_o = self.tokens_o

        if player == -1:
            pieces_p, pieces_o = pieces_o, pieces_p
            tokens_p, tokens_o = tokens_o, tokens_p

        pos, pos_next = action_to_pos(action)

        p_id = np.where(pieces_p == pos_next)[0][0]

        pieces_p[p_id] = pos

        if self.tokens_p[-1][4] == self.tokens_p[-2][4]:
            p_cap_id = tokens_o[-1][1]
            pieces_o[p_cap_id] = pos_next

            del self.tokens_p[-1]
            del self.tokens_o[-1]

        del self.tokens_p[-1]
        del self.tokens_o[-1]

        self.n_ply -= 1

        self.is_done = False
        self.winner = 0

        if self.is_done:
            self.update_is_done(player)

    def step(self, action: int, player: int, is_sim: bool = False):
        self.n_ply += 1

        pieces_p = self.pieces_p
        pieces_o = self.pieces_o
        color_p = self.color_p
        color_o = self.color_o
        tokens_p = self.tokens_p
        tokens_o = self.tokens_o

        if player == -1:
            pieces_p, pieces_o = pieces_o, pieces_p
            color_p, color_o = color_o, color_p
            tokens_p, tokens_o = tokens_o, tokens_p

        pos, pos_next = action_to_pos(action)

        p_id = np.where(pieces_p == pos)[0][0]
        p_cap_id = np.where(pieces_o == pos_next)[0]

        tokens_p.append([
            color_p[p_id],
            p_id,
            pos_next % 6,
            pos_next // 6,
            self.n_ply])

        tokens_o.append([
            4,
            p_id + 8,
            pos_next % 6,
            pos_next // 6,
            self.n_ply])

        if len(p_cap_id) > 0:
            p_cap_id = p_cap_id[0]
            pieces_o[p_cap_id] = CAPTURED

            tokens_p.append([
                color_o[p_cap_id] + 2,
                p_cap_id + 8,
                6, 6, self.n_ply])

            tokens_o.append([
                color_o[p_cap_id],
                p_cap_id,
                6, 6, self.n_ply])

        pieces_p[p_id] = pos_next

        self.update_is_done(player)

    def update_is_done(self, player):
        if self.n_ply > 200:
            self.is_done = True
            self.winner = 0
            return

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

        if player == -1:
            pieces = self.pieces_p
            color = self.color_p
            escape_pos = ESCAPE_POS_P
        else:
            pieces = self.pieces_o
            color = self.color_o
            escape_pos = ESCAPE_POS_O

        escaped = (color == BLUE) & ((pieces == escape_pos[0]) | (pieces == escape_pos[1]))

        if np.any(escaped):
            self.is_done = True
            self.win_type = WinType.ESCAPE
            self.winner = -player
            return

        self.is_done = False
        self.win_type = WinType.DRAW
        self.winner = 0

    def get_tokens(self, player: int):
        if player == 1:
            tokens = self.tokens_p
        else:
            tokens = self.tokens_o

        tokens = np.array(tokens[:MAX_TOKEN_LENGTH], dtype=np.uint8)
        tokens = np.resize(tokens, (MAX_TOKEN_LENGTH, TOKEN_SIZE))
        return tokens

    def get_last_tokens(self, player: int):
        if player == 1:
            tokens = self.tokens_p
        else:
            tokens = self.tokens_o

        if tokens[-1][4] == tokens[-2][4]:
            return tokens[-2:]
        else:
            return tokens[-1:]


class AfterstateType(Enum):
    NONE = -1
    ESCAPING = 0
    CAPTURING = 1


@dataclass
class AfterstateInfo:
    type: AfterstateType
    token_id: int
    piece_id: int

    def is_afterstate(self):
        return self.type != AfterstateType.NONE


ASTERSTATE_INFO_NONE = AfterstateInfo(AfterstateType.NONE, -1, -1)


class SimulationState:
    def __init__(self, state: State, root_player: int):
        if root_player == 1:
            self.pieces_p = state.pieces_p
            self.color_p = state.color_p
            self.tokens_p = state.tokens_p

            self.pieces_o = state.pieces_o
            self.color_o = state.color_o

            self.escape_pos_p = ESCAPE_POS_P
            self.escape_pos_o = ESCAPE_POS_O
        else:
            self.pieces_p = state.pieces_o
            self.color_p = state.color_o
            self.tokens_p = state.tokens_o

            self.pieces_o = state.pieces_p
            self.color_o = state.color_p

            self.escape_pos_p = ESCAPE_POS_O
            self.escape_pos_o = ESCAPE_POS_P

        self.color_o = np.copy(self.color_o)
        self.color_o[self.pieces_o >= 0] = UNCERTAIN_PIECE

        self.is_done = state.is_done
        self.winner = state.winner
        self.win_type = state.win_type
        self.n_ply = state.n_ply

    def step_afterstate(self, info: AfterstateInfo, color: int):
        self.color_o[info.piece_id] = color
        self.tokens_p[info.token_id][Token.COLOR] = color + 2

        if info.type == AfterstateType.CAPTURING:
            self.update_is_done_caused_by_capturing()

        elif info.type == AfterstateType.ESCAPING:
            if color == BLUE:
                self.is_done = True
                self.winner = -1
                self.win_type = WinType.ESCAPE
        else:
            print("This isn't a afterstate")

    def undo_step_afterstate(self, info: AfterstateInfo):
        self.color_o[info.piece_id] = UNCERTAIN_PIECE
        self.tokens_p[info.token_id][Token.COLOR] = 4

    def undo_step(self, action: int, player: int):
        if player == 1:
            self.undo_step_p(action)
        else:
            self.undo_step_o(action)

    def undo_step_p(self, action: int):
        pos, pos_next = action_to_pos(action)

        p_id = np.where(self.pieces_p == pos_next)[0][0]

        self.pieces_p[p_id] = pos

        if self.tokens_p[-1][Token.T] == self.tokens_p[-2][Token.T]:
            p_cap_id = self.tokens_p[-1][Token.ID] - 8
            self.pieces_o[p_cap_id] = pos_next

            del self.tokens_p[-1]

        del self.tokens_p[-1]

        self.n_ply -= 1
        self.is_done = False
        self.winner = 0

    def undo_step_o(self, action: int):
        pos, pos_next = action_to_pos(action)

        p_id = np.where(self.pieces_o == pos_next)[0][0]
        self.pieces_o[p_id] = pos

        if self.tokens_p[-1][Token.T] == self.tokens_p[-2][Token.T]:
            p_cap_id = self.tokens_p[-1][Token.ID]
            self.pieces_p[p_cap_id] = pos_next

            del self.tokens_p[-1]

        del self.tokens_p[-1]

        self.n_ply -= 1
        self.is_done = False
        self.winner = 0

    def step(self, action: int, player: int) -> AfterstateInfo:
        if player == 1:
            return self.step_p(action)
        else:
            self.step_o(action)
            return ASTERSTATE_INFO_NONE

    def step_p(self, action: int) -> AfterstateInfo:
        self.n_ply += 1

        pos, pos_next = action_to_pos(action)

        p_id = np.where(self.pieces_p == pos)[0][0]
        p_cap_id = np.where(self.pieces_o == pos_next)[0]

        info = ASTERSTATE_INFO_NONE

        self.tokens_p.append([
            self.color_p[p_id],
            p_id,
            pos_next % 6,
            pos_next // 6,
            self.n_ply])

        if len(p_cap_id) > 0:
            p_cap_id = p_cap_id[0]
            self.pieces_o[p_cap_id] = CAPTURED
            color = self.color_o[p_cap_id]

            self.tokens_p.append([
                color + 2,
                p_cap_id + 8,
                6, 6, self.n_ply])

            if color == UNCERTAIN_PIECE:
                info = AfterstateInfo(AfterstateType.CAPTURING,
                                      token_id=len(self.tokens_p) - 1,
                                      piece_id=p_cap_id)

        self.pieces_p[p_id] = pos_next

        self.update_is_done_caused_by_capturing()

        escaped = (self.pieces_o == self.escape_pos_o[0]) | (self.pieces_o == self.escape_pos_o[1])
        escaped = escaped & (self.color_o == UNCERTAIN_PIECE)
        escaped_id = np.where(escaped)[0]

        if len(escaped_id) > 0:
            escaped_id = escaped_id[0]

            info = AfterstateInfo(AfterstateType.ESCAPING,
                                  token_id=len(self.tokens_p) - 2,
                                  piece_id=escaped_id)

        return info

    def step_o(self, action: int):
        self.n_ply += 1

        pos, pos_next = action_to_pos(action)

        p_id = np.where(self.pieces_o == pos)[0][0]
        p_cap_id = np.where(self.pieces_p == pos_next)[0]

        self.tokens_p.append([
            4, p_id + 8,
            pos_next % 6,
            pos_next // 6,
            self.n_ply])

        if len(p_cap_id) > 0:
            p_cap_id = p_cap_id[0]
            self.pieces_p[p_cap_id] = CAPTURED

            self.tokens_p.append([
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

    def get_afterstate_tokens(self, info: AfterstateInfo):
        i = info.token_id
        return [self.tokens_p[i: i+1]]

    def get_last_tokens(self):
        return [self.tokens_p[-1:]]


def get_initial_state():
    blue_p = np.random.choice(np.arange(8), 4, replace=False)
    blue_o = np.random.choice(np.arange(8), 4, replace=False)

    color_p = np.zeros(8, dtype=np.int16)
    color_p[blue_p] = 1

    color_o = np.zeros(8, dtype=np.int16)
    color_o[blue_o] = 1

    return State(color_p, color_o)


def is_valid_action(state: State, action: int, player: int):
    xy = action % (N_ROWS * N_COLS)
    d = action // (N_ROWS * N_COLS)

    if player == 1 and not np.any(state.pieces_p == xy):
        return False

    elif player == -1 and not np.any(state.pieces_o == xy):
        return False

    xy_next = xy + DIRECTIONS[d]

    if xy_next < 0 or 36 <= xy_next or state.board[xy_next] * player > 0:
        return False

    return True


def action_to_pos(action):
    pos = action // 4
    d_i = action % 4
    d = DIRECTIONS[d_i]

    return pos, pos + d


def get_valid_actions(state: State, player: int):
    if player == 1:
        pieces = state.pieces_p
    else:
        pieces = state.pieces_o

    pieces = pieces[pieces >= 0]
    pos = np.stack([pieces]*4, axis=1)

    pos[pos[:, 1] % 6 == 0, 1] = -1000
    pos[pos[:, 2] % 6 == 5, 2] = -1000

    next_pos = (pos + DIRECTIONS).flatten()

    within_board = (0 <= next_pos) & (next_pos < 36)

    is_not_player_piece = pieces != np.stack([next_pos]*len(pieces), axis=1)
    is_not_player_piece = np.all(is_not_player_piece, axis=1)

    moves = pos * 4 + [0, 1, 2, 3]
    moves = moves.flatten()
    moves = moves[within_board & is_not_player_piece]

    return moves


def get_result(state: State, player: int):
    return state.winner * player


def is_done(state: State, player):
    return state.is_done


def test_performance():
    np.random.seed(12)

    state = get_initial_state()
    player = 1

    move_history = []

    start = time.perf_counter()

    while not state.is_done:
        moves = get_valid_actions(state, player)

        move = np.random.choice(moves)
        move_history.append(move)

        state.step(move, player)

        player = -player

    print(f"n_ply: {state.n_ply}")
    print(f"winner: {state.winner}")

    """for move in move_history[::-1]:
        player = -player
        state.undo_step(move, player)"""

    print(time.perf_counter() - start)


def test_moves():
    state = get_initial_state()

    player = 1
    done = False

    move_history = []

    while not done:
        moves = get_valid_actions(state, player)

        move = np.random.choice(moves)
        move_history.append(move)

        state.step(move, player)
        done = state.is_done

        """board = np.zeros(36, dtype=np.int8)

        board[state.pieces_p[(state.pieces_p >= 0) & (state.color_p == 1)]] = 1
        board[state.pieces_p[(state.pieces_p >= 0) & (state.color_p == 0)]] = 2
        board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == 1)]] = -1
        board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == 0)]] = -2

        print(str(board.reshape((6, 6))).replace('0', ' '))"""
        print(state.n_ply)

        player = -player

    print(f"winner: {state.winner}")

    for move in move_history[::-1]:
        player = -player
        state.undo_step(move, player)

        """board = np.zeros(36, dtype=np.int8)

        board[state.pieces_p[(state.pieces_p >= 0) & (state.color_p == 1)]] = 1
        board[state.pieces_p[(state.pieces_p >= 0) & (state.color_p == 0)]] = 2
        board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == 1)]] = -1
        board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == 0)]] = -2

        print(str(board.reshape((6, 6))).replace('0', ' '))"""
        print(state.n_ply)

    tokens = state.get_tokens(1)
    print(tokens)


if __name__ == "__main__":
    np.random.seed(12)

    start = time.perf_counter()
    for i in range(10):
        test_moves()
    end = time.perf_counter()

    print(f"time: {end - start}")


def evaluate_greedy(state: State, player: int):
    e = 0
    e += 100 * state.winner * player
    e += sum(state.captured_p) if player == 1 else sum(state.captured_o)
    return e


def greedy_action(state: State, player: int, epsilon=0.):
    valid_actions = get_valid_actions(state, player)

    if random.random() > epsilon:
        best_action = None
        best_score = -100000000

        for action in valid_actions:
            state.step(action, player)
            score = evaluate_greedy(state, player)
            state.undo_step(action, player)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    else:
        action = random.choice(valid_actions)
        return action
