import numpy as np
import geister as game

COLOR_NAMES = 'rbu'
CAPTURED_COLOR_NAMES = 'RB'
COLOR_DICT = {'r': game.RED, 'b': game.BLUE, 'u': game.UNCERTAIN_PIECE}
PIECE_NAMES = 'ABCDEFGH'
PIECE_DICT = {c: i for i, c in enumerate(PIECE_NAMES)}
DIRECTION_NAMES = 'NWES'
DIRECTION_DICT = {c: i for i, c in enumerate(DIRECTION_NAMES)}


def encode_set_message(color: np.ndarray, player: int) -> str:
    msg = ''

    if player == 1:
        color = color[::-1]

    for i in range(8):
        if color[i] == game.RED:
            msg += PIECE_NAMES[i]

    return f'SET:{msg}\r\n'


def decode_set_message(msg: str):
    color = np.full(shape=8, fill_value=game.BLUE)

    for i in range(4):
        c = msg[4 + i]
        color[COLOR_DICT[c]] = game.RED

    return color


def format_action_message(action: int, player: int) -> str:
    if player == 1:
        action = 31 - action

    p_id = action // 4
    d_id = action % 4

    p_name = PIECE_NAMES[p_id]
    d_name = DIRECTION_NAMES[d_id]

    return f'MOV:{p_name},{d_name}\r\n'


def decode_action_message(msg: str) -> int:
    p_name = msg[4]
    d_name = msg[6]

    p_id = PIECE_DICT[p_name]
    d_id = DIRECTION_DICT[d_name]

    return p_id * 4 + d_id


def parse_action_ack(s: str):
    if s[2] == 'R':
        return game.RED

    if s[2] == 'B':
        return game.BLUE

    return game.UNCERTAIN_PIECE


def is_done_message(s: str):
    if s.startswith('WON'):
        return True, 1

    if s.startswith('LST'):
        return True, -1

    if s.startswith('DRW'):
        return True, 0

    return False, 0


def encode_board_str(state: game.SimulationState):
    colors = np.concatenate([state.color_p, state.color_o[::-1]])
    pieces = np.concatenate([state.pieces_p, state.pieces_o[::-1]])
    x = pieces % 6
    y = pieces // 6

    if state.root_player == 1:
        x[pieces >= 0] = 5 - x[pieces >= 0]
        y[pieces >= 0] = 5 - y[pieces >= 0]

    x[pieces == game.CAPTURED] = 9
    y[pieces == game.CAPTURED] = 9

    msg = ''
    for i in range(16):
        c = COLOR_NAMES[colors[i]]

        if i < 8 and (x[i] < 6 and y[i] < 6):
            c = c.upper()

        if i >= 8 and (x[i] < 6 and y[i] < 6):
            c = 'u'

        msg += f'{x[i]}{y[i]}{c}'

    return f'MOV?{msg}\r\n'


def parse_board_str(s: str, player: int):
    pieces_o = np.zeros(8, dtype=np.int16)
    color_o = np.zeros(8, dtype=np.int16)

    offset = 24 + 4

    for i in range(8):
        x = int(s[offset + 0 + i*3])
        y = int(s[offset + 1 + i*3])
        c = s[offset + 2 + i*3]

        if x == 9 and y == 9:
            pieces_o[i] = -1
        else:
            pieces_o[i] = y * 6 + x

        color_o[i] = COLOR_DICT[c]

    if player == 1:
        pieces_o[pieces_o >= 0] = 35 - pieces_o[pieces_o >= 0]
        return pieces_o, color_o
    else:
        return pieces_o[::-1], color_o[::-1]
