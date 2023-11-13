from enum import Flag, auto

import numpy as np
import termcolor

import geister as game
import geister_objective_lib


class WatershedMomentType(Flag):
    NONE = 0
    CONTINUE = auto()
    WON = auto()
    LOST = auto()


def func(state1: game.SimulationState, state2: game.SimulationState, player: int):
    valid_actions = game.get_valid_actions(state1, player)

    e = np.zeros(shape=len(valid_actions), dtype=np.int32)

    for i, action in enumerate(valid_actions):
        tokens, afterstates = state1.step(action, player)

        for afterstate in afterstates:
            state1.step_afterstate(afterstate, state2.color_p[afterstate.piece_id])

        if state1.winner != 0:
            e[i] = state1.winner * player
        else:
            _, e[i] = geister_objective_lib.find_checkmate(
                state1.pieces_p, state1.color_p,
                state1.pieces_o, state2.color_p,
                -player, 1, 6
            )

            e[i] *= -1

        for afterstate in afterstates:
            state1.undo_step_afterstate(afterstate)

        state1.undo_step(action, player, tokens, afterstates)

    e = np.clip(e, -1, 1)

    watershed_moment_type = WatershedMomentType.NONE

    if np.all(e == e[0]):
        return watershed_moment_type, [], []

    if np.any(e == 0):
        watershed_moment_type |= WatershedMomentType.CONTINUE

    if np.any(e == 1):
        watershed_moment_type |= WatershedMomentType.WON

    if np.any(e == -1):
        watershed_moment_type |= WatershedMomentType.LOST

    return watershed_moment_type, valid_actions[e < 0], valid_actions[e > 0]


def find_watershed_moments(player1, player2, color1, color2, actions, print_info=False):
    state1 = game.SimulationState(color1, root_player=1)
    state2 = game.SimulationState(color2, root_player=-1)

    player1.init_state(state1)
    player2.init_state(state2)

    player = 1

    selected_won_move = []
    selected_lst_move = []

    for log_action in actions:
        flags, lst_moves, won_moves = func(state1, state2, player)

        if flags != WatershedMomentType.NONE:
            if player == 1:
                player_action = player1.select_next_action()
            else:
                player_action = player2.select_next_action()

            if print_info:
                print()
                print("player: ", player)
                print("action: ", player_action)
                print("won_move", won_moves)
                print("lst_move", lst_moves)
                print(state_to_str_objectively(state1.pieces_p, color1, state2.pieces_p, color2, colored=True))

            if WatershedMomentType.WON in flags:
                selected_won_move.append(player_action in won_moves)

            if WatershedMomentType.LOST in flags:
                selected_lst_move.append(player_action in lst_moves)

            # break

        player1.apply_action(log_action, player, state2.color_p)
        player2.apply_action(log_action, player, state1.color_p)

        player = -player

        if state1.is_done or state2.is_done or (log_action in won_moves) or (log_action in lst_moves):
            break

    return sum(selected_won_move), len(selected_won_move), sum(selected_lst_move), len(selected_lst_move)


def state_to_str_objectively(
    pos_p: np.ndarray, color_p: np.ndarray,
    pos_o: np.ndarray, color_o: np.ndarray,
    colored: bool = False,
    concat_line=True
):
    def pieces_to_str(s, color):
        if not colored:
            return s
        if color == game.BLUE:
            return termcolor.colored(s, color='blue')
        if color == game.RED:
            return termcolor.colored(s, color='red')
        return s

    line = [" " for _ in range(36)]
    for i in range(8):
        pos = pos_p[i]
        if 0 <= pos < 36:
            line[pos] = pieces_to_str(str(i), color_p[i])

        pos = pos_o[i]
        if 0 <= pos < 36:
            line[pos] = pieces_to_str('abcdefgh'[i], color_o[i])

    lines = ["|" + "  ".join(line[i*6: (i+1)*6]) + "|" for i in range(6)]

    n_cap_b = np.sum((pos_o == game.CAPTURED) & (color_o == game.BLUE))
    n_cap_r = np.sum((pos_o == game.CAPTURED) & (color_o == game.RED))

    lines.append(f"blue={n_cap_b} red={n_cap_r}")

    if concat_line:
        return "\r\n".join(lines)
    else:
        return lines


def state_to_str(
    state: game.SimulationState,
    predicted_color: np.ndarray,
    true_color: np.ndarray = None,
    colored: bool = False,
    concat_line=True
):

    color_int = (np.array(predicted_color) * 10).astype(dtype=np.int16)
    color_int = np.clip(color_int, 0, 9)

    if true_color is None:
        true_color = state.color_o

    if colored:
        B_str = termcolor.colored('B', color='blue')
        R_str = termcolor.colored('R', color='red')
        b_str = termcolor.colored('b', color='blue')
        r_str = termcolor.colored('r', color='red')
    else:
        B_str = 'B'
        R_str = 'R'
        b_str = 'b'
        r_str = 'r'

    line = [" " for _ in range(36)]
    for i in range(8):
        pos = state.pieces_p[i]
        color = state.color_p[i]
        if pos != game.CAPTURED:
            if color == game.BLUE:
                line[pos] = B_str
            else:
                line[pos] = R_str

        pos = state.pieces_o[i]
        color = true_color[i]

        if 0 <= pos < 36:
            if color == game.BLUE:
                line[pos] = b_str
            elif color == game.RED:
                line[pos] = r_str
            else:
                line[pos] = str(color_int[i])

    lines = ["|" + "  ".join(line[i*6: (i+1)*6]) + "|" for i in range(6)]

    n_cap_b = np.sum((state.pieces_o == game.CAPTURED) & (state.color_o == game.BLUE))
    n_cap_r = np.sum((state.pieces_o == game.CAPTURED) & (state.color_o == game.RED))

    lines.append(f"blue={n_cap_b} red={n_cap_r}")

    if concat_line:
        return "\r\n".join(lines)
    else:
        return lines
