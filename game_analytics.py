import numpy as np
import termcolor

import env.state as game


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


def states_to_str(
    states: list[game.SimulationState],
    predicted_colors: list[np.ndarray],
    true_colors: list[np.ndarray] = None,
    colored: bool = False,
    concat_line=True
) -> str | list[str]:
    lines_list = [
        state_to_str(s, p, c, colored, concat_line=False)
        for s, p, c in zip(states, predicted_colors, true_colors)
    ]

    lines = [" ".join(line) for line in zip(*lines_list)]

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
        if 0 <= pos < 36:
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
