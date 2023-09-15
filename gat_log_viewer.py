import numpy as np

import geister as game


NAME_TO_ID_1 = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1, 'H': 0}
NAME_TO_ID_2 = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}

NAME_TO_DIRECTION_ID = {'N': 0, 'W': 1, 'E': 2, 'S': 3}


def step(state1: game.SimulationState, state2: game.SimulationState, action, player):
    _, info_list = state1.step(action, player)

    if player == -1:
        return

    if len(info_list) == 0:
        return

    for info in info_list:
        state1.step_afterstate(info, state2.color_p[info.piece_id])


def func(lines):
    def init_str_to_color(s, name_to_id):
        red_indices = [name_to_id[name] for name in s[:4]]

        color = np.full(8, fill_value=game.BLUE)
        color[red_indices] = game.RED

        return color

    color_str1 = lines[0].split(':')[1]
    color_str2 = lines[1].split(':')[1]

    if lines[0][7] == '1':
        color_str1, color_str2 = color_str2, color_str1

    color1 = init_str_to_color(color_str1, NAME_TO_ID_1)
    color2 = init_str_to_color(color_str2, NAME_TO_ID_2)

    state1 = game.SimulationState(color1, root_player=1)
    state2 = game.SimulationState(color2, root_player=-1)

    player = 1

    for i, line in enumerate(lines[2:-1]):
        move_str = line.split(':')[1].split(',')

        piece_name = move_str[0]
        direction_name = move_str[1][0]

        name_to_id = NAME_TO_ID_1 if player == 1 else NAME_TO_ID_2

        piece_id = name_to_id[piece_name]
        direction_id = NAME_TO_DIRECTION_ID[direction_name]

        if player == 1:
            direction_id = 3 - direction_id

        action = piece_id * 4 + direction_id

        print(piece_id, direction_id)

        step(state1, state2, action, player)
        step(state2, state1, action, -player)

        board = np.zeros(36, dtype=np.int8)

        board[state1.pieces_p[(state1.pieces_p >= 0) & (state1.color_p == 1)]] = 1
        board[state1.pieces_p[(state1.pieces_p >= 0) & (state1.color_p == 0)]] = 2
        board[state2.pieces_p[(state2.pieces_p >= 0) & (state2.color_p == 1)]] = -1
        board[state2.pieces_p[(state2.pieces_p >= 0) & (state2.color_p == 0)]] = -2

        print(str(board.reshape((6, 6))).replace('0', ' '))
        print(i)

        player = -player

        if state1.is_done or state2.is_done:
            break


def main():
    with open('./log_gat/2023/log/log_server1/log-2023-03-19-01-57-48-483_HauntedRails_UCTAokiBeta.txt', mode='r') as f:
        lines = f.readlines()

    func(lines)
    print(f'num_lines: {len(lines)}')


if __name__ == '__main__':
    main()
