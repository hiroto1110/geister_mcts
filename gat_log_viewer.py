import glob
import time

import numpy as np
import orbax.checkpoint

import geister as game
import mcts
import game_analytics
from network_transformer import TransformerDecoderWithCache


NAME_TO_ID_1 = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1, 'H': 0}
NAME_TO_ID_2 = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}

NAME_TO_DIRECTION_ID = {'N': 0, 'W': 1, 'E': 2, 'S': 3}


def try_action(state: game.SimulationState, action, player, true_color_o):
    tokens, afterstates = state.step(action, player * state.root_player)

    for afterstate in afterstates:
        state.step_afterstate(afterstate, true_color_o[afterstate.piece_id])

    state_lines = game_analytics.state_to_str(state, [0.5] * 8, colored=True, concat_line=False)

    for afterstate in afterstates[::-1]:
        state.undo_step_afterstate(afterstate)
    state.undo_step(action, player * state.root_player, tokens, afterstates)

    return state_lines


def view_log(lines, player1: mcts.PlayerMCTS, player2: mcts.PlayerMCTS):
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

    player1.init_state(state1)
    player2.init_state(state2)

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

        log_action = piece_id * 4 + direction_id
        print(piece_id, direction_id)

        if player == 1:
            player_action = player1.select_next_action()
            state_lines = try_action(state1, player_action, player, state2.color_p)
        else:
            player_action = player2.select_next_action()
            state_lines = try_action(state2, player_action, player, state1.color_p)

        player1.apply_action(log_action, player, state2.color_p)
        player2.apply_action(log_action, player, state1.color_p)

        if player1.node.predicted_color is None:
            preditec_color1 = [0.5] * 8
        else:
            preditec_color1 = player1.node.predicted_color

        if player2.node.predicted_color is None:
            preditec_color2 = [0.5] * 8
        else:
            preditec_color2 = player2.node.predicted_color

        state_lines1 = game_analytics.state_to_str(state1, preditec_color1, colored=True, concat_line=False)
        state_lines2 = game_analytics.state_to_str(state2, preditec_color2, colored=True, concat_line=False)

        for l1, l2, l3 in zip(state_lines1, state_lines2, state_lines):
            print(l1, l2, l3)
        print(i)

        player = -player

        if state1.is_done or state2.is_done:
            break


def load_actions(lines):
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

    actions = []

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

        actions.append(piece_id * 4 + direction_id)

        player = -player

    return actions, color1, color2


def main():
    ckpt_dir = './checkpoints/run-2'
    step = 500 * 12

    print(f"model: {ckpt_dir}, step: {step}")

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer)

    ckpt = checkpoint_manager.restore(step)

    model = TransformerDecoderWithCache(**ckpt['model'])
    params = ckpt['state']['params']

    np.random.seed(1444)

    mcts_params = mcts.SearchParameters(
        num_simulations=100,
        dirichlet_alpha=None,
        n_ply_to_apply_noise=0,
        max_duplicates=8,
        depth_search_checkmate_leaf=4,
        depth_search_checkmate_root=6,
        should_do_visibilize_node_graph=False
    )

    player1 = mcts.PlayerMCTS(params, model, mcts_params)
    player2 = mcts.PlayerMCTS(params, model, mcts_params)

    log_dir = '/home/kuramitsu/lab/geister/log_gat'

    selected_won_cnt = 0
    won_cnt = 0

    selected_lst_cnt = 0
    lst_cnt = 0

    start = time.perf_counter()

    for file in glob.glob(f'{log_dir}/*/*.txt'):
        with open(file, mode='r') as f:
            lines = f.readlines()

        actions, color1, color2 = load_actions(lines)

        selected_won_cnt_i, won_cnt_i, selected_lst_cnt_i, lst_cnt_i =\
            game_analytics.find_watershed_moments(player1, player2, color1, color2, actions, print_info=False)

        selected_won_cnt += selected_won_cnt_i
        won_cnt += won_cnt_i
        selected_lst_cnt += selected_lst_cnt_i
        lst_cnt += lst_cnt_i

    print(f"won: {round(selected_won_cnt / won_cnt, 3)}, {selected_won_cnt}/{won_cnt}")
    print(f"lst: {round(selected_lst_cnt / lst_cnt, 3)}, {selected_lst_cnt}/{lst_cnt}")
    print("time: ", time.perf_counter() - start)


if __name__ == '__main__':
    main()
