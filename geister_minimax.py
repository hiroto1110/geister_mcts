import copy
import random
from dataclasses import dataclass

import ray
import numpy as np
from tqdm import tqdm

import geister as game


def create(player, depth):
    return SearchParam(player, depth, -100000, 100000)


@dataclass
class SearchParam:
    player: int
    depth: int
    alpha: float
    beta: float

    def deepen(self):
        return SearchParam(-self.player, self.depth - 1, -self.beta, -self.alpha)

    def null_window(self):
        return SearchParam(-self.player, self.depth - 1, -self.alpha - 1, -self.alpha)


def step_as_purple(state: game.State, action: int, player: int):
    next_state = copy.deepcopy(state)
    next_state.n_ply += 1

    pos = game.action_to_pos(action)
    d = game.action_to_direction(action)
    pos_next = pos + d

    piece = next_state.board[pos]
    piece_next = next_state.board[pos_next]

    assert piece * player > 0
    assert piece_next * player <= 0

    if piece_next != 0:
        if player == 1:
            next_state.captured_p[1] += 1
        else:
            next_state.captured_o[piece_next - 1] += 1

    next_state.board[pos_next] = piece
    next_state.board[pos] = 0

    flag = next_state.board == -1
    next_state.board[flag] = -2

    next_state.update_is_done(player)

    next_state.board[flag] = -1

    return next_state, next_state.is_done


def eval_state(state: game.State, player):
    e_cap = 0
    e_cap += state.captured_p[0] - state.captured_p[1]
    e_cap -= state.captured_o[0] - state.captured_o[1]

    e_win = state.winner * 100

    pos1 = np.where(state.board == 1)[0]
    x = (pos1 % 6)
    y = pos1 / 6
    d1 = np.abs(x - 2.5) + y

    pos2 = np.where(state.board < 0)[0]
    x = (pos2 % 6)
    y = pos2 / 6
    d2 = np.abs(x - 2.5) + 6 - y

    if d1.shape[0] == 0:
        d1 = 0

    if d2.shape[0] == 0:
        d2 = 0

    e_esc = np.mean(d1) - np.mean(d2)
    e_esc *= 4

    return (e_cap + e_esc + e_win) * player


def solve(state: game.State, p: SearchParam):
    if p.depth <= 0:
        return eval_state(state, p.player), -1

    actions = game.get_valid_actions(state, p.player)

    next_state, done = step_as_purple(state, actions[0], p.player)
    max_e, _ = solve(next_state, p.deepen())
    max_e *= -1
    max_a = actions[0]

    for action in actions[1:]:
        next_state, done = step_as_purple(state, action, p.player)
        e, _ = solve(next_state, p.null_window())
        e *= -1

        if p.beta <= e:
            return e, -1

        if p.alpha < e:
            p.alpha = e
            e, _ = solve(next_state, p.deepen())
            e *= -1

            if p.beta <= e:
                return e, -1

            if p.alpha < e:
                max_a = action
                p.alpha = e

        if max_e < e:
            max_e = e

    return max_e, max_a


def as_relative(state, player):
    if player == 1:
        return state

    state = copy.deepcopy(state)

    state.board = -state.board[::-1]
    state.captured_p, state.captured_o = state.captured_o, state.captured_p

    return state


@ray.remote(num_cpus=1, num_gpus=0)
def play_game(depth, epsilon, print_info=False):
    action_log = []

    init_state = game.get_initial_state()
    state = copy.deepcopy(init_state)

    player = 1
    done = False

    while not done:
        if random.random() >= epsilon:
            _, action = solve(as_relative(state, player), create(1, depth))

            if player == -1:
                action = 143 - action

                if action not in game.get_valid_actions(state, player):
                    print(game.get_valid_actions(state, player))
                    print(143 - action, action)
                    print()
        else:
            action = random.choice(game.get_valid_actions(state, player))

        action_log.append(int(action))

        state, done = game.step(state, action, player)

        if print_info:
            print(game.get_valid_actions(state, player))
            print(action in game.get_valid_actions(state, player))
            print(player, action)
            print()

            print(state.board.reshape(6, 6))
            print(state.captured_p)
            print(state.captured_o)
            print(state.n_ply)
            print()

        player = -player

    return init_state, action_log


def main():
    num_cpu = 32

    num_games = 10000

    depth = 3
    epsilon = 0.3

    # play_game(depth, epsilon, True)
    # return

    work_in_progresses = [play_game.remote(depth, epsilon) for _ in range(num_cpu - 2)]

    with open("log.txt", mode='w') as f:
        for _ in tqdm(range(num_games)):
            finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            init_state, action_log = ray.get(finished[0])

            f.write(' '.join([str(i) for i in init_state.board]) + ',')
            f.write(' '.join([str(i) for i in action_log]) + '\n')
            work_in_progresses.append(play_game.remote(depth, epsilon))


def line_to_tokens(line, n_tokens):
    board, actions = line.split(',')
    board = [int(i) for i in board.split()]
    actions = [int(i) for i in actions.split()]

    # print(f"length actions: {len(actions)}")

    state = game.State()
    state.board = np.array(board)

    pieces = np.where(state.board != 0)[0]
    types = state.board[pieces]
    labels = types[8:] + 2
    types[types < 0] = 0

    tokens = np.zeros((n_tokens, 5), dtype=np.int32)

    mask = np.full(n_tokens, 1, dtype=np.int32)
    mask[:len(tokens)] = 1

    for i in range(min(len(actions), n_tokens)):
        a = actions[i]
        player = 1 - 2 * (i % 2)
        state, done = game.step(state, a, player)

        pos = game.action_to_pos(a)
        d = game.action_to_direction(a)

        p_id = np.where(pieces == pos)[0][0]
        pieces[p_id] += d

        tokens[i, 0] = types[p_id]
        tokens[i, 1] = p_id
        tokens[i, 2] = pieces[p_id] % 6
        tokens[i, 3] = pieces[p_id] // 6
        tokens[i, 4] = i
        # print(state.board.reshape(6, 6))
        # print(tokens[i])
        # print()

    return tokens, mask, labels


def main_token(max_length):
    with open("log.txt", "r") as f:
        lines = f.readlines()

    games = []
    masks = []
    labels = []
    for i in range(1000):
        tokens, mask, label = line_to_tokens(lines[i], max_length)

        games.append(tokens)
        masks.append(mask)
        labels.append(label)

    return np.array(games, dtype=np.int32), np.array(masks, dtype=np.int32), np.array(labels, dtype=np.int32)


if __name__ == "__main__":
    main_token(100)
