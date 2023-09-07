from functools import partial
from typing import Any, Callable
import time

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import core, struct
from flax.training import checkpoints

import geister as game
import geister_lib
from network_transformer import TransformerDecoderWithCache

from graphviz import Digraph


class PredictState(struct.PyTreeNode):
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)


@partial(jax.jit, device=jax.devices("cpu")[0])
def predict(state: PredictState, tokens, cache_v, cache_k):
    pi, v, c, cv, ck = state.apply_fn({'params': state.params}, tokens, cache_v, cache_k, eval=True)

    pi = pi[0, 0]
    v = nn.softmax(v[0, 0]) * np.array([-1, -1, -1, 0, 1, 1, 1], dtype=np.int8)
    v = v.sum()
    c = nn.sigmoid(c[0, 0])

    return pi, v, c, cv, ck


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)


should_do_visibilize_node_graph = __name__ == '__main__'


class Node:
    def __init__(self, root_player: int) -> None:
        self.root_player = root_player
        self.winner = 0

        self.state_str = ""

        self.c_init = 1.5
        self.c_base = 19652

        self.cache_v = None
        self.cache_k = None
        self.valid_actions = None
        self.invalid_actions = np.full(shape=1, fill_value=-1)

        self.predicted_color = None
        self.predicted_v = 0

        self.children = [None] * game.ACTION_SPACE
        self.p = np.zeros(game.ACTION_SPACE)
        self.w = np.zeros(game.ACTION_SPACE)
        self.n = np.zeros(game.ACTION_SPACE)

    def apply_invalid_actions(self):
        if self.valid_actions is None:
            return

        self.valid_actions = np.setdiff1d(self.valid_actions, self.invalid_actions)

        self.valid_actions_mask = np.zeros(144, dtype=np.int8)
        self.valid_actions_mask[self.valid_actions] = 1

        self.p = np.where(self.valid_actions_mask, self.p, -np.inf)
        self.p = softmax(self.p)

    def setup_valid_actions(self, state, player):
        if self.valid_actions is not None:
            return

        self.valid_actions = game.get_valid_actions(state, player)
        self.valid_actions = np.setdiff1d(self.valid_actions, self.invalid_actions)

        self.valid_actions_mask = np.zeros(144, dtype=np.int8)
        self.valid_actions_mask[self.valid_actions] = 1

        self.p = np.where(self.valid_actions_mask, self.p, -np.inf)
        self.p = softmax(self.p)

    def calc_scores(self):
        c = self.c_init * np.log((self.n.sum() + 1 + self.c_base) / self.c_base)

        U = c * self.p * np.sqrt(self.n.sum() + 1) / (self.n + 1)
        Q = self.w / np.where(self.n != 0, self.n, 1)

        scores = U + Q
        scores = np.where(self.valid_actions_mask, scores, -np.inf)

        return scores

    def get_policy(self):
        return self.n / self.n.sum()


class AfterStateNode:
    def __init__(self, root_player: int) -> None:
        self.root_player = root_player
        self.winner = 0

        self.state_str = ""

        self.cache_v = None
        self.cache_k = None

        self.predicted_color = None
        self.predicted_v = 0

        self.children = [None] * 2
        self.p = np.zeros(2)
        self.w = np.zeros(2)
        self.n = np.zeros(2)


def visibilize_node_graph(node: Node, g: Digraph):
    if isinstance(node, Node):
        for child, w, n, p in zip(node.children, node.w, node.n, node.p):
            if child is None:
                continue

            v = w / n if n > 0 else 0
            v_ = child.predicted_v
            label = f"w = {v:.3f}\r\nv = {v_:.3f}\r\np = {p:.3f}"
            g.edge(node.state_str, child.state_str, label)

            visibilize_node_graph(child, g)

    else:
        for i in range(2):
            child = node.children[i]

            if child is None:
                continue

            color = 'blue' if i == 1 else 'red'

            v = node.w[i] / node.n[i] if node.n[i] > 0 else 0
            v_ = child.predicted_v
            p = node.p[i]
            label = f"{color}\r\nw = {v:.3f}\r\nv = {v_:.3f}\r\np = {p:.3f}"
            g.edge(node.state_str, child.state_str, label=label)

            visibilize_node_graph(child, g)


def expand_afterstate(node: Node,
                      state: game.SimulationState,
                      pred_state: PredictState,
                      info: game.AfterstateInfo):

    next_node = AfterStateNode(node.root_player)

    if should_do_visibilize_node_graph:
        next_node.state_str = sim_state_to_str(state)

    tokens = state.get_afterstate_tokens(info)
    v, _ = setup_node(next_node, node, pred_state, tokens)

    next_node.p[1] = next_node.predicted_color[info.piece_id]
    next_node.p[0] = 1 - next_node.p[1]

    return next_node, v


def expand(node: Node,
           state: game.SimulationState,
           pred_state: PredictState):

    next_node = Node(node.root_player)
    next_node.winner = state.winner

    if should_do_visibilize_node_graph:
        next_node.state_str = sim_state_to_str(state)

    if node.winner != 0:
        next_node.winner = node.winner

    """if next_node.winner == 0 and player == node.root_player:
        action = find_checkmate(state, player, depth=4)

        if action != -1:
            next_node.winner = player
            next_node.n[action] += 1"""

    if next_node.winner != 0:
        return next_node, next_node.winner

    tokens = state.get_last_tokens()
    v, next_node.p = setup_node(next_node, node, pred_state, tokens)

    return next_node, v


def expand_with_observation(node: Node,
                            state: game.State,
                            pred_state: PredictState):
    next_node = Node(node.root_player)
    next_node.winner = state.winner

    if state.winner != 0:
        return next_node

    cv, ck = node.cache_v, node.cache_k

    tokens = state.get_last_tokens(node.root_player)
    tokens = jnp.array([tokens], dtype=jnp.uint8)

    for i in range(tokens.shape[1]):
        pi, _, _, cv, ck = predict(pred_state, tokens[:, i:i+1], cv, ck)

    next_node.p = jax.device_get(pi)
    next_node.cache_v = cv
    next_node.cache_k = ck

    return next_node


def setup_node(node: Node, parent_node: Node, pred_state: PredictState, tokens):
    tokens = jnp.array(tokens, dtype=jnp.uint8)

    pi, v, c, cv, ck = predict(pred_state, tokens, parent_node.cache_v, parent_node.cache_k)

    if np.isnan(c).any():
        c = np.full(shape=8, fill_value=0.5)

    node.predicted_color = c
    node.predicted_v = v
    node.cache_v = cv
    node.cache_k = ck

    return jax.device_get(v), jax.device_get(pi)


def simulate_afterstate(node: AfterStateNode,
                        state: game.SimulationState,
                        player: int,
                        pred_state: PredictState,
                        info: game.AfterstateInfo) -> float:
    color = np.random.choice([game.RED, game.BLUE], p=node.p)

    state.step_afterstate(info, color)

    if node.children[color] is None:
        child, v = expand(node, state, pred_state)
        node.children[color] = child
    else:
        v = simulate(node.children[color], state, player, pred_state)

    state.undo_step_afterstate(info)

    node.n[color] += 1
    node.w[color] += v * player

    return v


def simulate(node: Node,
             state: game.SimulationState,
             player: int,
             pred_state: PredictState) -> float:

    if state.is_done:
        return state.winner

    if node.winner != 0:
        return node.winner

    node.setup_valid_actions(state, player)

    scores = node.calc_scores()
    action = np.argmax(scores)

    info = state.step(action, player)

    if node.children[action] is None:
        if info.is_afterstate():
            child, v = expand_afterstate(node, state, pred_state, info)
        else:
            child, v = expand(node, state, pred_state)

        node.children[action] = child
    else:
        if info.is_afterstate():
            v = simulate_afterstate(node.children[action], state, -player, pred_state, info)
        else:
            v = simulate(node.children[action], state, -player, pred_state)

    state.undo_step(action, player)

    node.n[action] += 1
    node.w[action] += v * player

    return v


def find_checkmate(state: game.State, player: int, depth: int):
    if player == 1:
        n_cap_ob = np.sum((state.pieces_o == game.CAPTURED) & (state.color_o == game.BLUE))
        return geister_lib.find_checkmate(state.pieces_p, state.color_p, state.pieces_o, n_cap_ob, player, depth)

    else:
        n_cap_ob = np.sum((state.pieces_p == game.CAPTURED) & (state.color_p == game.BLUE))
        return geister_lib.find_checkmate(state.pieces_o, state.color_o, state.pieces_p, n_cap_ob, player, depth)


def step(node1: Node,
         node2: Node,
         state: game.State,
         player: int,
         pred_state: PredictState,
         num_sim: int,
         alpha: float = None,
         eps: float = 0.25):

    node = node1 if player == 1 else node2

    sim_state = game.SimulationState(state, node.root_player)

    if should_do_visibilize_node_graph:
        node.state_str = sim_state_to_str(sim_state)

    if node.winner != 0:
        if np.sum(node.n) == 0:
            node.setup_valid_actions(state, player)
            action = np.random.choice(node.valid_actions)
        else:
            action = np.argmax(node.n)
    else:
        action = -1
    # start = time.perf_counter()
    checkmate_action = find_checkmate(state, player, depth=8)
    # action = -1
    # print(f"time: {time.perf_counter() - start}")

    if checkmate_action != -1:
        action = checkmate_action

    if action != -1:
        pass
        # print(f"find checkmate: {action}")

    else:
        node.setup_valid_actions(sim_state, 1)

        if alpha is not None:
            dirichlet_noise = np.random.dirichlet(alpha=[alpha]*len(node.valid_actions))

            for a, noise in zip(node.valid_actions, dirichlet_noise):
                node.p[a] = (1 - eps) * node.p[a] + eps * noise

        for _ in range(num_sim):
            simulate(node, sim_state, 1, pred_state)

        policy = node.get_policy()
        action = np.argmax(policy)

        if should_do_visibilize_node_graph:
            dg = Digraph(format='png')
            dg.attr('node', fontname="Myrica M")
            visibilize_node_graph(node, dg)
            dg.render(f'./graph/n_ply_{state.n_ply}')

    state.step(action, player)

    node1 = expand_with_observation(node1, state, pred_state)
    node2 = expand_with_observation(node2, state, pred_state)

    return action, node1, node2


def init_jit(state: PredictState, model: TransformerDecoderWithCache, data):
    cv, ck = model.create_cache(1, 0)

    for t in range(50):
        print(t)
        pi, v, _, cv, ck = predict(state, state.params, data[0][:1, t:t+1], cv, ck)


def create_root_node(state: game.State,
                     pred_state: PredictState,
                     model: TransformerDecoderWithCache,
                     player: int) -> Node:
    node = Node(player)
    cache_v, cache_k = model.create_cache(1, 0)

    tokens = state.tokens_p if player == 1 else state.tokens_o
    tokens = tokens[:8]
    tokens = jnp.array([tokens], dtype=jnp.uint8)

    for i in range(tokens.shape[1]):
        _, _, _, cache_v, cache_k = predict(pred_state, tokens[:, i:i+1], cache_v, cache_k)

    node.cache_v = cache_v
    node.cache_k = cache_k

    return node


def create_invalid_actions(actions, state: game.State, player: int, pieces_history: np.ndarray):
    invalid_actions = []
    exist_valid_action = False

    for a in actions:
        state.step(a, player)

        pieces = state.pieces_p if player == 1 else state.pieces_o
        is_equals = np.all(pieces_history == pieces, axis=1)

        state.undo_step(a, player)

        if np.any(is_equals):
            invalid_actions.append(a)
        else:
            exist_valid_action = True

    if not exist_valid_action:
        del_i = np.random.randint(0, len(invalid_actions))
        del invalid_actions[del_i]

    return np.array(invalid_actions, dtype=np.int16)


def play_game(pred_state: PredictState,
              model: TransformerDecoderWithCache,
              num_mcts_sim1: int, num_mcts_sim2: int,
              dirichlet_alpha: float,
              game_length: int = 200,
              print_board: bool = False):
    state = game.get_initial_state()

    player = 1

    node1 = create_root_node(state, pred_state, model, 1)
    node2 = create_root_node(state, pred_state, model, -1)

    action_history = np.zeros(game_length + 1, dtype=np.int16)
    pieces_history = np.zeros((2, 100, 8), dtype=np.int8)

    for i in range(game_length):
        pieces_history[i % 2, i // 2] = state.pieces_p if player == 1 else state.pieces_o

        node1.setup_valid_actions(state, player)
        node2.setup_valid_actions(state, player)

        invalid_actions = create_invalid_actions(node1.valid_actions, state, player, pieces_history[i % 2])
        node1.invalid_actions = node2.invalid_actions = invalid_actions

        node1.apply_invalid_actions()
        node2.apply_invalid_actions()

        action, node1, node2 = step(node1, node2,
                                    state, player,
                                    pred_state,
                                    num_mcts_sim1 if player == 1 else num_mcts_sim2,
                                    dirichlet_alpha)

        action_history[i] = action

        if print_board:
            board = np.zeros(36, dtype=np.int8)

            board[state.pieces_p[(state.pieces_p >= 0) & (state.color_p == 1)]] = 1
            board[state.pieces_p[(state.pieces_p >= 0) & (state.color_p == 0)]] = 2
            board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == 1)]] = -1
            board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == 0)]] = -2

            print(str(board.reshape((6, 6))).replace('0', ' '))
            print(i)

        if state.is_done:
            break

        player = -player

    return state, action_history


def sim_state_to_str(state: game.SimulationState):
    board = np.zeros(36, dtype=np.int8)

    board[state.pieces_p[(state.pieces_p >= 0) & (state.color_p == game.BLUE)]] = 1
    board[state.pieces_p[(state.pieces_p >= 0) & (state.color_p == game.RED)]] = 2
    board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == game.BLUE)]] = -1
    board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == game.RED)]] = -2
    board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == game.UNCERTAIN_PIECE)]] = -3

    s = str(board.reshape((6, 6))).replace('0', ' ')
    s = s.replace('[[', ' [').replace('[', '|')
    s = s.replace(']]', ']').replace(']', '|')

    n_cap_b = np.sum((state.pieces_o == game.CAPTURED) & (state.color_o == game.BLUE))
    n_cap_r = np.sum((state.pieces_o == game.CAPTURED) & (state.color_o == game.RED))

    s += f"\r\nblue={n_cap_b} red={n_cap_r}"

    return s


def test():
    # data = [jnp.load(f"data_{i}.npy") for i in range(4)]

    model_with_cache = TransformerDecoderWithCache(num_heads=8, embed_dim=128, num_hidden_layers=2)

    ckpt_dir = './checkpoints/'
    prefix = 'geister_'

    ckpt = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, prefix=prefix, target=None)
    pred_state = PredictState(apply_fn=model_with_cache.apply,
                              params=ckpt['params'])

    # init_jit(pred_state, model_with_cache, data)

    elapsed_times = []

    for i in range(1):
        start = time.perf_counter()
        play_game(pred_state, model_with_cache, 100, 100, 0.3, print_board=True)
        elapsed = time.perf_counter() - start
        print(f"time: {elapsed} s")

        if i > 0:
            elapsed_times.append(elapsed)

    print(f"avg time: {sum(elapsed_times) / len(elapsed_times)} s")


if __name__ == "__main__":
    test()
