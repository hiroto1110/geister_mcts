from functools import partial
from typing import Any, Callable
import time

import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax import core
from flax import struct
import jax
from flax.training import checkpoints

import geister as game
from network_transformer import TransformerDecoderWithCache


class PredictState(struct.PyTreeNode):
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)


@partial(jax.jit, device=jax.devices("cpu")[0])
def predict(state, params, tokens, cache_v, cache_k):
    return state.apply_fn({'params': params}, tokens, cache_v, cache_k, eval=True)


class Node:
    def __init__(self, root_player: int) -> None:
        self.root_player = root_player
        self.cache_v = None
        self.cache_k = None
        self.valid_actions = None

        self.children = [None] * game.ACTION_SPACE
        self.p = np.zeros(game.ACTION_SPACE)
        self.w = np.zeros(game.ACTION_SPACE)
        self.n = np.zeros(game.ACTION_SPACE)

    def get_policy(self):
        return self.n / self.n.sum()


def expand(node: Node,
           state: game.State,
           action: int,
           pred_state: PredictState):

    next_node = Node(node.root_player)
    node.children[action] = next_node

    cv, ck = node.cache_v, node.cache_k

    if state.tokens_p[-1][4] == state.tokens_p[-2][4]:
        n_tokens = 2
    else:
        n_tokens = 1

    tokens = state.tokens_p[-n_tokens:] if node.root_player == 1 else state.tokens_o[-n_tokens:]
    tokens = jnp.array([tokens], dtype=jnp.uint8)

    for i in range(n_tokens):
        pi, v, _, cv, ck = predict(pred_state, pred_state.params, tokens[:, i:i+1], cv, ck)

    next_node.p = np.array(jax.device_get(nn.softmax(pi))[0, 0])
    next_node.cache_v = cv
    next_node.cache_k = ck

    return jax.device_get(v)[0, 0, 0]


def simulate(node: Node,
             state: game.State,
             player: int,
             pred_state: PredictState,
             c_puct: float = 1) -> float:

    if game.is_done(state, player):
        return state.winner

    U = c_puct * node.p * np.sqrt(node.n.sum() + 1) / (1 + node.n)
    Q = node.w / np.where(node.n != 0, node.n, 1)

    if node.valid_actions is None:
        node.valid_actions = game.get_valid_actions(state, player)

    valid_actions_mask = np.zeros(144, dtype=np.int8)
    valid_actions_mask[node.valid_actions] = 1

    scores = U + Q
    scores = np.where(valid_actions_mask, scores, -np.inf)

    action = np.argmax(scores)

    state.step(action, player)

    if node.children[action] is None:
        v = -expand(node, state, action, pred_state)
    else:
        v = -simulate(node.children[action], state, -player, pred_state, c_puct)

    state.undo_step(action, player)

    node.n[action] += 1
    node.w[action] += v

    return v


def manhattan_distance(p1, p2):
    x1 = p1 % 6
    y1 = p1 // 6
    x2 = p2 % 6
    y2 = p2 // 6

    return abs(x1 - x2) + abs(y1 - y2)


def pos_diff(p1, p2):
    x1 = p1 % 6
    y1 = p1 // 6
    x2 = p2 % 6
    y2 = p2 // 6

    return (x1 - x2), (y1 - y2)


def search_checkmate(state: game.State, player: int):
    if player == 1:
        escape_pos = game.ESCAPE_POS_P
        pieces_p = state.pieces_p[state.color_p == game.BLUE]
        pieces_o = state.pieces_o
    else:
        escape_pos = game.ESCAPE_POS_O
        pieces_p = state.pieces_o[state.color_o == game.BLUE]
        pieces_o = state.pieces_p

    pieces_p = pieces_p[pieces_p >= 0]
    pieces_o = pieces_o[pieces_o >= 0]

    for pos in escape_pos:
        d_p = manhattan_distance(pos, pieces_p)
        d_o = manhattan_distance(pos, pieces_o)

        if d_p.min() >= d_o.min():
            continue

        p_pos = pieces_p[np.argmin(d_p)]

        x_diff, y_diff = pos_diff(p_pos, pos)
        if y_diff != 0:
            action_d = 0 if y_diff > 0 else 3
        else:
            action_d = 1 if x_diff > 0 else 2

        action = p_pos * 4 + action_d
        return True, action

    return False, -1


def step(node1: Node,
         node2: Node,
         state: game.State,
         player: int,
         pred_state: PredictState,
         num_sim: int,
         alpha: float = None,
         eps: float = 0.25):
    is_checkmate, action = search_checkmate(state, player)

    if not is_checkmate:
        node = node1 if player == 1 else node2

        if alpha is not None:
            valid_actions = game.get_valid_actions(state, player)
            dirichlet_noise = np.random.dirichlet(alpha=[alpha]*len(valid_actions))

            for a, noise in zip(valid_actions, dirichlet_noise):
                node.p[a] = (1 - eps) * node.p[a] + eps * noise

        for i in range(num_sim):
            # start = time.perf_counter()
            simulate(node, state, player, pred_state)
            # print(f"sim: {i}, {time.perf_counter() - start}")

        policy = node.get_policy()
        action = np.argmax(policy)
    else:
        # print("find checkmate!")
        pass

    state.step(action, player)

    if node1.children[action] is None:
        expand(node1, state, action, pred_state)

    if node2.children[action] is None:
        expand(node2, state, action, pred_state)

    return action, node1.children[action], node2.children[action]


def init_jit(state: PredictState, model: TransformerDecoderWithCache, data):
    cv, ck = model.create_cache(1, 0)

    for t in range(200):
        print(t)
        pi, v, _, cv, ck = predict(state, state.params, data[0][:1, t:t+1], cv, ck)


def create_root_node(state: game.State, pred_state: PredictState, model: TransformerDecoderWithCache, player: int):
    node = Node(player)
    cache_v, cache_k = model.create_cache(1, 0)

    tokens = state.tokens_p if player == 1 else state.tokens_o

    for token in tokens:
        token = jnp.array(token, dtype=jnp.uint8).reshape(1, 1, -1)
        _, _, _, cache_v, cache_k = predict(pred_state, pred_state.params, token, cache_v, cache_k)

    node.cache_v = cache_v
    node.cache_k = cache_k

    return node


def play_test_game(pred_state, model):
    state = game.get_initial_state()

    player = 1

    node1 = create_root_node(state, pred_state, model, 1)
    node2 = create_root_node(state, pred_state, model, -1)

    for i in range(300):
        # start = time.perf_counter()
        _, node1, node2 = step(node1, node2, state, player, pred_state, num_sim=50, alpha=0.3)
        # print(f"step: {i}, {time.perf_counter() - start}")

        board = np.zeros(36, dtype=np.int8)

        board[state.pieces_p[(state.pieces_p >= 0) & (state.color_p == 1)]] = 1
        board[state.pieces_p[(state.pieces_p >= 0) & (state.color_p == 0)]] = 2
        board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == 1)]] = -1
        board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == 0)]] = -2

        print(str(board.reshape((6, 6))).replace('0', ' '))
        print(i)

        if game.is_done(state, player):
            break

        player = -player

    return state.winner


def test():
    # data = [jnp.load(f"data_{i}.npy") for i in range(4)]

    model_with_cache = TransformerDecoderWithCache(num_heads=8, embed_dim=128, num_hidden_layers=2)

    ckpt_dir = './checkpoints/'
    prefix = 'geister_'

    ckpt = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, prefix=prefix, target=None)
    pred_state = PredictState(apply_fn=model_with_cache.apply,
                              params=ckpt['params'])

    # init_jit(pred_state, model_with_cache, data)

    for i in range(1):
        start = time.perf_counter()
        play_test_game(pred_state, model_with_cache)
        print(f"time: {time.perf_counter() - start} s")


if __name__ == "__main__":
    test()
