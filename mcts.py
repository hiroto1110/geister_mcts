from functools import partial
import time

import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import jax
import optax
from flax.training import checkpoints

import geister as game
from network_transformer import TransformerDecoder, TransformerDecoderWithCache, TrainState


@partial(jax.jit, device=jax.devices("cpu")[0])
def predict(state, params, tokens, cache_v, cache_k):
    return state.apply_fn({'params': params}, tokens, cache_v, cache_k, eval=True)


class Node:
    def __init__(self, root_player: int) -> None:
        self.root_player = root_player
        self.cache_v = None
        self.cache_k = None

        self.children = [None] * game.ACTION_SPACE
        self.p = np.zeros(game.ACTION_SPACE)
        self.w = np.zeros(game.ACTION_SPACE)
        self.n = np.zeros(game.ACTION_SPACE)

    def get_policy(self):
        return self.n / self.n.sum()


def expand(node: Node,
           state: game.State,
           action: int,
           train_state: TrainState):

    next_node = Node(node.root_player)
    node.children[action] = next_node

    cv, ck = jax.device_put(node.cache_v), jax.device_put(node.cache_k)

    if state.tokens_p[-1][4] == state.tokens_p[-2][4]:
        index = -2
    else:
        index = -1

    tokens = state.tokens_p[index:] if node.root_player == 1 else state.tokens_o[index:]
    tokens = jnp.array(tokens, dtype=jnp.uint8)

    for i in range(tokens.shape[0]):
        token = tokens[i].reshape(1, 1, -1)
        pi, v, _, cv, ck = predict(train_state, train_state.params, token, cv, ck)

    next_node.p = nn.softmax(jax.device_get(pi)[0, 0])
    next_node.cache_v = cv
    next_node.cache_k = ck

    return jax.device_get(v)[0, 0, 0]


def simulate(node: Node,
             state: game.State,
             player: int,
             train_state: TrainState,
             c_puct: float = 1) -> float:

    if game.is_done(state, player):
        return state.winner

    U = c_puct * node.p * np.sqrt(node.n.sum() + 1) / (1 + node.n)
    Q = node.w / np.where(node.n != 0, node.n, 1)

    valid_actions = game.get_valid_actions(state, player)

    action_mask = np.zeros(game.ACTION_SPACE)
    action_mask[valid_actions] = 1

    scores = U + Q
    scores = np.where(action_mask, scores, -np.inf)

    action = np.random.choice(np.where(scores == scores.max())[0])

    state.step(action, player)

    if node.children[action] is None:
        v = -expand(node, state, action, train_state)
    else:
        v = -simulate(node.children[action], state, -player, train_state, c_puct)

    state.undo_step(action, player)

    node.n[action] += 1
    node.w[action] += v

    return v


def step(node1: Node, node2: Node, state: game.State, player: int, train_state: TrainState, num_sim: int):
    node = node1 if player == 1 else node2

    for i in range(num_sim):
        # start = time.perf_counter()
        simulate(node, state, player, train_state)
        # print(f"sim: {i}, {time.perf_counter() - start}")

    policy = node.get_policy()
    action = np.argmax(policy)

    state.step(action, player)

    if node1.children[action] is None:
        expand(node1, state, action, train_state)

    if node2.children[action] is None:
        expand(node2, state, action, train_state)

    return policy, node1.children[action], node2.children[action]


def init_jit(state: TrainState, model: TransformerDecoderWithCache, data):
    cv, ck = model.create_cache(1, 0)

    for t in range(140):
        print(t)
        pi, v, _, cv, ck = predict(state, state.params, data[0][:1, t:t+1], cv, ck)


def create_root_node(state: game.State, train_state: TrainState, model: TransformerDecoderWithCache, player: int):
    node = Node(player)
    cache_v, cache_k = model.create_cache(1, 0)

    tokens = state.tokens_p if player == 1 else state.tokens_o

    for token in tokens:
        token = jnp.array(token, dtype=jnp.uint8).reshape(1, 1, -1)
        _, _, _, cache_v, cache_k = predict(train_state, train_state.params, token, cache_v, cache_k)

    node.cache_v = cache_v
    node.cache_k = cache_k

    return node


def test():
    data = [jnp.load(f"data_{i}.npy") for i in range(4)]

    model = TransformerDecoder(num_heads=8, embed_dim=128, num_hidden_layers=2)
    model_with_cache = TransformerDecoderWithCache(num_heads=8, embed_dim=128, num_hidden_layers=2)

    key, key1, key2 = jax.random.split(jax.random.PRNGKey(0), 3)

    variables = model.init(key1, data[0][:1])
    train_state = TrainState.create(
        apply_fn=model_with_cache.apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=0.00005),
        dropout_rng=key2,
        epoch=0)

    ckpt_dir = './checkpoints/'
    prefix = 'geister_'

    train_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, prefix=prefix, target=train_state)

    # init_jit(train_state, model_with_cache, data)

    state = game.get_initial_state()

    player = 1

    node1 = create_root_node(state, train_state, model_with_cache, 1)
    node2 = create_root_node(state, train_state, model_with_cache, -1)

    for i in range(200):
        start = time.perf_counter()
        node1, node2 = step(node1, node2, state, player, train_state, num_sim=25)
        print(f"step: {i}, {time.perf_counter() - start}")

        board = np.zeros(36, dtype=np.int8)

        board[state.pieces_p[(state.pieces_p >= 0) & (state.color_p == 1)]] = 1
        board[state.pieces_p[(state.pieces_p >= 0) & (state.color_p == 0)]] = 2
        board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == 1)]] = -1
        board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == 0)]] = -2

        print(str(board.reshape((6, 6))).replace('0', ' '))

        if game.is_done(state, player):
            break

        player = -player

    state.update_is_done(-player)
    print(state.is_done, state.winner)


if __name__ == "__main__":
    test()
