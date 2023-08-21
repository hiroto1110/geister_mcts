from typing import Tuple
from functools import partial
import time

import numpy as np
import jax.numpy as jnp
import jax
import optax
from flax.training import checkpoints

import geister_numba as game
from network_transformer import TransformerDecoderWithCache, TrainState


@partial(jax.jit, device=jax.devices("cpu")[0])
def predict(state, params, tokens):
    return state.apply_fn({'params': params}, tokens, eval=True)


class Node:
    def __init__(self, parent, action: int) -> None:
        self.parent = parent
        self.action = action
        self.token = []

        self.childs = [None] * game.ACTION_SPACE
        self.p = np.zeros(game.ACTION_SPACE)
        self.w = np.zeros(game.ACTION_SPACE)
        self.n = np.zeros(game.ACTION_SPACE)

    def get_policy(self):
        return self.n / self.n.sum()

    def apply_pi(self, pi):
        self.p = np.array(pi)

    def apply_v(self, v):
        if self.parent is None:
            return

        self.parent.w[self.action] -= v
        self.parent.apply_v(-v)


def simulate(node: Node, state: game.State, player: int, c_puct: float) -> Tuple[Node, np.ndarray]:
    if game.is_done(state, player):
        return node, np.zeros(0)

    U = c_puct * node.p * np.sqrt(node.n.sum()) / (1 + node.n)
    Q = node.w / np.where(node.n != 0, node.n, 1)

    valid_actions = game.get_valid_actions(state, player)

    action_mask = np.zeros(game.ACTION_SPACE)
    action_mask[valid_actions] = 1

    scores = U + Q
    scores = np.where(action_mask, scores, -np.inf)

    action = np.random.choice(np.where(scores == scores.max())[0])

    state.step(action, player)

    if node.childs[action] is None:
        next_node = Node(node, action)
        node.childs[action] = next_node

        leaf_node = next_node
        tokens = game.get_tokens(state, -player, 200)
    else:
        next_node = node.childs[action]
        leaf_node, tokens = simulate(next_node, state, -player, c_puct)

    state.undo_step(action, player)

    node.n = node.n.copy()
    node.n[action] += 1

    return leaf_node, tokens


def step_batch(train_state: TrainState, params, states: list, player: int, num_sim: int):
    nodes = [Node(None, -1) for _ in range(len(states))]

    for i in range(num_sim):
        results = [simulate(node, state, player, 1) for node, state in zip(nodes, states)]

        length = max([tokens.shape[0] for _, tokens in results])

        tokens_batch = [np.resize(tokens, (length, game.TOKEN_SIZE)) for _, tokens in results]
        tokens_batch = np.stack(tokens_batch, axis=0).astype(np.int16)

        pi, v, _, _ = predict(train_state, params, tokens_batch)

        for j in range(len(states)):
            index = results[j][1].shape[0]
            results[j][0].apply_pi(pi[j, index])
            results[j][0].apply_v(v[j, index])

    actions = []

    for j in range(len(states)):
        policy = nodes[j].get_policy()
        action = np.argmax(policy)

        actions.append(action)

        # print(policy)
        # print(action)
        # print(game.get_valid_actions(states[j], player))

        states[j].step(action, player)


def test():
    data = [jnp.load(f"data_{i}.npy") for i in range(4)]

    model = TransformerDecoderWithCache(num_heads=8, embed_dim=128, num_hidden_layers=2)

    key, key1, key2 = jax.random.split(jax.random.PRNGKey(0), 3)

    variables = model.init(key1, data[0][:1])
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=0.00005),
        dropout_rng=key2,
        epoch=0)

    ckpt_dir = './checkpoints/'
    prefix = 'geister_'

    train_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, prefix=prefix, target=train_state)

    n = 32
    states = [game.get_initial_state() for _ in range(n)]

    player = 1

    for i in range(100):
        start = time.perf_counter()
        step_batch(train_state, train_state.params, states, player, num_sim=25)
        print(time.perf_counter() - start)

        player = -player


if __name__ == "__main__":
    test()
