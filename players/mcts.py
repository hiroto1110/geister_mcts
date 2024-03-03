from functools import partial
from typing import List, Any
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from graphviz import Digraph

import env.state as game
import env.checkmate_lib as checkmate_lib

import game_analytics
from network.transformer import TransformerWithCache
from network.checkpoints import CheckpointManager

from players.base import PlayerBase
from players.config import PlayerMCTSConfig, SearchParameters


@dataclass
class PredictState:
    model: TransformerWithCache
    params: dict[str, Any]


@partial(jax.jit, device=jax.devices("cpu")[0], static_argnames=["model"])
def predict(
    params: dict[str, Any],
    model: TransformerWithCache,
    x: jnp.ndarray,
    cache: jnp.ndarray,
    read_memory_i=None,
    write_memory_i=None
):
    x, pi, v, c, cache = model.apply(
        {'params': params},
        x, cache, read_memory_i, write_memory_i, eval=True
    )

    v = nn.softmax(v)
    c = nn.sigmoid(c)

    return x, pi, v, c, cache


class NodeBase:
    def __init__(self, action_space: int, has_afterstate: bool) -> None:
        self.children: list[NodeBase] = [None] * action_space
        self.p = np.zeros(action_space)
        self.w = np.zeros(action_space)
        self.n = np.zeros(action_space, dtype=np.int16)

        self.has_afterstate = has_afterstate

        self.winner = 0

        self.state_str = ""

        self.cache: jnp.ndarray = None
        self.predicted_color: jnp.ndarray = None
        self.predicted_v: jnp.ndarray = None

    def visualize_graph(self, g: Digraph):
        pass


class Node(NodeBase):
    def __init__(self) -> None:
        super().__init__(
            action_space=game.ACTION_SPACE,
            has_afterstate=False
        )

        self.valid_actions_mask = None
        self.invalid_actions = np.zeros(shape=0, dtype=np.uint8)

        self.predicted_color = None
        self.predicted_v = None

    def apply_invalid_actions(self):
        if self.valid_actions_mask is None:
            return

        self.valid_actions_mask[self.invalid_actions] = 0
        self.p = np.where(self.valid_actions_mask, self.p, 0)

    def setup_valid_actions(self, state, player):
        if self.valid_actions_mask is not None:
            return

        self.valid_actions_mask = game.get_valid_actions_mask(state, player)
        self.valid_actions_mask[self.invalid_actions] = 0

        self.valid_actions = np.where(self.valid_actions_mask)[0]

        self.p = np.where(self.valid_actions_mask, self.p, -np.inf)
        self.p = np.array(nn.softmax(self.p))

    def calc_scores(self, player: int, params: SearchParameters):
        c = params.c_init * np.log((self.n.sum() + 1 + params.c_base) / params.c_base)

        U = c * self.p * np.sqrt(self.n.sum() + 1) / (self.n + 1)
        Q = player * self.w / np.where(self.n != 0, self.n, 1)

        scores = U + Q
        scores = np.where(self.valid_actions_mask, scores, -np.inf)

        return scores

    def get_policy(self):
        return self.n / self.n.sum()

    def visualize_graph(self, g: Digraph):
        for child, w, n, p in zip(self.children, self.w, self.n, self.p):
            if child is None:
                continue

            v = w / n if n > 0 else 0
            label = f"w = {v:.3f}\r\np = {p:.3f}"
            g.edge(self.state_str, child.state_str, label)

            child.visualize_graph(g)


class AfterStateNode(NodeBase):
    def __init__(self, afterstates: List[game.Afterstate]):
        super().__init__(
            action_space=2,
            has_afterstate=True
        )

        self.afterstate = afterstates[0]
        self.remaining_afterstates = afterstates[1:]

    def visualize_graph(self, g: Digraph):
        for i in range(len(self.children)):
            child = self.children[i]

            if child is None:
                continue

            color = 'blue' if i == 1 else 'red'

            v = self.w[i] / self.n[i] if self.n[i] > 0 else 0
            p = self.p[i]
            label = f"{color}\r\nw = {v:.3f}\r\np = {p:.3f}"
            g.edge(self.state_str, child.state_str, label=label)

            child.visualize_graph(g)


def sim_state_to_str(state: game.SimulationState, v, color: np.ndarray):
    s = game_analytics.state_to_str(state, color, colored=False)

    vp = str.join(",", [str(int(v_i * 100)) for v_i in v])
    v = np.sum(v * np.array([-1, -1, -1, 0, 1, 1, 1]))
    s = f"[{vp}]\r\nv={v:.3f}\r\n" + s

    return s


def expand_afterstate(
    node: Node,
    tokens: List[List[int]],
    afterstates: List[game.Afterstate],
    state: game.SimulationState,
    pred_state: PredictState,
    params: SearchParameters
) -> tuple[AfterStateNode, float]:

    next_node = AfterStateNode(afterstates)

    v, _ = setup_node(next_node, pred_state, tokens, node.cache, params.value_weight)

    if params.visibilize_node_graph:
        next_node.state_str = sim_state_to_str(state, next_node.predicted_v, node.predicted_color)

    next_node.p[1] = next_node.predicted_color[next_node.afterstate.piece_id]
    next_node.p[0] = 1 - next_node.p[1]

    return next_node, v


def expand(
    node: Node,
    tokens: List[List[int]],
    state: game.SimulationState,
    pred_state: PredictState,
    params: SearchParameters,
    force_to_setup=False
) -> tuple[Node, float]:

    next_node = Node()
    next_node.winner = state.winner

    if not force_to_setup and next_node.winner != 0:
        if params.visibilize_node_graph:
            next_node.state_str = sim_state_to_str(state, [next_node.winner], [0.5]*8)

        return next_node, next_node.winner

    v, next_node.p = setup_node(next_node, pred_state, tokens, node.cache, params.value_weight)

    if params.visibilize_node_graph:
        next_node.state_str = sim_state_to_str(state, next_node.predicted_v, node.predicted_color)

    return next_node, v


def try_expand_checkmate(
    node: Node,
    tokens: List[List[int]],
    state: game.SimulationState,
    player: int,
    pred_state: PredictState,
    params: SearchParameters
) -> tuple[bool, Node, float]:

    action, e, escaped_id = find_checkmate(state, player, depth=params.depth_search_checkmate_leaf)

    if e > 0:
        next_node = Node()
        next_node.winner = 1

        if params.visibilize_node_graph:
            next_node.state_str = sim_state_to_str(state, [100], [0.5]*8)

        return True, next_node, 1

    if e == 0 or escaped_id == -1:
        return False, None, 0

    afterstate = game.Afterstate(game.AfterstateType.ESCAPING, escaped_id)
    next_node, v = expand_afterstate(node, tokens, [afterstate], state, pred_state, params)

    return True, next_node, v


def setup_node(
    node: NodeBase,
    pred_state: PredictState,
    tokens: list,
    cache: jnp.ndarray,
    v_weight=jnp.array([-1, -1, -1, 0, 1, 1, 1])
):
    tokens = jnp.array(tokens, dtype=jnp.uint8)
    tokens = tokens.reshape(-1, game.TOKEN_SIZE)

    for i in range(tokens.shape[0]):
        _, pi, v, c, cache = predict(pred_state.params, pred_state.model, tokens[i], cache)

    if np.isnan(c).any():
        c = np.full(shape=8, fill_value=0.5)

    node.predicted_color = c
    node.predicted_v = v
    node.cache = cache

    v = np.sum(v * v_weight)

    return jax.device_get(v), jax.device_get(pi)


def simulate_afterstate(
    node: AfterStateNode,
    state: game.SimulationState,
    player: int,
    pred_state: PredictState,
    params: SearchParameters
) -> float:
    p = 0.5 + (node.p - 0.5) * 1
    color = np.random.choice([game.RED, game.BLUE], p=p)

    tokens = state.step_afterstate(node.afterstate, color)

    if node.children[color] is None:
        if len(node.remaining_afterstates) > 0:
            child, v = expand_afterstate(node, tokens, node.remaining_afterstates, state, pred_state, params)

        elif len(tokens) == 0 or state.is_done:
            child, v = expand(node, tokens, state, pred_state, params)

        else:
            exists_checkmate, child, v = try_expand_checkmate(node, tokens, state, player, pred_state, params)

            if not exists_checkmate:
                child, v = expand(node, tokens, state, pred_state, params)

        node.children[color] = child
    else:
        child = node.children[color]

        if child.has_afterstate:
            v = simulate_afterstate(child, state, player, pred_state, params)
        else:
            v = simulate(child, state, player, pred_state, params)

    state.undo_step_afterstate(node.afterstate)

    if v is not None:
        node.n[color] += 1
        node.w[color] += v

    return v


def simulate(
    node: Node,
    state: game.SimulationState,
    player: int,
    pred_state: PredictState,
    params: SearchParameters
) -> float:

    if state.is_done:
        return state.winner

    if node.winner != 0:
        return node.winner

    node.setup_valid_actions(state, player)

    if player == 1:
        scores = node.calc_scores(player, params)
        action = np.argmax(scores)
    else:
        action = np.random.choice(range(game.ACTION_SPACE), p=node.p)

    tokens, afterstates = state.step(action, player)

    if node.children[action] is None:
        if len(afterstates) > 0:
            child, v = expand_afterstate(node, tokens, afterstates, state, pred_state, params)
        else:
            exists_checkmate, child, v = try_expand_checkmate(node, tokens, state, -player, pred_state, params)

            if not exists_checkmate:
                child, v = expand(node, tokens, state, pred_state, params)

        node.children[action] = child
    else:
        child = node.children[action]

        if child.has_afterstate:
            v = simulate_afterstate(child, state, -player, pred_state, params)
        else:
            v = simulate(child, state, -player, pred_state, params)

    state.undo_step(action, player, tokens, afterstates)

    node.n[action] += 1
    node.w[action] += v

    return v


def find_checkmate(state: game.SimulationState, player: int, depth: int):
    return checkmate_lib.find_checkmate(
        state.pieces_p, state.color_p,
        state.pieces_o, state.color_o,
        player, state.root_player, depth
    )


def create_invalid_actions(actions, state: game.SimulationState, pieces_history: np.ndarray, max_duplicates=0):
    invalid_actions = []
    exist_valid_action = False

    for a in actions:
        tokens, info = state.step(a, 1)

        pieces = state.pieces_p
        is_equals = np.all(pieces_history == pieces, axis=1)

        state.undo_step(a, 1, tokens, info)

        if np.sum(is_equals) > max_duplicates:
            invalid_actions.append(a)
        else:
            exist_valid_action = True

    if not exist_valid_action:
        del_i = np.random.randint(0, len(invalid_actions))
        del invalid_actions[del_i]

    return np.array(invalid_actions, dtype=np.int16)


def select_action_with_mcts(
    node: Node,
    state: game.SimulationState,
    pred_state: PredictState,
    params: SearchParameters,
    pieces_history: np.ndarray = None
) -> int:

    if params.visibilize_node_graph:
        node.state_str = sim_state_to_str(state, [0], [0.5]*8)

    if state.n_ply <= 8:
        node.setup_valid_actions(state, 1)
        return np.random.choice(node.valid_actions)

    action, e, escaped_id = find_checkmate(state, 1, depth=params.depth_search_checkmate_root)

    if e < 0:
        # print(f"find checkmate: ({e}, {action}, {escaped_id}), {state.pieces_o}")
        pass

    if e > 0:
        pass
        # print(f"find checkmate: ({e}, {action}, {escaped_id}), {state.pieces_o}")

    else:
        node.setup_valid_actions(state, 1)

        if pieces_history is not None:
            node.invalid_actions = create_invalid_actions(
                node.valid_actions,
                state,
                pieces_history,
                params.max_duplicates
            )
            node.apply_invalid_actions()

        if params.dirichlet_alpha is not None:
            dirichlet_noise = np.random.dirichlet(alpha=[params.dirichlet_alpha]*len(node.valid_actions))

            for a, noise in zip(node.valid_actions, dirichlet_noise):
                node.p[a] = (1 - params.dirichlet_eps) * node.p[a] + params.dirichlet_eps * noise

        if params.n_ply_to_apply_noise < state.n_ply:
            for i in range(params.num_simulations):
                sorted_n = np.sort(node.n)
                diff = sorted_n[-1] - sorted_n[-2]

                if diff > params.num_simulations - i:
                    break

                simulate(node, state, 1, pred_state, params)

            action = np.argmax(node.n)
        else:
            for i in range(params.num_simulations):
                simulate(node, state, 1, pred_state, params)

            policy = node.get_policy()
            action = np.random.choice(range(len(policy)), p=policy)

        if params.visibilize_node_graph:
            dg = Digraph(format='png')
            dg.attr('node', fontname="Myrica M")
            dg.attr('edge', fontname="Myrica M")
            node.visualize_graph(dg)
            dg.render(f'./data/graph/n_ply_{state.n_ply}')

    return action


def apply_action(
    node: Node,
    state: game.SimulationState,
    action: int,
    player: int,
    true_color_o: np.ndarray,
    pred_state: PredictState,
    params: SearchParameters
) -> tuple[Node, list[list[int]]]:

    tokens, afterstates = state.step(action, player)

    if len(afterstates) > 0:
        for i in range(len(afterstates)):
            child_afterstate, _ = expand_afterstate(node, tokens, afterstates[i:], state, pred_state, params)
            tokens_afterstate = state.step_afterstate(afterstates[i], true_color_o[afterstates[i].piece_id])
            tokens += tokens_afterstate

        child, _ = expand(child_afterstate, tokens_afterstate, state, pred_state, params, force_to_setup=True)
    else:
        child, _ = expand(node, tokens, state, pred_state, params, force_to_setup=True)

    return child, tokens


def create_memory(node: Node, pred_state: PredictState, model: TransformerWithCache) -> jnp.ndarray:
    write_memory = model.create_zero_memory()
    next_memory = []

    cache = node.cache

    for i in range(model.config.length_memory_block):
        x, _, _, _, cache = predict(
            pred_state.params, pred_state.model, write_memory[i], cache, write_memory_i=jnp.array(i)
        )
        next_memory.append(x)

    return jnp.array(next_memory)


def create_root_node(
    state: game.SimulationState,
    pred_state: PredictState,
    model: TransformerWithCache,
    cache_length: int = 220,
    prev_node: Node = None,
) -> tuple[Node, list[list[int]], np.ndarray]:
    node = Node()

    tokens = state.create_init_tokens()

    if model.has_memory_block():
        if prev_node is not None:
            memory = create_memory(prev_node, pred_state, model)
        else:
            memory = model.create_zero_memory()

        cache = model.create_cache(cache_length + model.config.length_memory_block * 2)

        for i in range(len(memory)):
            _, _, _, _, cache = predict(
                pred_state.params, pred_state.model, memory[i], cache, read_memory_i=jnp.array(i)
            )
    else:
        memory = None
        cache = model.create_cache(cache_length)

    setup_node(node, pred_state, tokens, cache)

    return node, tokens, memory


class PlayerMCTS(PlayerBase):
    def __init__(
        self,
        params,
        model: TransformerWithCache,
        search_params: SearchParameters,
    ) -> None:
        self.pred_state = PredictState(model, params)
        self.model = model
        self.search_params = search_params

        self.node: Node = None
        self.memory: np.ndarray = None

    @classmethod
    def from_config(cls, config: PlayerMCTSConfig, project_dir: str) -> "PlayerMCTS":
        ckpt = CheckpointManager(f"{project_dir}/{config.name}").load(config.step)

        return PlayerMCTS(
            params=ckpt.params,
            model=ckpt.model.create_caching_model(),
            search_params=config.mcts_params.sample()
        )

    def init_state(self, state: game.SimulationState):
        self.state = state
        self.pieces_history = []

        self.node, tokens, self.memory = create_root_node(state, self.pred_state, self.model, prev_node=self.node)
        return tokens

    def select_next_action(self) -> int:
        self.pieces_history.append(self.state.pieces_p.copy())

        action = select_action_with_mcts(
            self.node, self.state, self.pred_state, self.search_params,
            pieces_history=np.array(self.pieces_history, dtype=np.int16)
        )

        return action

    def apply_action(self, action: int, player: int, true_color_o: np.ndarray):
        self.node, tokens = apply_action(
            self.node, self.state, action, player, true_color_o, self.pred_state, self.search_params)
        return tokens


def test_mcts():
    def c(n, c_init, c_base):
        return c_init * np.log((np.sum(n) + 1 + c_base) / c_base) * np.sqrt(np.sum(n) + 1) / (n + 1)

    num_a = 10
    num_sim = 50

    p = nn.softmax(np.random.random(num_a))
    w = np.random.random(num_a) - 0.5
    n = np.zeros(num_a)

    print("p:", [round(float(p_i), 2) for p_i in p])
    print("w:", [round(p_i, 2) for p_i in w])

    score_history = np.zeros((num_sim, num_a))
    n_history = np.zeros((num_sim, num_a))

    for i in range(num_sim):
        score = w + p * c(n, c_init=1.25, c_base=25)
        a = np.argmax(score)
        n[a] += 1

        score_history[i] = score
        n_history[i] = n

    print(n)

    import matplotlib.pyplot as plt

    plt.plot(np.arange(num_sim), score_history)
    plt.savefig("test.png")


if __name__ == "__main__":
    test_mcts()
