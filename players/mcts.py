from functools import partial
from typing import List, Any
from dataclasses import dataclass
import time

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from graphviz import Digraph

import env.state as game
import env.lib.checkmate_lib as checkmate_lib

import game_analytics
from network.transformer import TransformerWithCache
from network.checkpoints import CheckpointManager

from players.base import PlayerBase, ActionSelectionResult
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


@dataclass
class PredictResult:
    action: jnp.ndarray
    values: jnp.ndarray
    color: jnp.ndarray
    cache: jnp.ndarray

    def get_value(self, weight: jnp.ndarray) -> float:
        return jnp.sum(self.values * weight)


def predict_with_tokens(pred_state: PredictState, tokens: list, cache: jnp.ndarray) -> PredictResult:
    x = jnp.array(tokens, dtype=jnp.uint8)
    x = x.reshape(-1, game.TOKEN_SIZE)

    for i in range(x.shape[0]):
        _, p, v, c, cache = predict(pred_state.params, pred_state.model, x[i], cache)

    return PredictResult(p, v, c, cache)


class NodeBase:
    def __init__(self, state: game.State, action_space: int) -> None:
        self.state = state

        self.children: list[NodeBase] = [None] * action_space
        self.p = np.zeros(action_space)
        self.w = np.zeros(action_space)
        self.n = np.zeros(action_space, dtype=np.int16)

    def simulate(
        self,
        state: game.State,
        player: int,
        pred_state: PredictState,
        params: SearchParameters
    ) -> float:
        pass

    def get_node_str(self) -> str:
        pass

    def visualize_graph(self, g: Digraph):
        pass


class EndNode(NodeBase):
    def __init__(self, state: game.State, winner: int, win_type: game.WinType) -> None:
        super().__init__(state, action_space=0)

        self.winner = winner
        self.win_type = win_type

    def simulate(self, *args, **kwargs) -> float:
        return self.winner

    def get_node_str(self) -> str:
        return game_analytics.state_to_str(self.state, predicted_color=[0.5]*8, colored=False)


class Node(NodeBase):
    def __init__(self, state: game.State, pred_result: PredictResult) -> None:
        super().__init__(state, action_space=game.ACTION_SPACE)

        self.predic_result = pred_result

        self.p = pred_result.action

        self.valid_actions_mask = None
        self.invalid_actions = np.zeros(shape=0, dtype=np.uint8)

    def simulate(
        self,
        state: game.State,
        player: int,
        pred_state: PredictState,
        params: SearchParameters
    ) -> float:
        self.setup_valid_actions(state, player)

        if player == 1:
            scores = self.calc_scores(player, params)
            action = np.argmax(scores)
        else:
            action = np.random.choice(range(game.ACTION_SPACE), p=self.p)

        state, result = state.step(action, player)

        if self.children[action] is None:
            if result.winner != 0:
                child = EndNode(state, result.winner, result.win_type)
                v = result.winner

            elif len(result.afterstates) > 0:
                child, v = expand(self, result.tokens, result.afterstates, state, pred_state, params)
            else:
                is_checkmate, child, v = try_expand_checkmate(self, result.tokens, state, -player, pred_state, params)

                if not is_checkmate:
                    child, v = expand(self, result.tokens, [], state, pred_state, params)

            self.children[action] = child
        else:
            child = self.children[action]
            v = child.simulate(state, -player, pred_state, params)

        self.n[action] += 1
        self.w[action] += v

        return v

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

    def get_node_str(self) -> str:
        return sim_state_to_str(self.state, self.predic_result.values, self.predic_result.color)

    def visualize_graph(self, g: Digraph):
        for child, w, n, p in zip(self.children, self.w, self.n, self.p):
            if child is None:
                continue

            v = w / n if n > 0 else 0
            label = f"w = {v:.3f}\r\np = {p:.3f}"
            g.edge(self.get_node_str(), child.get_node_str(), label=label)

            child.visualize_graph(g)


class AfterStateNode(NodeBase):
    def __init__(self, state: game.State, pred_result: PredictResult, afterstates: List[game.Afterstate]):
        super().__init__(state, action_space=2)

        self.predic_result = pred_result

        self.afterstate = afterstates[0]
        self.remaining_afterstates = afterstates[1:]

        self.p[1] = pred_result.color[self.afterstate.piece_id]
        self.p[0] = 1 - self.p[1]

    def simulate(
        self,
        state: game.State,
        player: int,
        pred_state: PredictState,
        params: SearchParameters
    ) -> float:
        p = 0.5 + (self.p - 0.5) * 1
        color: int = np.random.choice([game.RED, game.BLUE], p=p)

        state, result = state.step_afterstate(self.afterstate, color)

        if self.children[color] is None:
            if result.winner != 0:
                child = EndNode(state, result.winner, result.win_type)
                v = result.winner

            elif len(self.remaining_afterstates) > 0:
                child, v = expand(self, result.tokens, self.remaining_afterstates, state, pred_state, params)

            elif len(result.tokens) == 0:
                child, v = expand(self, result.tokens, [], state, pred_state, params)

            else:
                is_checkmate, child, v = try_expand_checkmate(self, result.tokens, state, player, pred_state, params)

                if not is_checkmate:
                    child, v = expand(self, result.tokens, [], state, pred_state, params)

            self.children[color] = child
        else:
            child = self.children[color]
            v = child.simulate(state, player, pred_state, params)

        self.n[color] += 1
        self.w[color] += v

        return v

    def get_node_str(self) -> str:
        return sim_state_to_str(self.state, self.predic_result.values, self.predic_result.color)

    def visualize_graph(self, g: Digraph):
        for i in range(len(self.children)):
            child = self.children[i]

            if child is None:
                continue

            color = 'blue' if i == 1 else 'red'

            v = self.w[i] / self.n[i] if self.n[i] > 0 else 0
            p = self.p[i]
            label = f"{color}\r\nw = {v:.3f}\r\np = {p:.3f}"
            g.edge(self.get_node_str(), child.get_node_str(), label=label)

            child.visualize_graph(g)


def sim_state_to_str(state: game.State, v, color: np.ndarray):
    s = game_analytics.state_to_str(state, color, colored=False)

    vp = str.join(",", [str(int(v_i * 100)) for v_i in v])
    v = np.sum(v * np.array([-1, -1, -1, 0, 1, 1, 1]))
    s = f"[{vp}]\r\nv={v:.3f}\r\n" + s

    return s


def expand(
    node: Node,
    tokens: List[List[int]],
    afterstates: List[game.Afterstate],
    state: game.State,
    pred_state: PredictState,
    params: SearchParameters
) -> tuple[Node, float]:

    if len(tokens) > 0:
        result = predict_with_tokens(pred_state, tokens, node.predic_result.cache)
    else:
        result = node.predic_result

    if len(afterstates) > 0:
        next_node = AfterStateNode(state, result, afterstates)
    else:
        next_node = Node(state, result)

    return next_node, result.get_value(params.value_weight)


def try_expand_checkmate(
    node: Node,
    tokens: List[List[int]],
    state: game.State,
    player: int,
    pred_state: PredictState,
    params: SearchParameters
) -> tuple[bool, Node, float]:

    action, e, escaped_id = find_checkmate(state, player, depth=params.depth_search_checkmate_leaf)

    if e > 0:
        next_node = EndNode(state, winner=1, win_type=game.WinType.ESCAPE)
        return True, next_node, 1

    if e == 0 or escaped_id == -1:
        return False, None, 0

    afterstate = game.Afterstate(game.AfterstateType.ESCAPING, escaped_id)
    next_node, v = expand(node, tokens, [afterstate], state, pred_state, params)

    return True, next_node, v


def find_checkmate(state: game.State, player: int, depth: int) -> tuple[int, int, int]:
    return checkmate_lib.find_checkmate(
        state.board[game.POS_P], state.board[game.COL_P],
        state.board[game.POS_O], state.board[game.COL_O],
        player, 1, depth
    )


def create_invalid_actions(actions, state: game.State, pieces_history: np.ndarray, max_duplicates=0):
    invalid_actions = []
    exist_valid_action = False

    for a in actions:
        state_a, _ = state.step(a, 1)

        pieces = state_a.board[:2].reshape(-1)
        is_equals = np.all(pieces_history == pieces, axis=1)

        if np.sum(is_equals) > max_duplicates:
            invalid_actions.append(a)
        else:
            exist_valid_action = True

    if not exist_valid_action:
        del_i = np.random.randint(0, len(invalid_actions))
        del invalid_actions[del_i]

    return np.array(invalid_actions, dtype=np.int16)


@dataclass
class ActionSelectionResultMCTS(ActionSelectionResult):
    num_simulations: int
    checkmate_action: int
    checkmate_eval: int
    checkmate_escaped_id: int


def select_action_with_mcts(
    node: Node,
    state: game.State,
    pred_state: PredictState,
    params: SearchParameters,
    pieces_history: np.ndarray = None
) -> ActionSelectionResultMCTS:

    start_t = time.perf_counter()

    if state.n_ply <= 6:
        node.setup_valid_actions(state, 1)
        return ActionSelectionResultMCTS(np.random.choice(node.valid_actions), 0, -1, 0, -1)

    action, e, escaped_id = find_checkmate(state, 1, depth=params.depth_search_checkmate_root)

    if e > 0:
        return ActionSelectionResultMCTS(action, 0, action, e, escaped_id)
        # print(f"find checkmate: ({e}, {action}, {escaped_id}), {state.pieces_o}")

    if time.perf_counter() - start_t > params.time_limit:
        node.setup_valid_actions(state, 1)
        return ActionSelectionResultMCTS(np.random.choice(node.valid_actions), 0, action, e, escaped_id)

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
            if time.perf_counter() - start_t > params.time_limit:
                break

            sorted_n = np.sort(node.n)
            diff = sorted_n[-1] - sorted_n[-2]

            if diff > params.num_simulations - i:
                break

            node.simulate(state, 1, pred_state, params)

        action = np.argmax(node.n)
    else:
        for i in range(params.num_simulations):
            if time.perf_counter() - start_t > params.time_limit:
                break
            node.simulate(state, 1, pred_state, params)

        policy = node.get_policy()
        action = np.random.choice(range(len(policy)), p=policy)

    return ActionSelectionResultMCTS(action, i, action, e, escaped_id)


def create_memory(node: Node, pred_state: PredictState) -> jnp.ndarray:
    write_memory = pred_state.model.create_zero_memory()
    next_memory = []

    cache = node.predic_result.cache

    for i in range(pred_state.model.config.length_memory_block):
        x, _, _, _, cache = predict(
            pred_state.params, pred_state.model, write_memory[i], cache, write_memory_i=jnp.array(i)
        )
        next_memory.append(x)

    return jnp.array(next_memory)


def create_root_node(
    tokens: list[list[int]],
    pred_state: PredictState,
    cache_length: int = 220,
    prev_node: Node = None,
) -> tuple[PredictResult, np.ndarray]:
    model = pred_state.model

    if model.has_memory_block():
        if prev_node is not None:
            memory = create_memory(prev_node, pred_state)
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

    result = predict_with_tokens(pred_state, tokens, cache)

    return result, memory


@dataclass(frozen=True)
class PlayerStateMCTS:
    node: Node
    memory: jnp.ndarray
    pieces_history: list[np.ndarray]


@dataclass(frozen=True)
class PlayerMCTS(PlayerBase[PlayerStateMCTS, ActionSelectionResultMCTS]):
    params: dict
    model: TransformerWithCache
    mcts_params: SearchParameters

    @classmethod
    def from_config(cls, config: PlayerMCTSConfig, project_dir: str) -> "PlayerMCTS":
        ckpt = CheckpointManager(f"{project_dir}/{config.name}").load(config.step)

        return PlayerMCTS(
            params=ckpt.params,
            model=ckpt.model.create_caching_model(),
            mcts_params=config.mcts_params.sample()
        )

    def get_pred_state(self) -> PredictState:
        return PredictState(self.model, self.params)

    def init_state(self, state: game.State, prev_state: PlayerStateMCTS = None) -> PlayerStateMCTS:
        tokens = state.create_init_tokens()

        result, memory = create_root_node(tokens, self.get_pred_state(), prev_node=prev_state)

        return PlayerStateMCTS(Node(state, result), memory, []), tokens

    def select_next_action(self, state: game.State, player_state: PlayerStateMCTS) -> ActionSelectionResultMCTS:
        result = select_action_with_mcts(
            player_state.node, state, self.get_pred_state(), self.mcts_params,
            pieces_history=np.array(player_state.pieces_history, dtype=np.int16)
        )

        return result

    def apply_action(
        self,
        state: game.State,
        player_state: PlayerStateMCTS,
        action: int, player: int,
        true_color_o: np.ndarray
    ) -> tuple[PlayerStateMCTS, game.State, list[list[int]], game.StepResult]:
        if player == -1:
            action = 31 - action

        pred_state = self.get_pred_state()
        node = player_state.node

        state, result = state.step(action, player)

        tokens = result.tokens
        afterstates = result.afterstates

        for i in range(len(afterstates)):
            node, _ = expand(node, result.tokens, afterstates[i:], state, pred_state, self.mcts_params)
            state, result = state.step_afterstate(afterstates[i], true_color_o[afterstates[i].piece_id])
            tokens += result.tokens

        node, _ = expand(node, result.tokens, [], state, pred_state, self.mcts_params)

        pieces_history = player_state.pieces_history + [state.board[:2].reshape(-1)]
        next_player_state = PlayerStateMCTS(node, player_state.memory, pieces_history)

        return next_player_state, state, tokens, result

    def visualize_state(self, player_state: PlayerStateMCTS, output_file: str):
        if all([c is None for c in player_state.node.children]):
            return

        dg = Digraph(format='png')
        dg.attr('node', fontname="Myrica M")
        dg.attr('edge', fontname="Myrica M")
        player_state.node.visualize_graph(dg)
        dg.render(output_file)


def test_play_game():
    np.random.seed(4)

    mcts_params = SearchParameters(
        num_simulations=100,
        time_limit=20
    )
    from network.checkpoints import Checkpoint
    ckpt = Checkpoint.from_json_file("./data/projects/run-7/main/600.json")

    player1 = PlayerMCTS(ckpt.params, ckpt.model.create_caching_model(), mcts_params)
    player2 = PlayerMCTS(ckpt.params, ckpt.model.create_caching_model(), mcts_params)

    from players.base import play_game

    result = play_game(player1, player2, print_board=True, visualization_directory="./data/graph")

    for t1, t2 in zip(result.tokens1, result.tokens2):
        print(t1, t2)

    model = ckpt.model.create_caching_model()
    cache = model.create_cache(240)

    memory = model.create_zero_memory()

    for i in range(len(memory)):
        _, _, _, _, cache = predict(
            ckpt.params, model, memory[i], cache, read_memory_i=jnp.array(i)
        )

    cache1 = cache
    cache2 = cache

    tokens1 = result.tokens1
    tokens2 = result.tokens2

    for t1, t2 in zip(tokens1, tokens2):
        _, p, v1, c, cache1 = predict(ckpt.params, model, jnp.array(t1), cache1)
        _, p, v2, c, cache2 = predict(ckpt.params, model, jnp.array(t2), cache2)
        print(np.array(t1), np.array(t2), np.array(v1 * 100, dtype=np.int16), np.array(v2 * 100, dtype=np.int16))


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
    test_play_game()
