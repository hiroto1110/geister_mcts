from functools import partial
from typing import List, Any, Callable
from dataclasses import dataclass, replace, field

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import core, struct

from graphviz import Digraph
from line_profiler import profile
import termcolor

import geister as game
import geister_lib
from network_transformer import TransformerDecoderWithCache
from buffer import Sample
import server_util
import naotti2020


class PredictState(struct.PyTreeNode):
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)


@partial(jax.jit, device=jax.devices("cpu")[0])
# @jax.jit
def predict(state: PredictState, tokens, cache_v, cache_k):
    pi, v, c, cv, ck = state.apply_fn({'params': state.params}, tokens, cache_v, cache_k, eval=True)

    v = nn.softmax(v)
    c = nn.sigmoid(c)

    return pi, v, c, cv, ck


@dataclass
class SearchParameters:
    num_simulations: int
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    n_ply_to_apply_noise: int = 20
    max_duplicates: int = 3
    c_init: float = 1.25
    c_base: int = 19652
    depth_search_checkmate_root: int = 7
    depth_search_checkmate_leaf: int = 4
    v_weight: np.ndarray = field(default_factory=lambda: np.array([-1, -1, -1, 0, 1, 1, 1]))
    should_do_visibilize_node_graph: bool = False

    def replace(self, **args):
        return replace(self, **args)


class Node:
    def __init__(self) -> None:
        self.has_afterstate = False
        self.winner = 0

        self.state_str = ""

        self.cache_v = None
        self.cache_k = None
        self.valid_actions_mask = None
        self.invalid_actions = np.zeros(shape=0, dtype=np.uint8)

        self.predicted_color = None
        self.predicted_v = 0

        self.children = [None] * game.ACTION_SPACE
        self.p = np.zeros(game.ACTION_SPACE)
        self.w = np.zeros(game.ACTION_SPACE)
        self.n = np.zeros(game.ACTION_SPACE, dtype=np.int16)

    def apply_invalid_actions(self):
        if self.valid_actions_mask is None:
            return

        self.valid_actions_mask[self.invalid_actions] = 0

        # self.valid_actions = np.where(self.valid_actions_mask)[0]

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


class AfterStateNode:
    def __init__(self, afterstates: List[game.Afterstate]):
        self.afterstate = afterstates[0]
        self.remaining_afterstates = afterstates[1:]
        self.has_afterstate = True

        self.winner = 0

        self.state_str = ""

        self.cache_v = None
        self.cache_k = None

        self.predicted_color = None
        self.predicted_v = 0

        self.children = [None] * 2
        self.p = np.zeros(2)
        self.w = np.zeros(2)
        self.n = np.zeros(2, dtype=np.int16)


def visibilize_node_graph(node: Node, g: Digraph):
    if isinstance(node, Node):
        for child, w, n, p in zip(node.children, node.w, node.n, node.p):
            if child is None:
                continue

            v = w / n if n > 0 else 0
            label = f"w = {v:.3f}\r\np = {p:.3f}"
            g.edge(node.state_str, child.state_str, label)

            visibilize_node_graph(child, g)

    else:
        for i in range(2):
            child = node.children[i]

            if child is None:
                continue

            color = 'blue' if i == 1 else 'red'

            v = node.w[i] / node.n[i] if node.n[i] > 0 else 0
            p = node.p[i]
            label = f"{color}\r\nw = {v:.3f}\r\np = {p:.3f}"
            g.edge(node.state_str, child.state_str, label=label)

            visibilize_node_graph(child, g)


def sim_state_to_str(state: game.SimulationState, v, color: np.ndarray):
    s = state_to_str(state, color)

    vp = str.join(",", [str(int(v_i * 100)) for v_i in v])
    v = np.sum(v * np.array([-1, -1, -1, 0, 1, 1, 1]))
    s = f"[{vp}]\r\nv={v:.3f}\r\n" + s

    return s


def state_to_str(state: game.SimulationState, predicted_color: np.ndarray, colored=False) -> str:
    color_int = (np.array(predicted_color) * 10).astype(dtype=np.int16)
    color_int = np.clip(color_int, 0, 9)

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
        if pos != game.CAPTURED:
            if color == game.BLUE:
                line[pos] = B_str
            else:
                line[pos] = R_str

        pos = state.pieces_o[i]
        color = state.color_o[i]
        if pos != game.CAPTURED:
            if color == game.BLUE:
                line[pos] = b_str
            elif color == game.RED:
                line[pos] = r_str
            else:
                line[pos] = str(color_int[i])

    lines = ["|" + "  ".join(line[i*6: (i+1)*6]) + "|" for i in range(6)]

    s = "\r\n".join(lines)

    n_cap_b = np.sum((state.pieces_o == game.CAPTURED) & (state.color_o == game.BLUE))
    n_cap_r = np.sum((state.pieces_o == game.CAPTURED) & (state.color_o == game.RED))

    s += f"\r\nblue={n_cap_b} red={n_cap_r}"

    return s


@profile
def expand_afterstate(node: Node,
                      tokens: List[List[int]],
                      afterstates: List[game.Afterstate],
                      state: game.SimulationState,
                      pred_state: PredictState,
                      params: SearchParameters):
    next_node = AfterStateNode(afterstates)

    v, _ = setup_node(next_node, pred_state, tokens, node.cache_v, node.cache_k, params.v_weight)

    if params.should_do_visibilize_node_graph:
        next_node.state_str = sim_state_to_str(state, next_node.predicted_v, node.predicted_color)

    next_node.p[1] = next_node.predicted_color[next_node.afterstate.piece_id]
    next_node.p[0] = 1 - next_node.p[1]

    return next_node, v


@profile
def expand(node: Node,
           tokens: List[List[int]],
           state: game.SimulationState,
           pred_state: PredictState,
           params: SearchParameters):

    next_node = Node()
    next_node.winner = state.winner

    if next_node.winner != 0:
        if params.should_do_visibilize_node_graph:
            next_node.state_str = sim_state_to_str(state, [next_node.winner], [0.5]*8)

        return next_node, next_node.winner

    v, next_node.p = setup_node(next_node, pred_state, tokens, node.cache_v, node.cache_k, params.v_weight)

    if params.should_do_visibilize_node_graph:
        next_node.state_str = sim_state_to_str(state, next_node.predicted_v, node.predicted_color)

    return next_node, v


def try_expand_checkmate(node: Node,
                         tokens: List[List[int]],
                         state: game.SimulationState,
                         player: int,
                         pred_state: PredictState,
                         params: SearchParameters):

    action, e, escaped_id = find_checkmate(state, player, depth=params.depth_search_checkmate_leaf)

    if e > 0:
        next_node = Node()
        next_node.winner = 1

        if params.should_do_visibilize_node_graph:
            next_node.state_str = sim_state_to_str(state, [100], [0.5]*8)

        return True, next_node, 1

    if e == 0 or escaped_id == -1:
        return False, None, 0

    afterstate = game.Afterstate(game.AfterstateType.ESCAPING, escaped_id)
    next_node, v = expand_afterstate(node, tokens, [afterstate], state, pred_state, params)

    return True, next_node, v


def setup_node(node: Node, pred_state: PredictState, tokens, cv, ck, v_weight):
    tokens = jnp.array(tokens, dtype=jnp.uint8)
    tokens = tokens.reshape(-1, game.TOKEN_SIZE)

    for i in range(tokens.shape[0]):
        pi, v, c, cv, ck = predict(pred_state, tokens[i], cv, ck)

    if np.isnan(c).any():
        c = np.full(shape=8, fill_value=0.5)

    node.predicted_color = c
    node.predicted_v = v
    node.cache_v = cv
    node.cache_k = ck

    v = np.sum(v * v_weight)

    return jax.device_get(v), jax.device_get(pi)


@profile
def simulate_afterstate(node: AfterStateNode,
                        state: game.SimulationState,
                        player: int,
                        pred_state: PredictState,
                        params: SearchParameters) -> float:
    color = np.random.choice([game.RED, game.BLUE], p=node.p)

    tokens = state.step_afterstate(node.afterstate, color)

    if node.children[color] is None:
        if len(node.remaining_afterstates) > 0:
            child, v = expand_afterstate(node, tokens, node.remaining_afterstates, state, pred_state, params)

        elif not state.is_done:
            exists_checkmate, child, v = try_expand_checkmate(node, tokens, state, player, pred_state, params)

            if not exists_checkmate:
                child, v = expand(node, tokens, state, pred_state, params)
        else:
            child, v = expand(node, tokens, state, pred_state, params)

        node.children[color] = child
    else:
        child = node.children[color]

        if child.has_afterstate:
            v = simulate_afterstate(child, state, player, pred_state, params)
        else:
            v = simulate(child, state, player, pred_state, params)

    state.undo_step_afterstate(node.afterstate)

    node.n[color] += 1
    node.w[color] += v

    return v


@profile
def simulate(node: Node,
             state: game.SimulationState,
             player: int,
             pred_state: PredictState,
             params: SearchParameters) -> float:

    if state.is_done:
        return state.winner

    if node.winner != 0:
        return node.winner

    node.setup_valid_actions(state, player)

    scores = node.calc_scores(player, params)
    action = np.argmax(scores)

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
    return geister_lib.find_checkmate(state.pieces_p,
                                      state.color_p,
                                      state.pieces_o,
                                      state.color_o,
                                      player,
                                      state.root_player,
                                      depth)


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


@profile
def select_action_with_mcts(node: Node,
                            state: game.SimulationState,
                            pred_state: PredictState,
                            params: SearchParameters,
                            pieces_history: np.ndarray = None):

    if params.should_do_visibilize_node_graph:
        node.state_str = sim_state_to_str(state, [0], [0.5]*8)

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
            node.invalid_actions = create_invalid_actions(node.valid_actions, state,
                                                          pieces_history,
                                                          params.max_duplicates)
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

        if params.should_do_visibilize_node_graph:
            dg = Digraph(format='png')
            dg.attr('node', fontname="Myrica M")
            dg.attr('edge', fontname="Myrica M")
            visibilize_node_graph(node, dg)
            dg.render(f'./graph/n_ply_{state.n_ply}')

    return action


def apply_action(node: Node,
                 state: game.SimulationState,
                 action: int,
                 player: int,
                 true_color_o: np.ndarray,
                 pred_state: PredictState,
                 params: SearchParameters):

    tokens, afterstates = state.step(action, player * state.root_player)

    if len(afterstates) > 0:
        for i in range(len(afterstates)):
            child_afterstate, _ = expand_afterstate(node, tokens, afterstates[i:], state, pred_state, params)
            tokens_afterstate = state.step_afterstate(afterstates[i], true_color_o[afterstates[i].piece_id])
            tokens += tokens_afterstate

        child, _ = expand(child_afterstate, tokens_afterstate, state, pred_state, params)
    else:
        child, _ = expand(node, tokens, state, pred_state, params)

    return child, tokens


def create_root_node(state: game.SimulationState,
                     pred_state: PredictState,
                     model: TransformerDecoderWithCache) -> Node:
    node = Node()
    if model.is_linear_attention:
        cv, ck = model.create_linear_cache()
    else:
        cv, ck = model.create_cache(100)

    tokens = state.create_init_tokens()

    setup_node(node, pred_state, tokens, cv, ck, 1)

    return node, tokens


class PlayerMCTS:
    def __init__(self,
                 params,
                 model: TransformerDecoderWithCache,
                 search_params: SearchParameters) -> None:

        self.pred_state = PredictState(model.apply, params)
        self.model = model
        self.search_params = search_params
        self.tokens = []

    def update_params(self, params):
        self.pred_state = PredictState(self.model.apply, params)

    def init_state(self, state: game.SimulationState):
        self.state = state
        self.pieces_history = np.zeros((101, 8), dtype=np.int8)
        self.tokens = []

        self.node, tokens = create_root_node(state, self.pred_state, self.model)
        self.tokens += tokens

    def select_next_action(self) -> int:
        self.pieces_history[self.state.n_ply // 2] = self.state.pieces_p

        action = select_action_with_mcts(self.node, self.state, self.pred_state, self.search_params,
                                         pieces_history=self.pieces_history)

        return action

    def apply_action(self, action: int, player: int, true_color_o: np.ndarray):
        self.node, tokens = apply_action(
            self.node, self.state, action, player, true_color_o, self.pred_state, self.search_params)
        self.tokens += tokens

    def create_sample(self, actions: np.ndarray, true_color_o: np.ndarray) -> Sample:
        tokens = np.zeros((200, 5), dtype=np.uint8)
        tokens[:min(200, len(self.tokens))] = self.tokens[:200]

        actions = actions[tokens[:, 4]]
        reward = 3 + int(self.state.winner * self.state.win_type.value)

        return Sample(tokens, actions, reward, true_color_o)


class PlayerNaotti2020:
    def __init__(self, depth_min, depth_max, print_log=False) -> None:
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.print_log = print_log

    def init_state(self, state: game.SimulationState):
        self.state = state
        self.turn_count = 0

        depth = np.random.randint(self.depth_min, self.depth_max + 1)
        naotti2020.initGame(depth, self.print_log)

    def select_next_action(self) -> int:
        board_msg = server_util.encode_board_str(self.state)
        naotti2020.recvBoard(board_msg)

        action_msg = naotti2020.solve(self.turn_count)
        action = server_util.decode_action_message(action_msg)

        if self.state.root_player == 1:
            p_id = action // 4
            d_id = action % 4

            action = p_id * 4 + (3 - d_id)

        self.turn_count += 2

        return action

    def apply_action(self, action: int, player: int, true_color_o: np.ndarray):
        _, afterstates = self.state.step(action, player * self.state.root_player)

        for i in range(len(afterstates)):
            _ = self.state.step_afterstate(afterstates[i], true_color_o[afterstates[i].piece_id])


def play_game(player1: PlayerMCTS, player2: PlayerMCTS, game_length=180, print_board=False):
    state1, state2 = game.get_initial_state_pair()
    player1.init_state(state1)
    player2.init_state(state2)

    action_history = np.zeros(200, dtype=np.int16)

    player = 1

    for i in range(game_length):
        if player == 1:
            action = player1.select_next_action()
        else:
            action = player2.select_next_action()

        player1.apply_action(action, player, state2.color_p)
        player2.apply_action(action, player, state1.color_p)

        action_history[i] = action

        if print_board:
            board = np.zeros(36, dtype=np.int8)

            board[state1.pieces_p[(state1.pieces_p >= 0) & (state1.color_p == 1)]] = 1
            board[state1.pieces_p[(state1.pieces_p >= 0) & (state1.color_p == 0)]] = 2
            board[state2.pieces_p[(state2.pieces_p >= 0) & (state2.color_p == 1)]] = -1
            board[state2.pieces_p[(state2.pieces_p >= 0) & (state2.color_p == 0)]] = -2

            print(str(board.reshape((6, 6))).replace('0', ' '))
            print(i)

        if state1.is_done or state2.is_done:
            break

        player = -player

    return action_history, state1.color_p, state2.color_p


def test():
    import orbax.checkpoint

    ckpt_dir = './checkpoints/fresh-terrain-288'
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer)

    ckpt = checkpoint_manager.restore(1330)

    model = TransformerDecoderWithCache(**ckpt['model'])
    params = ckpt['state']['params']

    np.random.seed(12)

    mcts_params = SearchParameters(num_simulations=100,
                                   dirichlet_alpha=0.1,
                                   n_ply_to_apply_noise=0,
                                   max_duplicates=8,
                                   depth_search_checkmate_leaf=5,
                                   depth_search_checkmate_root=9,
                                   should_do_visibilize_node_graph=False)

    player1 = PlayerMCTS(params, model, mcts_params)
    player2 = PlayerMCTS(params, model, mcts_params)

    player1 = PlayerNaotti2020(6, 6)

    game_result = [0, 0, 0]

    for i in range(100):
        if True or i % 2 == 0:
            play_game(player1, player2, game_length=180, print_board=True)
        else:
            play_game(player2, player1, game_length=180, print_board=True)

        game_result[player2.state.winner + 1] += 1
        print(game_result, game_result[2] / sum(game_result))


def test_ab():
    pieces_p = np.array([17, 19, 23, 25, 28, 30, -1, -1])
    colors_p = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    pieces_o = np.array([2, 4, 11, 18, 31, -1, -1, -1])
    colors_o = np.array([1, 1, 0, 1, 1, 0, 0, 0])

    pieces_p[pieces_p > 5] -= 6
    pieces_o[pieces_o > 5] -= 6

    state = game.SimulationState(colors_p, -1)
    state.pieces_p[:] = pieces_p
    state.pieces_o[:] = pieces_o
    state.color_p[:] = colors_p
    state.color_o[pieces_o == game.CAPTURED] = colors_o[pieces_o == game.CAPTURED]

    print(state_to_str(state, [0.5]*8, colored=True))
    print()

    _ = find_checkmate(state, -1, 6)

    import time

    n = 100
    start = time.perf_counter()

    for i in range(n):
        action, e, escaped_id = find_checkmate(state, -1, 6)
        print(i, action, e, escaped_id)

    t = time.perf_counter() - start
    print(t / n)


if __name__ == "__main__":
    test()
