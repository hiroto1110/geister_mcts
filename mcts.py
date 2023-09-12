from functools import partial
from typing import List, Any, Callable
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
from line_profiler import profile


class PredictState(struct.PyTreeNode):
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)


@partial(jax.jit, device=jax.devices("cpu")[0])
def predict(state: PredictState, tokens, cache_v, cache_k):
    pi, v, c, cv, ck = state.apply_fn({'params': state.params}, tokens, cache_v, cache_k, eval=True)

    pi = pi[0, 0]
    v = nn.softmax(v[0, 0])
    c = nn.sigmoid(c[0, 0])

    return pi, v, c, cv, ck


should_do_visibilize_node_graph = __name__ == '__main__'


class Node:
    def __init__(self, root_player: int) -> None:
        self.root_player = root_player
        self.is_afterstate = False
        self.winner = 0

        self.state_str = ""

        self.c_init = 1.25
        self.c_base = 19652

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

        self.p = np.where(self.valid_actions_mask, self.p, -np.inf)
        self.p = np.array(nn.softmax(self.p))

    def setup_valid_actions(self, state, player):
        if self.valid_actions_mask is not None:
            return

        self.valid_actions_mask = game.get_valid_actions_mask(state, player)
        self.valid_actions_mask[self.invalid_actions] = 0

        self.valid_actions = np.where(self.valid_actions_mask)[0]

        self.p = np.where(self.valid_actions_mask, self.p, -np.inf)
        self.p = np.array(nn.softmax(self.p))

    def calc_scores(self, player: int):
        c = self.c_init * np.log((self.n.sum() + 1 + self.c_base) / self.c_base)

        U = c * self.p * np.sqrt(self.n.sum() + 1) / (self.n + 1)
        Q = player * self.w / np.where(self.n != 0, self.n, 1)

        scores = U + Q
        scores = np.where(self.valid_actions_mask, scores, -np.inf)

        return scores

    def get_policy(self):
        return self.n / self.n.sum()


class AfterStateNode:
    def __init__(self, root_player: int) -> None:
        self.root_player = root_player
        self.is_afterstate = True
        self.winner = 0

        self.piece_id = -1

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


def sim_state_to_str(state: game.SimulationState, predicted_v):
    board = np.zeros(36, dtype=np.int8)

    board[state.pieces_p[(state.pieces_p >= 0) & (state.color_p == game.BLUE)]] = 1
    board[state.pieces_p[(state.pieces_p >= 0) & (state.color_p == game.RED)]] = 2
    board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == game.BLUE)]] = -1
    board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == game.RED)]] = -2
    board[state.pieces_o[(state.pieces_o >= 0) & (state.color_o == game.UNCERTAIN_PIECE)]] = -3

    s = str(board.reshape((6, 6))).replace('0', ' ')
    s = s.replace('[[', ' [').replace('[', '|')
    s = s.replace(']]', ']').replace(']', '|')

    vp = str.join(",", [str(int(v * 100)) for v in predicted_v])
    v = np.sum(predicted_v * np.array([-1, -1, -1, 0, 1, 1, 1]))
    s = f"[{vp}]\r\nv={v:.3f}\r\n" + s

    n_cap_b = np.sum((state.pieces_o == game.CAPTURED) & (state.color_o == game.BLUE))
    n_cap_r = np.sum((state.pieces_o == game.CAPTURED) & (state.color_o == game.RED))

    s += f"\r\nblue={n_cap_b} red={n_cap_r}"

    return s


@profile
def expand_afterstate(node: Node,
                      tokens: List[List[int]],
                      info: game.AfterstateInfo,
                      state: game.SimulationState,
                      pred_state: PredictState):
    next_node = AfterStateNode(node.root_player)
    next_node.piece_id = info.piece_id

    v, _ = setup_node(next_node, pred_state, tokens, node.cache_v, node.cache_k)

    if should_do_visibilize_node_graph:
        next_node.state_str = sim_state_to_str(state, next_node.predicted_v)

    next_node.p[1] = next_node.predicted_color[info.piece_id]
    next_node.p[0] = 1 - next_node.p[1]

    return next_node, v


@profile
def expand(node: Node,
           tokens: List[List[int]],
           state: game.SimulationState,
           pred_state: PredictState):

    next_node = Node(node.root_player)
    next_node.winner = state.winner

    if next_node.winner != 0:
        if should_do_visibilize_node_graph:
            next_node.state_str = sim_state_to_str(state, [next_node.winner])

        return next_node, next_node.winner

    v, next_node.p = setup_node(next_node, pred_state, tokens, node.cache_v, node.cache_k)

    if should_do_visibilize_node_graph:
        next_node.state_str = sim_state_to_str(state, next_node.predicted_v)

    return next_node, v


def setup_node(node: Node, pred_state: PredictState, tokens, cv, ck):
    tokens = jnp.array(tokens, dtype=jnp.uint8)
    tokens = tokens.reshape(-1, game.TOKEN_SIZE)

    for i in range(tokens.shape[0]):
        token = tokens[i].reshape(1, 1, game.TOKEN_SIZE)
        pi, v, c, cv, ck = predict(pred_state, token, cv, ck)

    if np.isnan(c).any():
        c = np.full(shape=8, fill_value=0.5)

    node.predicted_color = c
    node.predicted_v = v
    node.cache_v = cv
    node.cache_k = ck

    v = np.sum(v * np.array([-1, -1, -1, 0, 1, 1, 1]))

    return jax.device_get(v), jax.device_get(pi)


@profile
def simulate_afterstate(node: AfterStateNode,
                        state: game.SimulationState,
                        player: int,
                        pred_state: PredictState,
                        info: List[game.AfterstateInfo]) -> float:
    color = np.random.choice([game.RED, game.BLUE], p=node.p)

    tokens = state.step_afterstate(info[0], color)

    if node.children[color] is None:
        if len(info) > 1:
            child, v = expand_afterstate(node, tokens, info[1], state, pred_state)
        else:
            child, v = expand(node, tokens, state, pred_state)

        node.children[color] = child
    else:
        if len(info) > 1:
            v = simulate_afterstate(node.children[color], state, player, pred_state, info[1:])
        else:
            v = simulate(node.children[color], state, player, pred_state)

    state.undo_step_afterstate(info[0])

    node.n[color] += 1
    node.w[color] += v

    return v


@profile
def simulate(node: Node,
             state: game.SimulationState,
             player: int,
             pred_state: PredictState) -> float:

    if state.is_done:
        return state.winner

    if node.winner != 0:
        return node.winner

    node.setup_valid_actions(state, player)

    scores = node.calc_scores(player)
    action = np.argmax(scores)

    tokens, info = state.step(action, player)

    if node.children[action] is None:
        if len(info) > 0:
            child, v = expand_afterstate(node, tokens, info[0], state, pred_state)
        else:
            child, v = expand(node, tokens, state, pred_state)

        node.children[action] = child
    else:
        if len(info) > 0:
            v = simulate_afterstate(node.children[action], state, -player, pred_state, info)
        else:
            v = simulate(node.children[action], state, -player, pred_state)

    state.undo_step(action, player, tokens, info)

    node.n[action] += 1
    node.w[action] += v

    return v


def find_checkmate(state: game.SimulationState, depth: int):
    n_cap_ob = np.sum((state.pieces_o == game.CAPTURED) & (state.color_o == game.BLUE))

    return geister_lib.find_checkmate(state.pieces_p,
                                      state.color_p,
                                      state.pieces_o,
                                      n_cap_ob,
                                      state.root_player,
                                      depth)


def create_invalid_actions(actions, state: game.SimulationState, pieces_history: np.ndarray):
    invalid_actions = []
    exist_valid_action = False

    for a in actions:
        tokens, info = state.step(a, 1)

        pieces = state.pieces_p
        is_equals = np.all(pieces_history == pieces, axis=1)

        state.undo_step(a, 1, tokens, info)

        if np.any(is_equals):
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
                            pieces_history: np.ndarray,
                            pred_state: PredictState,
                            num_sim: int,
                            alpha: float = None,
                            eps: float = 0.25):

    if should_do_visibilize_node_graph:
        node.state_str = sim_state_to_str(state, [0])

    action = find_checkmate(state, depth=4)

    if action != -1:
        pass
        # print(f"find checkmate: {action}")

    else:
        node.setup_valid_actions(state, 1)
        node.invalid_actions = create_invalid_actions(node.valid_actions, state, pieces_history)
        node.apply_invalid_actions()

        if alpha is not None:
            dirichlet_noise = np.random.dirichlet(alpha=[alpha]*len(node.valid_actions))

            for a, noise in zip(node.valid_actions, dirichlet_noise):
                node.p[a] = (1 - eps) * node.p[a] + eps * noise

        for _ in range(num_sim):
            simulate(node, state, 1, pred_state)

        policy = node.get_policy()
        action = np.argmax(policy)

        if should_do_visibilize_node_graph:
            dg = Digraph(format='png')
            dg.attr('node', fontname="Myrica M")
            dg.attr('edge', fontname="Myrica M")
            visibilize_node_graph(node, dg)
            dg.render(f'./graph/n_ply_{state.n_ply}')

    return action


def apply_action(node: Node,
                 state: game.SimulationState,
                 action: int, player: int,
                 true_color_o: np.ndarray,
                 pred_state: PredictState):

    tokens, info = state.step(action, player * node.root_player)

    if len(info) > 0:
        for i in range(len(info)):
            child_afterstate, _ = expand_afterstate(node, tokens, info[i], state, pred_state)
            tokens_afterstate = state.step_afterstate(info[i], true_color_o[info[i].piece_id])
            tokens += tokens_afterstate

        child, _ = expand(child_afterstate, tokens_afterstate, state, pred_state)
    else:
        child, _ = expand(node, tokens, state, pred_state)

    return child, tokens


def create_root_node(state: game.SimulationState,
                     pred_state: PredictState,
                     model: TransformerDecoderWithCache,
                     player: int) -> Node:
    node = Node(player)
    cv, ck = model.create_cache(1, 0)

    tokens = state.create_init_tokens()

    setup_node(node, pred_state, tokens, cv, ck)

    return node, tokens


class PlayerMCTS:
    def __init__(self,
                 pred_state: PredictState,
                 model: TransformerDecoderWithCache,
                 num_mcts_sim: int,
                 dirichlet_alpha: float) -> None:

        self.pred_state = pred_state
        self.model = model
        self.num_mcts_sim = num_mcts_sim
        self.dirichlet_alpha = dirichlet_alpha

        self.tokens = []

    def init_state(self, color: np.ndarray, player: int):
        self.player = player
        self.state = game.SimulationState(color, player)
        self.node = create_root_node(self.state, self.pred_state, self.model, self.player)

        self.pieces_history = np.zeros((110, 8), dtype=np.int8)

    def decide_next_move(self, true_color_o: np.ndarray):
        self.pieces_history[self.state.n_ply // 2] = self.state.pieces_p

        action = select_action_with_mcts(self.node, self.state, self.pieces_history,
                                         self.pred_state, self.num_mcts_sim, self.dirichlet_alpha)

        self.node, tokens = apply_action(self.node, self.state, action, self.player, true_color_o, self.pred_state)

        self.tokens += tokens

        return action

    def recieve_opponent_move(self, action: int, true_color_o: np.ndarray):
        self.node, tokens = apply_action(self.node, self.state, action, -self.player, true_color_o, self.pred_state)

        self.tokens += tokens


def play_game(pred_state: PredictState,
              model: TransformerDecoderWithCache,
              num_mcts_sim1: int, num_mcts_sim2: int,
              dirichlet_alpha: float,
              record_player: int,
              game_length: int = 200,
              print_board: bool = False):
    player = 1

    state1, state2 = game.get_initial_state_pair()
    node1, tokens1 = create_root_node(state1, pred_state, model, 1)
    node2, tokens2 = create_root_node(state2, pred_state, model, -1)

    tokens = tokens1 if record_player == 1 else tokens2

    action_history = np.zeros(game_length + 1, dtype=np.int16)

    pieces_history1 = np.zeros((100, 8), dtype=np.int8)
    pieces_history2 = np.zeros((100, 8), dtype=np.int8)

    for i in range(game_length):
        if player == 1:
            pieces_history1[i // 2] = state1.pieces_p

            action = select_action_with_mcts(node1, state1, pieces_history1,
                                             pred_state, num_mcts_sim1, dirichlet_alpha)
        else:
            pieces_history2[i // 2] = state2.pieces_p

            action = select_action_with_mcts(node2, state2, pieces_history2,
                                             pred_state, num_mcts_sim2, dirichlet_alpha)

        node1, tokens1_i = apply_action(node1, state1, action, player, state2.color_p, pred_state)
        node2, tokens2_i = apply_action(node2, state2, action, player, state1.color_p, pred_state)

        if record_player == 1:
            tokens += tokens1_i
        else:
            tokens += tokens2_i

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

    reward = int(state1.winner * state1.win_type.value * record_player)
    color = state2.color_p if record_player == 1 else state1.color_p

    return tokens, action_history, reward, color


def init_jit(state: PredictState, model: TransformerDecoderWithCache, data):
    cv, ck = model.create_cache(1, 0)

    for t in range(100):
        print(t)
        _, _, _, cv, ck = predict(state, data[0][:1, t:t+1], cv, ck)


def test():
    # data = [jnp.load(f"data_{i}.npy") for i in range(4)]

    model_with_cache = TransformerDecoderWithCache(num_heads=8, embed_dim=128, num_hidden_layers=3)

    ckpt_dir = './checkpoints/'
    prefix = 'geister_'

    ckpt = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, prefix=prefix, target=None)
    pred_state = PredictState(apply_fn=model_with_cache.apply,
                              params=ckpt['params'])

    # init_jit(pred_state, model_with_cache, data)

    np.random.seed(13)

    for i in range(1):
        start = time.perf_counter()

        tokens_ls, actions, reward, color = play_game(pred_state,
                                                      model_with_cache,
                                                      100, 100, 0.3,
                                                      record_player=1,
                                                      game_length=200,
                                                      print_board=True)

        elapsed = time.perf_counter() - start
        print(f"time: {elapsed} s")

        tokens = np.zeros((200, 5), dtype=np.uint8)
        tokens[:min(200, len(tokens_ls))] = tokens_ls[:200]

        mask = np.zeros(200, dtype=np.uint8)
        mask[:len(tokens_ls)] = 1

        actions = actions[tokens[:, 4]]

        print(tokens)
        print(mask)
        print(actions)
        print(reward + 3)
        print(color)


if __name__ == "__main__":
    test()
