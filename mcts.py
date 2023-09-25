from functools import partial
from typing import List, Any, Callable

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import core, struct
from flax.training import checkpoints

import geister as game
import geister_lib
from network_transformer import TransformerDecoderWithCache
from buffer import Sample

from graphviz import Digraph
from line_profiler import profile


class PredictState(struct.PyTreeNode):
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)


@partial(jax.jit, device=jax.devices("cpu")[0])
def predict(state: PredictState, tokens, cache_v, cache_k):
    pi, v, c, cv, ck = state.apply_fn({'params': state.params}, tokens, cache_v, cache_k, eval=True)

    v = nn.softmax(v)
    c = nn.sigmoid(c)

    return pi, v, c, cv, ck


should_do_visibilize_node_graph = __name__ == '__main__'


class Node:
    def __init__(self, root_player: int, weight_v: np.ndarray) -> None:
        self.root_player = root_player
        self.weight_v = weight_v

        self.has_afterstate = False
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

        self.p = np.where(self.valid_actions_mask, self.p, 0)

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


class CheckmateNode:
    def __init__(self) -> None:
        pass


class AfterStateNode:
    def __init__(self, root_player: int, weight_v: np.ndarray, afterstates: List[game.Afterstate]):
        self.root_player = root_player
        self.weight_v = weight_v

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
                      afterstates: List[game.Afterstate],
                      state: game.SimulationState,
                      pred_state: PredictState):
    next_node = AfterStateNode(node.root_player, node.weight_v, afterstates)

    v, _ = setup_node(next_node, pred_state, tokens, node.cache_v, node.cache_k)

    if should_do_visibilize_node_graph:
        next_node.state_str = sim_state_to_str(state, next_node.predicted_v)

    next_node.p[1] = next_node.predicted_color[next_node.afterstate.piece_id]
    next_node.p[0] = 1 - next_node.p[1]

    return next_node, v


@profile
def expand(node: Node,
           tokens: List[List[int]],
           state: game.SimulationState,
           pred_state: PredictState):

    next_node = Node(node.root_player, node.weight_v)
    next_node.winner = state.winner

    if next_node.winner != 0:
        if should_do_visibilize_node_graph:
            next_node.state_str = sim_state_to_str(state, [next_node.winner])

        return next_node, next_node.winner

    v, next_node.p = setup_node(next_node, pred_state, tokens, node.cache_v, node.cache_k)

    if should_do_visibilize_node_graph:
        next_node.state_str = sim_state_to_str(state, next_node.predicted_v)

    return next_node, v


def try_expand_checkmate(node: Node,
                         tokens: List[List[int]],
                         state: game.SimulationState,
                         player: int,
                         pred_state: PredictState):

    _, e, escaped_id = find_checkmate(state, player, depth=6)

    if e == 0:
        return False, None, 0

    if e < 0 and state.color_o[escaped_id] == game.RED:
        return False, None, 0

    winner = 0
    if e < 0 and state.color_o[escaped_id] == game.BLUE:
        winner = -1

    if e > 0:
        winner = 1

    if winner != 0:
        next_node = Node(node.root_player, node.weight_v)
        next_node.winner = winner

        if should_do_visibilize_node_graph:
            next_node.state_str = sim_state_to_str(state, [1])

        return True, next_node, winner

    afterstate = game.Afterstate(game.AfterstateType.ESCAPING, escaped_id)

    next_node, v = expand_afterstate(node, tokens, [afterstate], state, pred_state)

    return True, next_node, v


def setup_node(node: Node, pred_state: PredictState, tokens, cv, ck):
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

    v = np.sum(v * node.weight_v)

    return jax.device_get(v), jax.device_get(pi)


@profile
def simulate_afterstate(node: AfterStateNode,
                        state: game.SimulationState,
                        player: int,
                        pred_state: PredictState) -> float:
    color = np.random.choice([game.RED, game.BLUE], p=node.p)

    tokens = state.step_afterstate(node.afterstate, color)

    if node.children[color] is None:
        if len(node.remaining_afterstates) > 0:
            child, v = expand_afterstate(node, tokens, node.remaining_afterstates, state, pred_state)

        elif not state.is_done:
            exists_checkmate, child, v = try_expand_checkmate(node, tokens, state, player, pred_state)

            if not exists_checkmate:
                child, v = expand(node, tokens, state, pred_state)
        else:
            child, v = expand(node, tokens, state, pred_state)

        node.children[color] = child
    else:
        child = node.children[color]

        if child.has_afterstate:
            v = simulate_afterstate(child, state, player, pred_state)
        else:
            v = simulate(child, state, player, pred_state)

    state.undo_step_afterstate(node.afterstate, tokens)

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

    tokens, afterstates = state.step(action, player)

    if node.children[action] is None:
        if len(afterstates) > 0:
            child, v = expand_afterstate(node, tokens, afterstates, state, pred_state)
        else:
            exists_checkmate, child, v = try_expand_checkmate(node, tokens, state, -player, pred_state)

            if not exists_checkmate:
                child, v = expand(node, tokens, state, pred_state)

        node.children[action] = child
    else:
        child = node.children[action]

        if child.has_afterstate:
            v = simulate_afterstate(child, state, -player, pred_state)
        else:
            v = simulate(child, state, -player, pred_state)

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
                            num_sim: int,
                            alpha: float = None,
                            eps=0.25,
                            is_select_by_argmax=True,
                            pieces_history: np.ndarray = None,
                            max_duplicates=0,
                            checkmate_search_depth=7):

    if should_do_visibilize_node_graph:
        node.state_str = sim_state_to_str(state, [0])

    action, e, escaped_id = find_checkmate(state, 1, depth=checkmate_search_depth)

    if e < 0:
        print(f"find checkmate: ({e}, {action}, {escaped_id}), {state.pieces_o}")
        pass

    if e > 0:
        pass
        print(f"find checkmate: ({e}, {action}, {escaped_id}), {state.pieces_o}")

    else:
        node.setup_valid_actions(state, 1)

        if pieces_history is not None:
            node.invalid_actions = create_invalid_actions(node.valid_actions, state, pieces_history, max_duplicates)
            node.apply_invalid_actions()

        if alpha is not None:
            dirichlet_noise = np.random.dirichlet(alpha=[alpha]*len(node.valid_actions))

            for a, noise in zip(node.valid_actions, dirichlet_noise):
                node.p[a] = (1 - eps) * node.p[a] + eps * noise

        for _ in range(num_sim):
            simulate(node, state, 1, pred_state)

        policy = node.get_policy()
        if is_select_by_argmax:
            action = np.argmax(policy)
        else:
            action = np.random.choice(range(len(policy)), p=policy)

        if should_do_visibilize_node_graph:
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
                 pred_state: PredictState):

    tokens, afterstates = state.step(action, player * node.root_player)

    if len(afterstates) > 0:
        for i in range(len(afterstates)):
            child_afterstate, _ = expand_afterstate(node, tokens, afterstates[i:], state, pred_state)
            tokens_afterstate = state.step_afterstate(afterstates[i], true_color_o[afterstates[i].piece_id])
            tokens += tokens_afterstate

        child, _ = expand(child_afterstate, tokens_afterstate, state, pred_state)
    else:
        child, _ = expand(node, tokens, state, pred_state)

    return child, tokens


def create_root_node(state: game.SimulationState,
                     pred_state: PredictState,
                     model: TransformerDecoderWithCache,
                     weight_v=np.array([-1, -1, -1, 0, 1, 1, 1])) -> Node:
    node = Node(state.root_player, weight_v)
    if model.is_linear_attention:
        cv, ck = model.create_linear_cache()
    else:
        cv, ck = model.create_cache(200)

    tokens = state.create_init_tokens()

    setup_node(node, pred_state, tokens, cv, ck)

    return node, tokens


class PlayerMCTS:
    def __init__(self,
                 params,
                 model: TransformerDecoderWithCache,
                 num_mcts_sim: int,
                 dirichlet_alpha: float,
                 n_ply_to_apply_noise: int,
                 max_duplicates: int) -> None:

        self.pred_state = PredictState(model.apply, params)
        self.model = model
        self.num_mcts_sim = num_mcts_sim
        self.dirichlet_alpha = dirichlet_alpha
        self.n_ply_to_apply_noise = n_ply_to_apply_noise
        self.max_duplicates = max_duplicates
        self.tokens = []

        self.weight_v = np.array([-1, -1, -1, 0, 1, 1, 1])

    def update_params(self, params):
        self.pred_state = PredictState(self.model.apply, params)

    def init_state(self, state: game.SimulationState):
        self.state = state
        self.pieces_history = np.zeros((101, 8), dtype=np.int8)
        self.tokens = []

        self.node, tokens = create_root_node(state, self.pred_state, self.model, self.weight_v)
        self.tokens += tokens

    def select_next_action(self) -> int:
        self.pieces_history[self.state.n_ply // 2] = self.state.pieces_p

        is_select_by_argmax = self.state.n_ply > self.n_ply_to_apply_noise

        action = select_action_with_mcts(self.node, self.state, self.pred_state,
                                         num_sim=self.num_mcts_sim,
                                         alpha=self.dirichlet_alpha,
                                         is_select_by_argmax=is_select_by_argmax,
                                         pieces_history=self.pieces_history,
                                         max_duplicates=self.max_duplicates)

        return action

    def apply_action(self, action: int, player: int, true_color_o: np.ndarray):
        self.node, tokens = apply_action(self.node, self.state, action, player, true_color_o, self.pred_state)
        self.tokens += tokens

    def create_sample(self, actions: np.ndarray, true_color_o: np.ndarray) -> Sample:
        tokens = np.zeros((200, 5), dtype=np.uint8)
        tokens[:min(200, len(self.tokens))] = self.tokens[:200]

        mask = np.zeros(200, dtype=np.uint8)
        mask[:len(self.tokens)] = 1

        actions = actions[tokens[:, 4]]
        reward = 3 + int(self.state.winner * self.state.win_type.value)

        return Sample(tokens, mask, actions, reward, true_color_o)


def play_game(player1: PlayerMCTS, player2: PlayerMCTS, game_length=200, print_board=False):
    state1, state2 = game.get_initial_state_pair()
    player1.init_state(state1)
    player2.init_state(state2)

    action_history = np.zeros(game_length + 1, dtype=np.int16)

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


def init_jit(state: PredictState, model: TransformerDecoderWithCache, data):
    cv, ck = model.create_cache(0)

    for t in range(100):
        print(t)
        _, _, _, cv, ck = predict(state, data[0][0, t], cv, ck)


# should_do_visibilize_node_graph = False


def test():
    # data = [jnp.load(f"data_{i}.npy") for i in range(4)]

    model_with_cache = TransformerDecoderWithCache(num_heads=8,
                                                   embed_dim=128,
                                                   num_hidden_layers=4,
                                                   is_linear_attention=True)

    ckpt_dir = './checkpoints_backup_195/'
    prefix = 'geister_linear_'

    ckpt = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, prefix=prefix, target=None)

    # init_jit(pred_state, model_with_cache, data)

    np.random.seed(120)

    win_count = np.zeros(7)

    player1 = PlayerMCTS(ckpt['params'], model_with_cache, num_mcts_sim=100, dirichlet_alpha=0.1,
                         n_ply_to_apply_noise=0, max_duplicates=1)
    player2 = PlayerMCTS(ckpt['params'], model_with_cache, num_mcts_sim=100, dirichlet_alpha=0.1,
                         n_ply_to_apply_noise=0, max_duplicates=1)

    player1.n_ply_to_apply_noise = 0
    player2.n_ply_to_apply_noise = 0

    player1.weight_v = np.array([-1, -1, -1, 0, 1, 1, 1])
    player2.weight_v = np.array([-1, -1, -1, 0, 1, 1, 1])

    for i in range(1):
        play_game(player1, player2, game_length=200, print_board=True)

        # winner, win_type = play_test_game(pred_state, model_with_cache, 20, 0.3, print_board=False)

        # index = int(winner * win_type.value) + 3
        # win_count[index] += 1

        print(win_count)


if __name__ == "__main__":
    test()
