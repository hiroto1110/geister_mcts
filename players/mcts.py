from functools import partial
from typing import Any, Literal
from dataclasses import dataclass
import time

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from graphviz import Digraph

import env.state as game
from env.checkmate import find_checkmate

import game_analytics
from network.transformer import TransformerWithCache
from network.checkpoints import CheckpointManager

from players.base import PlayerBase, ActionSelectionResult, PlayerConfig, GameResult
from players.config import SearchParameters, SearchParametersRange
from players.strategy import Strategy, StateWithStrategy, StrategyFactories, StrategyTokenProducer

import batch


@dataclass
class PredictState:
    model: TransformerWithCache
    params: dict[str, Any]


@partial(jax.jit, device=jax.devices("cpu")[0], static_argnames=["model"])
def predict(
    params: dict[str, Any],
    model: TransformerWithCache,
    x: jnp.ndarray,
    cache: jnp.ndarray
):
    x, p, v, c, cache = model.apply(
        {'params': params},
        x, cache, eval=True
    )

    v = nn.softmax(v)
    c = nn.sigmoid(c)

    return x, p, v, c, cache


@dataclass
class PredictResult:
    action: jnp.ndarray
    values: jnp.ndarray
    color: jnp.ndarray
    cache: jnp.ndarray

    def get_value(self, weight: jnp.ndarray) -> float:
        return jnp.sum(self.values * weight)


def predict_with_strategy(pred_state: PredictState, strategy: jnp.ndarray, cache: jnp.ndarray) -> PredictResult:
    _, p, v, c, cache = predict(pred_state.params, pred_state.model, strategy, cache)

    return PredictResult(p, v, c, cache)


def predict_with_tokens(pred_state: PredictState, tokens: list[list[int]], cache: jnp.ndarray) -> PredictResult:
    x = jnp.array(tokens, dtype=jnp.uint8)
    x = x.reshape(-1, x.shape[-1])

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
    def __init__(self, state: game.State, pred_result: PredictResult, afterstates: list[game.Afterstate]):
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
    tokens: list[list[int]],
    afterstates: list[game.Afterstate],
    state: game.State,
    pred_state: PredictState,
    params: SearchParameters
) -> tuple[Node, float]:

    if len(tokens) > 0:
        result = predict_with_tokens(pred_state, tokens, node.predic_result.cache)

        if params.test_c:
            result.color = node.predic_result.color

    else:
        result = node.predic_result

    if len(afterstates) > 0:
        next_node = AfterStateNode(state, result, afterstates)
    else:
        next_node = Node(state, result)

    return next_node, result.get_value(params.value_weight)


def try_expand_checkmate(
    node: Node,
    tokens: list[list[int]],
    state: game.State,
    player: int,
    pred_state: PredictState,
    params: SearchParameters
) -> tuple[bool, Node, float]:
    result = find_checkmate(state, player, depth=params.depth_search_checkmate_leaf)

    if result.eval > 0:
        next_node = EndNode(state, winner=1, win_type=game.WinType.ESCAPE)
        return True, next_node, 1

    if result.eval == 0 or result.escaped_id == -1:
        return False, None, 0

    afterstate = game.Afterstate(game.AfterstateType.ESCAPING, result.escaped_id)
    next_node, v = expand(node, tokens, [afterstate], state, pred_state, params)

    return True, next_node, v


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
    pieces_history: np.ndarray = None,
    actions: list[int] = None
) -> ActionSelectionResultMCTS:
    start_t = time.perf_counter()

    if state.n_ply <= 6:
        node.setup_valid_actions(state, 1)
        return ActionSelectionResultMCTS(np.random.choice(node.valid_actions), 0, -1, 0, -1)

    result = find_checkmate(state, 1, depth=params.depth_search_checkmate_root)

    if result.eval > 0:
        return ActionSelectionResultMCTS(
            result.action,
            0,
            result.action,
            result.eval,
            result.escaped_id
        )
        # print(f"find checkmate: ({e}, {action}, {escaped_id}), {state.pieces_o}")

    if time.perf_counter() - start_t > params.time_limit:
        node.setup_valid_actions(state, 1)
        return ActionSelectionResultMCTS(
            np.random.choice(node.valid_actions),
            0,
            result.action,
            result.eval,
            result.escaped_id
        )

    node.setup_valid_actions(state, 1)

    if actions is not None:
        node.invalid_actions = [i for i in range(32) if i not in actions]
        node.apply_invalid_actions()

    elif pieces_history is not None:
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

    return ActionSelectionResultMCTS(action, i, action, result.eval, result.escaped_id)


def create_root_node(
    tokens: list[list[int]],
    past_strategy_table: jnp.ndarray,
    pred_state: PredictState,
    cache_length: int = 220,
) -> tuple[PredictResult, np.ndarray]:
    model = pred_state.model

    cache = model.create_cache(cache_length)

    result = predict_with_strategy(pred_state, past_strategy_table, cache)
    result = predict_with_tokens(pred_state, tokens, result.cache)

    return result


@dataclass(frozen=True)
class PlayerStateMCTS:
    node: Node
    pieces_history: list[np.ndarray]
    past_strategy_tables: list[np.ndarray]


@dataclass(frozen=True)
class PlayerMCTS(PlayerBase[PlayerStateMCTS, ActionSelectionResultMCTS]):
    params: dict
    model: TransformerWithCache
    mcts_params: SearchParameters
    strategy: Strategy = None

    def get_pred_state(self) -> PredictState:
        return PredictState(self.model, self.params)
    
    def init_state(self, state: game.State, prev_state: PlayerStateMCTS = None) -> tuple[game.State, PlayerStateMCTS, list[list[int]]]:
        if self.strategy is not None:
            state = StateWithStrategy(state.board, state.n_ply, self.strategy)

        if prev_state is not None:
            past_strategy_tables = prev_state.past_strategy_tables
            st = np.sum(past_strategy_tables, axis=0)
        else:
            past_strategy_tables = []
            st = np.zeros((4, 4, 2, 2), dtype=np.uint8)

        tokens = state.create_init_tokens()
        result = create_root_node(tokens, st, self.get_pred_state())

        next_state = PlayerStateMCTS(
            Node(state, result),
            pieces_history=[],
            past_strategy_tables=past_strategy_tables
        )
        return state, next_state, tokens

    def select_next_action(
        self,
        state: game.State,
        player_state: PlayerStateMCTS,
        actions: list[int] = None
    ) -> ActionSelectionResultMCTS:
        result = select_action_with_mcts(
            player_state.node, state, self.get_pred_state(), self.mcts_params,
            pieces_history=np.array(player_state.pieces_history, dtype=np.int16),
            actions=actions
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
        next_player_state = PlayerStateMCTS(node, pieces_history, player_state.past_strategy_tables)

        return next_player_state, state, tokens, result

    def apply_game_result(self, player_state: PlayerStateMCTS, result: GameResult, player_id: int) -> PlayerStateMCTS:
        opponent_id = 1 - player_id
        st = StrategyTokenProducer.create_strategy_table(result.tokens[opponent_id])

        tables = player_state.past_strategy_tables + [st]

        return PlayerStateMCTS(
            player_state.node,
            pieces_history=[],
            past_strategy_tables=tables
        )

    def _create_sample(
        self,
        x: np.ndarray, p: np.ndarray, v: np.ndarray, c: np.ndarray, player_state: PlayerStateMCTS
    ) -> np.ndarray:
        st = player_state.past_strategy_tables[-1]

        return batch.FORMAT_X7_ST_PVC.from_tuple(x, st, p, v, c)

    def visualize_state(self, player_state: PlayerStateMCTS, output_file: str):
        if all([c is None for c in player_state.node.children]):
            return

        dg = Digraph(format='png')
        dg.attr('node', fontname="Myrica M")
        dg.attr('edge', fontname="Myrica M")
        player_state.node.visualize_graph(dg)
        dg.render(output_file)


@dataclass(frozen=True)
class PlayerMCTSConfig(PlayerConfig[PlayerMCTS]):
    base_name: str
    step: int
    mcts_params: SearchParametersRange
    strategy_factory: StrategyFactories

    @property
    def necessary_checkpoint_step(self) -> int | None:
        return self.step

    @property
    def name(self) -> str:
        return f"{self.base_name}-{self.step}"

    def create_player(self, project_dir: str) -> PlayerMCTS:
        ckpt = CheckpointManager(f"{project_dir}/{self.base_name}").load(self.step)

        return PlayerMCTS(
            params=ckpt.params,
            model=ckpt.model.create_caching_model(),
            mcts_params=self.mcts_params.sample(),
            strategy=self.strategy_factory.create()
        )

    def get_checkpoint(self, project_dir: str):
        return CheckpointManager(f"{project_dir}/{self.base_name}").load(self.step)
