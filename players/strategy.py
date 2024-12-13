from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

from players.base import TokenProducer
import env.state as game
import batch


class StrategyToken(IntEnum):
    DEF = 5
    ATK = 6


@dataclass(frozen=True)
class Strategy:
    table: np.ndarray

    @staticmethod
    def create_empty_table() -> np.ndarray:
        return np.zeros((2, 4, 4))

    @property
    def attack(self) -> np.ndarray:
        return self.table[0]

    @property
    def defence(self) -> np.ndarray:
        return self.table[1]


def find_closest_pieses(pieses_pos: np.ndarray, target_pos: int) -> list[int]:
    mask = pieses_pos == game.CAPTURED

    x = (pieses_pos % 6) - (target_pos % 6)
    y = (pieses_pos // 6) - (target_pos // 6)

    d = np.abs(x) + np.abs(y)
    d[mask] = 100

    closest_mask = d == np.min(d)
    pieses_id = np.arange(len(pieses_pos))[closest_mask]

    return list(pieses_id)


def is_action_to_enter_deadlock(state: game.State, action: int, player: int) -> bool:
    if player == 1:
        pos_p, pos_o = state.board[game.POS_P], state.board[game.POS_O]
        escaping_pos = game.ESCAPE_POS_P
    else:
        pos_p, pos_o = state.board[game.POS_O], state.board[game.POS_P]
        escaping_pos = game.ESCAPE_POS_O

    defenders_0 = find_closest_pieses(pos_o, target_pos=escaping_pos[0])
    defenders_1 = find_closest_pieses(pos_o, target_pos=escaping_pos[1])
    defenders = defenders_0 + defenders_1

    defender_pos = pos_o[defenders]
    defender_pos = np.stack([defender_pos] * 4, axis=0)
    defender_pos[0] -= 6
    defender_pos[1] -= 1
    defender_pos[2] += 1
    defender_pos[3] += 6

    defender_pos[1, defender_pos[1] % 6 == 5] = -1
    defender_pos[2, defender_pos[2] % 6 == 0] = -1

    defender_pos = defender_pos.flatten()

    p_id, d = game.action_to_id(action)
    action_pos = pos_p[p_id] + d

    if action_pos not in defender_pos:
        return False

    return True


class StrategyTokenProducer(TokenProducer):
    def __init__(self):
        self.tokens: np.ndarray = None
        self.attacked_id_history: np.ndarray = None

    @classmethod
    def get_batch_format(cls) -> batch.BatchFormat:
        return batch.FORMAT_X7ARC

    def init_game(self, game_length: int):
        self.tokens = np.zeros((2, game_length + 40, 7), dtype=np.uint8)
        self.attacked_id_history = np.zeros((2, game_length + 40, 8), dtype=np.uint8)

    def on_step(self, state: game.State, action: int, player: int):
        if not is_action_to_enter_deadlock(state, action, 1):
            return

        player_id = 0 if player == 1 else 1
        piece_id, _ = game.action_to_id(action)

        self.attacked_id_history[player_id, state.n_ply + 1, piece_id] = 1

    def add_tokens(self, state: game.State, tokens_in_step: list[list[int]], player_id: int):
        empty_mask = np.all(self.tokens[player_id] == 0, axis=-1)
        idx = np.arange(len(empty_mask))[empty_mask].min()
        self.tokens[player_id, idx: idx + len(tokens_in_step), :5] = [tokens[:5] for tokens in tokens_in_step]

        if not any([t[game.Token.X] == 6 for t in tokens_in_step]):
            return

        captured_id = tokens_in_step[0][game.Token.ID]

        mask = self.tokens[player_id, :, game.Token.X] == 6

        if captured_id < 8:
            mask *= self.tokens[player_id, :, game.Token.ID] < 8
        else:
            mask *= self.tokens[player_id, :, game.Token.ID] >= 8

        if not np.any(mask):
            mask[:8] = 1
            mask_pos = 0
        else:
            mask_poses = np.arange(len(mask), dtype=np.uint16)[mask]
            mask_pos = mask_poses.max()
            mask[:mask_pos] = 0

        last_captured_t = self.tokens[player_id, mask_pos, game.Token.T]

        if captured_id < 8:
            attacked_id = self.attacked_id_history[1 - player_id, last_captured_t:].any(axis=0)

            count_attacked = np.sum(attacked_id == 1)
            count_captured = np.sum((attacked_id == 1) * (state.pos_o == game.CAPTURED))

            if count_attacked == 0:
                self.tokens[player_id, mask, 5] = 0
            else:
                self.tokens[player_id, mask, 5] = 1 + np.clip(count_captured, 0, 1)
        else:
            attacked_id = self.attacked_id_history[player_id, last_captured_t:].any(axis=0)

            count_r = np.sum(attacked_id * (state.col_p == game.RED))
            count_b = np.sum(attacked_id * (state.col_p == game.BLUE))

            if count_b == count_r:
                self.tokens[player_id, mask, 6] = 0
            elif count_r > count_b:
                self.tokens[player_id, mask, 6] = 1
            else:
                self.tokens[player_id, mask, 6] = 2

    @staticmethod
    def create_strategy_table(tokens: np.ndarray) -> np.ndarray:
        # [cap_r, cap_b, st type, count]
        strategy = np.zeros((4, 4, 2, 2), dtype=np.uint8)

        captured_count = np.zeros((2, 2), dtype=np.uint8)

        for c, id, _, _, _, st1, st2 in tokens[tokens[:, game.Token.X] == 6]:
            if id < 8:
                i = 0
                st = st1
            else:
                i = 1
                st = st2
                c = c - 2

            captured_count[i, c] += 1
            cap_r = captured_count[i, 0]
            cap_b = captured_count[i, 1]

            if st == 1:
                strategy[cap_r, cap_b, i, 0] += 1
            if st == 2:
                strategy[cap_r, cap_b, i, 1] += 1

        return strategy


@dataclass(frozen=True)
class StateWithStrategy(game.State):
    strategy: Strategy

    def create_init_tokens(self):
        tokens = super().create_init_tokens()

        st_token_def = self.strategy.defence[0, 0]
        st_token_atk = self.strategy.attack[0, 0]

        for i in range(len(tokens)):
            tokens[i] = list(tokens[i]) + [0, 0]
            tokens[i][StrategyToken.DEF] = st_token_def
            tokens[i][StrategyToken.ATK] = st_token_atk

        return tokens

    def _step(self, state: game.State, result: game.StepResult) -> tuple["StateWithStrategy", game.StepResult]:
        state = StateWithStrategy(state.board, state.n_ply, self.strategy)

        tokens = [list(token) + [0, 0] for token in result.tokens]
        result = replace(result, tokens=tokens)

        if result.winner != 0:
            return state, result

        for i in range(len(tokens)):
            if tokens[i][game.Token.X] == 6:
                captured_id = tokens[i][game.Token.ID]
                break
        else:
            return state, result

        if captured_id > 8:
            cap_r = state.get_num_captured(player=1, color=game.RED)
            cap_b = state.get_num_captured(player=1, color=game.BLUE)
            st_table = self.strategy.defence
            idx = StrategyToken.DEF
        else:
            cap_r = state.get_num_captured(player=0, color=game.RED)
            cap_b = state.get_num_captured(player=0, color=game.BLUE)
            st_table = self.strategy.attack
            idx = StrategyToken.ATK

        tokens[i][idx] = st_table[cap_r, cap_b]

        return state, result

    def step(self, action: int, player: int) -> tuple["StateWithStrategy", game.StepResult]:
        state, result = super().step(action, player)

        return self._step(state, result)

    def step_afterstate(self, afterstate: game.Afterstate, color: int) -> tuple["StateWithStrategy", game.StepResult]:
        state, result = super().step_afterstate(afterstate, color)

        return self._step(state, result)
