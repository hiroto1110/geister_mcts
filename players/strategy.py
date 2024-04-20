import numpy as np

from base import PlayerBase
from env.state import State, CAPTURED, BLUE, RED, POS_P, POS_O, COL_P, COL_O


class StrategyPlayer(PlayerBase):
    strategy: np.ndarray

    def get_cap_prob(self, state: State):
        b = state.board

        cap_p_b = np.sum((b[POS_P] == CAPTURED) * (b[COL_P] == BLUE))
        cap_p_r = np.sum((b[POS_P] == CAPTURED) * (b[COL_P] == RED))
        cap_o_b = np.sum((b[POS_O] == CAPTURED) * (b[COL_O] == BLUE))
        cap_o_r = np.sum((b[POS_O] == CAPTURED) * (b[COL_O] == RED))

        n = np.array([cap_p_b, cap_p_r, cap_o_b, cap_o_r], dtype=np.uint8)
        n = np.clip(n, 0, 1)

        s = self.strategy[:16].reshape(2, 2, 2, 2)

        return s[tuple(n)]

    def select_action(self):
        pass
