import numpy as np

from env.state import Token
from batch import get_reward


def create_log_win_rates(win_rates: np.ndarray) -> dict:
    return {f'win_rates/{i}': w for i, w in enumerate(win_rates) if w > 0}


def create_log_last_games(last_games: np.ndarray) -> dict:
    log = {}
    for i in range(7):
        log[f'game_result/{i}'] = (get_reward(last_games) == i).mean()

    return log


def calc_n_captured(tokens: np.ndarray, t: np.ndarray) -> np.ndarray:
    # tokens: [..., n ply, 5]
    tokens = tokens.reshape(-1, tokens.shape[-2], tokens.shape[-1])

    # [..., n ply, 5]
    color_mask = np.eye(5)[tokens[..., Token.COLOR]]
    color_mask = color_mask[..., :4]

    # [..., n ply, ]
    pos_mask = tokens[..., Token.X] == 6
    captured_mask = color_mask * pos_mask[..., np.newaxis]

    # [..., n ply, 4]
    n_captured = np.cumsum(captured_mask, axis=-2)
    # [..., len(t), 4]
    n_captured = n_captured[:, t]
    n_captured = n_captured.mean(axis=(0))

    return n_captured
