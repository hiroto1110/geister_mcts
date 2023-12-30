import numpy as np

from env.state import Token
from buffer import Batch


def create_log(
        win_rates: np.ndarray,
        last_games: Batch
) -> dict[str, object]:

    log = {}

    log.update(create_log_win_rates(win_rates))
    log.update(create_log_last_games(last_games))

    return log


def create_log_win_rates(win_rates: np.ndarray) -> dict:
    return {f'win_rates/{i}': w for i, w in enumerate(win_rates) if w > 0}


def create_log_last_games(last_games: Batch) -> dict:
    log = {}

    t = np.arange(40, 200, step=40)
    n_captured = calc_n_captured(last_games.tokens, t)

    for i in range(n_captured.shape[0]):
        for color in range(n_captured.shape[1]):
            log[f'n_captured/{color}_{t[i]}'] = n_captured[i, color]

    for i in range(7):
        log[f'game_result/{i}'] = (last_games.reward == i).mean()

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


def test():
    batch = Batch.from_npz('./data/replay_buffer/run-3.npz')
    t = np.arange(25, 200, step=25)
    n = calc_n_captured(batch.tokens, t)

    import matplotlib.pyplot as plt

    for i in range(4):
        plt.plot(t, n[:, i], label=f'{i}')

    plt.legend()
    plt.savefig("test.png")


if __name__ == '__main__':
    test()