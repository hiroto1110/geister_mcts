import numpy as np


def create_pos_history_from_tokens(tokens: np.ndarray) -> np.ndarray:
    pos_history = np.zeros((tokens.shape[0], 16), dtype=np.uint8)

    if tokens[0, 3] < 3:
        pos = np.array([1, 2, 3, 4, 7, 8, 9, 10, 25, 26, 27, 28, 31, 32, 33, 34])
    else:
        pos = np.array([25, 26, 27, 28, 31, 32, 33, 34, 1, 2, 3, 4, 7, 8, 9, 10])

    for t, (c, id, x, y, n) in enumerate(tokens):
        if x < 6 and y < 6:
            pos[id] = x + 6 * y
        else:
            pos[id] = 36

        pos_history[t] = pos

    return pos_history


def expand(tokens, actions, reward, color, src_len, dst_len):
    n_expand = (dst_len - src_len) // 4
    if reward != 0:
        n_expand = np.random.randint(0, n_expand)


