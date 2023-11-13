import numpy as np

import geister as game
from geister import Token


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
    
    if n_expand == 0:
        return tokens

    mask = np.zeros(src_len)
    mask[tokens[:, Token.T]] = tokens[:, Token.X] < 6

    mask = mask.reshape(-1, 2, 5)
    mask = np.all(mask, axis=1)

    inserted_t = np.random.choice(np.where(mask)[0], size=n_expand, replace=False)

    for t in inserted_t:
        t *= 2
        i = np.where(tokens[Token.T] == t)[0][0]
        tokens = np.insert(tokens, i, tokens[i])

        tokens[:, Token.T][tokens[:, Token.T] >= t] += 4 


def test():
    from scipy.stats import beta
    import seaborn
    import matplotlib.pyplot as plt

    a = 1 + np.arange(20)
    b = 1 + np.arange(20)

    aa, bb = np.meshgrid(a, b)

    p = beta.ppf(0.1, aa, bb)

    ax = seaborn.heatmap(p > 0.5)
    ax.set(xlabel="won", ylabel="lost")
    ax.xaxis.tick_top()
    plt.savefig("test.png")

    print(p)


if __name__ == "__main__":
    test()
