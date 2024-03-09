import numpy as np

from batch import astuple, load, create_batch
from env.state import Token


def init_feature() -> np.ndarray:
    return np.zeros((3, 6, 2, 2, 2, 2, 2))


def create_feature(batch: np.ndarray, feature: np.ndarray):
    tokens, actions, rewards, colors = astuple(batch)

    if tokens[0, Token.Y] > 3:
        should_invert_y = True
    else:
        should_invert_y = False

    num_captured = np.zeros(4, dtype=np.int16)

    for i in np.arange(tokens.shape[0])[tokens[:, Token.T] != 0]:
        token = tokens[i]

        if np.any(num_captured >= 4):
            break

        if token[Token.X] == 6 and token[Token.Y] == 6:
            num_captured[token[Token.COLOR]] += 1

        elif 8 <= token[Token.ID] < 16:
            x = token[Token.X]
            y = token[Token.Y]
            color = colors[token[Token.ID] - 8]

            if should_invert_y:
                y = 5 - y

            if x >= 3:
                x = 5 - x

            n = np.clip(num_captured - 2, 0, 1)

            tokens[i, 5] = feature[(x, y, *n, 0)]
            tokens[i, 6] = feature[(x, y, *n, 1)]

            feature[(x, y, *n, color)] += 1

    return feature, create_batch(tokens, actions, rewards, colors)


def test():
    import tqdm
    np.set_printoptions(threshold=10000)

    batch = load("./data/replay_buffer/run-7-new.npy")
    print(batch.shape)

    print(astuple(batch[3, -1])[0])

    return

    for i in tqdm.tqdm(range(batch.shape[0])):
        f = init_feature()

        for j in range(batch.shape[1]):
            f, batch[i, j] = create_feature(batch[i, j], f)

    np.save("./data/replay_buffer/run-7-new.npy", batch)


if __name__ == "__main__":
    test()
