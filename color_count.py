import numpy as np

from batch import get_tokens, get_color, load
from env.state import Token


def init_feature() -> np.ndarray:
    return np.zeros((3, 6, 3, 3, 3, 3, 2))


def create_feature(batch: np.ndarray):
    tokens = get_tokens(batch)
    colors = get_color(batch)

    feature = init_feature()

    if tokens[0, Token.Y] > 3:
        should_invert_y = True
    else:
        should_invert_y = False

    num_captured = np.zeros(4, dtype=np.int16)

    for token in tokens[tokens[:, Token.T] != 0]:
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

            n = np.clip(num_captured - 1, 0, 2)

            feature[(x, y, *n, color)] += 1

    return feature


def test():
    import tqdm

    batch = load("./data/replay_buffer/run-7.npy")
    print(batch.shape)

    features = np.zeros((batch.shape[0], batch.shape[1], 3, 6, 2 * 3**4), dtype=np.uint8)

    for i in tqdm.tqdm(range(batch.shape[0])):
        f = init_feature()

        for j in range(batch.shape[1]):
            features[i, j] = f.reshape(3, 6, -1)
            f += create_feature(batch[i, j])

    features = features.reshape(batch.shape[0], batch.shape[1], -1)

    print(features.shape)

    batch = np.concatenate([batch, features], axis=-1, dtype=np.uint8)

    np.save("./data/replay_buffer/run-4.npy", batch)


if __name__ == "__main__":
    test()
