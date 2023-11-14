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
        if np.all(tokens[t] == 0):
            break

        if x < 6 and y < 6:
            pos[id] = x + 6 * y
        else:
            pos[id] = 36

        pos_history[t] = pos

    return pos_history


def expand(tokens, actions, reward, src_len, dst_len):
    n_expand = (dst_len - src_len) // 4
    if reward != 0:
        """if np.random.random() < 0.7:
            n_expand = 0
        else:
            n_expand = np.random.randint(0, n_expand)"""
        n_expand = np.random.randint(0, n_expand)

    if n_expand == 0:
        return tokens, actions

    inserted_mask = np.any(tokens != 0, axis=1)
    inserted_mask &= tokens[:, Token.COLOR] != 2
    inserted_mask &= tokens[:, Token.COLOR] != 3
    inserted_mask &= tokens[:, Token.X] < 6

    mask = np.zeros(dst_len, dtype='bool')
    mask[tokens[:, Token.T]] = inserted_mask
    mask[0] = 0

    mask[1:] &= mask[:-1]
    mask[:-1] &= mask[1:]

    mask[np.max(tokens[:, Token.T]):] = 0

    if sum(mask) < n_expand:
        n_expand = sum(mask)

    inserted_t = np.random.choice(np.where(mask)[0], size=n_expand, replace=False)
    inserted_t = np.sort(inserted_t)

    for t in inserted_t[::-1]:
        i = np.where(tokens[:, Token.T] == t)[0][0]
        token1 = tokens[i]
        token2 = tokens[i + 1]

        action1 = actions[i - 1]
        action2 = actions[i]

        last_pos1 = token1[Token.X] + 6 * token1[Token.Y] - game.DIRECTIONS[action1 % 4]
        last_x1 = last_pos1 % 6
        last_y1 = last_pos1 // 6

        last_pos2 = token2[Token.X] + 6 * token2[Token.Y] - game.DIRECTIONS[action2 % 4]
        last_x2 = last_pos2 % 6
        last_y2 = last_pos2 // 6

        assert 0 <= last_x1 < 6, f"{last_x1}, t:{t}-{inserted_t}, a:{action1}, {tokens[i-2:i+3]}"
        assert 0 <= last_y1 < 6, f"{last_y1}, t:{t}-{inserted_t}, a: {action1}, {tokens[i-2:i+3]}"
        assert 0 <= last_x2 < 6, f"{last_x2}, t:{t}-{inserted_t}, a: {action2}, {tokens[i-2:i+3]}"
        assert 0 <= last_y2 < 6, f"{last_y2}, t:{t}-{inserted_t}, a: {action2}, {tokens[i-2:i+3]}"

        inserted_tokens = [
            (token1[Token.COLOR], token1[Token.ID], token1[Token.X], token1[Token.Y], t + 0),
            (token2[Token.COLOR], token2[Token.ID], token2[Token.X], token2[Token.Y], t + 1),
            (token1[Token.COLOR], token1[Token.ID], last_x1, last_y1, t + 2),
            (token2[Token.COLOR], token2[Token.ID], last_x2, last_y2, t + 3)
        ]

        inserted_actions = [
            action1,
            action2,
            action1 - action1 % 4 + (3 - action1 % 4),
            action2 - action2 % 4 + (3 - action2 % 4),
        ]

        tokens[tokens[:, Token.T] >= t, Token.T] += 4
        tokens = np.concatenate([tokens[:i], inserted_tokens, tokens[i:-4]])

        actions = np.concatenate([actions[:i-1], inserted_actions, actions[i-1:-4]])

    return tokens, actions


def main():
    np.random.seed(10)

    import buffer
    batch = buffer.load_batch(['replay_buffer/run-2.npz'], shuffle=False)

    tokens = np.zeros((batch.tokens.shape[0], 222, 5), dtype=np.uint8)
    tokens[:, :200, :] = batch.tokens

    actions = np.zeros((batch.policy.shape[0], 222), dtype=np.uint8)
    actions[:, :200] = batch.policy

    for i in range(len(batch.tokens)):
        tokens[i], actions[i] = expand(tokens[i], actions[i], batch.reward[i], 180, 200)
        print(i)

    batch = buffer.Batch(tokens, None, actions, batch.reward, batch.colors)
    buffer.save_batch(batch, 'replay_buffer/run-2-e.npz')


if __name__ == "__main__":
    main()
