import os
import numpy as np


COLOR_COUNT_FEATURE_LENGTH = 3 * 6 * 2 * 3**4


def load(path: str) -> np.ndarray:
    if path.endswith('.npy'):
        return np.load(path)

    if path.endswith('.npz'):
        return _load_npz(path)

    raise ValueError(f'Unsupported Suffix: {path}')


def _load_npz(path: str) -> np.ndarray:
    data = np.load(path)
    return create_batch(data['t'], data['p'], data['r'], data['c'])


def save(path: str, batch: np.ndarray, append: bool):
    if append and os.path.exists(path):
        old_batch = load(path)
        batch = np.concatenate([old_batch, batch])

    np.save(path, batch)


def create_batch(
    x: np.ndarray, action: np.ndarray, reward: np.ndarray, color: np.ndarray, color_count: np.ndarray
) -> np.ndarray:
    """
    x: [..., seq_len, 5]
    action: [..., seq_len]
    reward: [..., 1]
    color: [..., 8]
    color_count: [..., 3, 6, 2 * 3**4]
    """
    seq_len = x.shape[-2]

    x_flatten = x.reshape((*x.shape[:-2], seq_len * 5))
    action_flatten = action.reshape((*x.shape[:-2], seq_len))
    color_count_flatten = color_count.reshape((*x.shape[:-2], COLOR_COUNT_FEATURE_LENGTH))

    return np.concatenate(
        [x_flatten, action_flatten, reward.astype(np.uint8), color, color_count_flatten],
        axis=-1,
        dtype=np.uint8
    )


def astuple(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    batch: [..., seq_len * 6 + 9 + COLOR_COUNT_FEATURE_LENGTH]
    """
    return get_tokens(batch), get_action(batch), get_reward(batch), get_color(batch), get_color_count(batch)


def get_tokens(batch: np.ndarray) -> np.ndarray:
    seq_len = get_seq_len(batch.shape[-1])
    return batch[..., :seq_len * 5].reshape((*batch.shape[:-1], seq_len, 5))


def get_action(batch: np.ndarray) -> np.ndarray:
    seq_len = get_seq_len(batch.shape[-1])
    return batch[..., seq_len * 5: seq_len * 6].reshape((*batch.shape[:-1], seq_len))


def get_reward(batch: np.ndarray) -> np.ndarray:
    seq_len = get_seq_len(batch.shape[-1])
    return batch[..., seq_len * 6]


def get_color(batch: np.ndarray) -> np.ndarray:
    seq_len = get_seq_len(batch.shape[-1])
    return batch[..., seq_len * 6 + 1: seq_len * 6 + 9]


def get_color_count(batch: np.ndarray) -> np.ndarray:
    seq_len = get_seq_len(batch.shape[-1])
    return batch[..., seq_len * 6 + 9:].reshape((*batch.shape[:-1], 3, 6, -1))


def is_won(batch: np.ndarray) -> bool:
    return get_reward(batch) > 3


def get_length_of_one_sample(seq_len: int) -> int:
    return seq_len * 6 + 9 + COLOR_COUNT_FEATURE_LENGTH


def get_seq_len(length_of_one_sample: int) -> int:
    assert (length_of_one_sample - 9 - COLOR_COUNT_FEATURE_LENGTH) % 6 == 0
    return (length_of_one_sample - 9 - COLOR_COUNT_FEATURE_LENGTH) // 6


class ReplayBuffer:
    def __init__(
            self,
            buffer_size: int,
            sample_shape: tuple,
            seq_length: int
    ):
        self.buffer_size = buffer_size
        self.seq_length = seq_length
        self.index = 0
        self.n_samples = 0

        self.buffer = np.zeros((buffer_size, *sample_shape, get_length_of_one_sample(seq_length)), dtype=np.uint8)

    def __len__(self):
        return self.n_samples

    def get_last_indices(self, n: int) -> np.ndarray:
        return (self.index - n + np.arange(n)) % self.buffer_size

    def get_last_minibatch(self, batch_size):
        indices = self.get_last_indices(batch_size)
        return self.create_batch_from_indices(indices)

    def get_minibatch(self, batch_size):
        indices = np.random.choice(range(self.n_samples), size=batch_size)

        return self.create_batch_from_indices(indices)

    def create_batch_from_indices(self, indices) -> np.ndarray:
        return self.buffer[indices]

    def add_sample(self, sample: np.ndarray):
        self.buffer[self.index] = sample

        self.n_samples = max(self.n_samples, self.index + 1)
        self.index = (self.index + 1) % self.buffer_size

    def load(self, file_name: str):
        buffer = load(file_name)

        n_samples = min(self.buffer_size, buffer.shape[0])

        self.buffer[:n_samples] = buffer[-n_samples:]

        self.n_samples = n_samples
        self.index = self.n_samples % self.buffer_size
