import numpy as np


def load(path: str):
    if path.endswith('.npy'):
        return np.load(path)

    if path.endswith('.npz'):
        return _load_npz(path)

    raise ValueError(f'Unsupported Suffix: {path}')


def _load_npz(path: str) -> np.ndarray:
    data = np.load(path)
    return create_batch(data['t'], data['p'], data['r'], data['c'])


def create_batch(x: np.ndarray, action: np.ndarray, reward: np.ndarray, color: np.ndarray) -> np.ndarray:
    """
    x: [..., seq_len, 5]
    action: [..., seq_len]
    reward: [..., 1]
    color: [..., 8]
    """
    seq_len = x.shape[-2]

    x_flatten = x.reshape(-1, seq_len * 5)

    return np.concatenate([x_flatten, action, reward, color], axis=-1, dtype=np.uint8)


def astuple(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    batch: [..., seq_len * 6 + 9]
    """
    seq_len = (batch.shape[-1] - 9) // 6
    splited = []

    total = 0
    for length in [seq_len * 5, seq_len, 1, 8]:
        splited.append(batch[..., total: total + length])
        total += length

    return tuple(splited)


def get_length_of_one_sample(seq_len: int) -> int:
    return seq_len * 6 + 9


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
        assert file_name.endswith('.npy')

        buffer = np.load(file_name)

        n_samples = min(self.buffer_size, buffer.shape[0])

        self.buffer[:n_samples] = buffer[-n_samples:]

        self.n_samples = n_samples
        self.index = self.n_samples % self.buffer_size
