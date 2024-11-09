import os
import dataclasses
import numpy as np


@dataclasses.dataclass
class Feature:
    length_const: int
    length_per_token: int
    shape: list[int] = (-1,)


@dataclasses.dataclass
class BatchFormat:
    features: list[Feature]
    dtype = np.uint8

    def from_tuple(self, *a: np.ndarray) -> np.ndarray:
        def _reshape(x: np.ndarray, feature: Feature) -> np.ndarray:
            return x.reshape((*x.shape[:-len(feature.shape)], -1)).astype(self.dtype)
        
        reshaped_a = [_reshape(a_i, f_i) for a_i, f_i in zip(a, self.features)]

        return np.concatenate(reshaped_a, axis=-1, dtype=self.dtype)
    
    def astuple(self, batch: np.ndarray) -> list[np.ndarray]:
        num_tokens = (batch.shape[-1] - self.length_const) // self.length_per_token

        results = []

        for feature in self.features:
            length = feature.length_const + num_tokens * feature.length_per_token

            results.append(batch[..., :length])
            batch = batch[..., length:]

        return tuple(results)

    @property
    def length_const(self) -> int:
        return sum([f.length_const for f in self.features])
    
    @property
    def length_per_token(self) -> int:
        return sum([f.length_per_token for f in self.features])


FORMAT_XARC = BatchFormat([
    Feature(length_const=0, length_per_token=5, shape=(-1, 5)),
    Feature(length_const=0, length_per_token=1),
    Feature(length_const=1, length_per_token=0),
    Feature(length_const=8, length_per_token=0),
])


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
    x: np.ndarray, action: np.ndarray, reward: np.ndarray, color: np.ndarray
) -> np.ndarray:
    """
    x: [..., seq_len, 5]
    action: [..., seq_len]
    reward: [..., 1]
    color: [..., 8]
    """
    seq_len = x.shape[-2]

    x_flatten = x.reshape((*x.shape[:-2], seq_len * x.shape[-1]))
    action_flatten = action.reshape((*x.shape[:-2], seq_len))
    reward_flatten = reward.reshape((*x.shape[:-2], 1)).astype(np.uint8)

    return np.concatenate(
        [x_flatten, action_flatten, reward_flatten, color],
        axis=-1,
        dtype=np.uint8
    )


def astuple(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    batch: [..., seq_len * 6 + 9 + COLOR_COUNT_FEATURE_LENGTH]
    """
    return get_posses(batch), get_tokens(batch), get_action(batch), get_reward(batch), get_color(batch)
    # return get_tokens(batch), get_action(batch), get_reward(batch), get_color(batch)


def get_posses(batch: np.ndarray) -> np.ndarray:
    seq_len = get_seq_len(batch.shape[-1])
    return batch[..., :seq_len * 16].reshape((*batch.shape[:-1], seq_len, 16))


def get_tokens(batch: np.ndarray) -> np.ndarray:
    seq_len = get_seq_len(batch.shape[-1])
    return batch[..., seq_len * 16: seq_len * 21].reshape((*batch.shape[:-1], seq_len, 5))


def get_action(batch: np.ndarray) -> np.ndarray:
    seq_len = get_seq_len(batch.shape[-1])
    return batch[..., seq_len * 21: seq_len * 22].reshape((*batch.shape[:-1], seq_len))


def get_reward(batch: np.ndarray) -> np.ndarray:
    seq_len = get_seq_len(batch.shape[-1])
    return batch[..., seq_len * 22]


def get_color(batch: np.ndarray) -> np.ndarray:
    seq_len = get_seq_len(batch.shape[-1])
    return batch[..., seq_len * 22 + 1: seq_len * 22 + 9]


def is_won(batch: np.ndarray) -> bool:
    return get_reward(batch) > 3


def get_length_of_one_sample(seq_len: int) -> int:
    return seq_len * 22 + 9


def get_seq_len(length_of_one_sample: int) -> int:
    assert (length_of_one_sample - 9) % 22 == 0, f"{length_of_one_sample}"
    return (length_of_one_sample - 9) // 22


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
