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

            feature_batch = batch[..., :length]
            feature_batch = feature_batch.reshape((*feature_batch.shape[:-1], *feature.shape))
            results.append(feature_batch)

            batch = batch[..., length:]

        return tuple(results)

    def get_length_of_one_sample(self, seq_len: int) -> int:
        return seq_len * self.length_per_token + self.length_const

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

FORMAT_X7ARC = BatchFormat([
    Feature(length_const=0, length_per_token=7, shape=(-1, 7)),
    Feature(length_const=0, length_per_token=1),
    Feature(length_const=1, length_per_token=0),
    Feature(length_const=8, length_per_token=0),
])


def load(path: str) -> np.ndarray:
    if path.endswith('.npy'):
        return np.load(path)

    raise ValueError(f'Unsupported Suffix: {path}')


def save(path: str, batch: np.ndarray, append: bool):
    if append and os.path.exists(path):
        old_batch = load(path)
        batch = np.concatenate([old_batch, batch])

    np.save(path, batch)


class ReplayBuffer:
    def __init__(
        self,
        format: BatchFormat,
        buffer_size: int,
        sample_shape: tuple,
        seq_length: int
    ):
        self.format = format
        self.buffer_size = buffer_size
        self.seq_length = seq_length
        self.index = 0
        self.n_samples = 0

        self.buffer = np.zeros((buffer_size, *sample_shape, format.get_length_of_one_sample(seq_length)), dtype=np.uint8)

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
