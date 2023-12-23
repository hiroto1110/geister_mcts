import os
from dataclasses import dataclass, astuple, field

import numpy as np
import env.state as game


@dataclass
class Batch:
    tokens: np.ndarray = field(default_factory=lambda: np.zeros(0))
    policy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    reward: np.ndarray = field(default_factory=lambda: np.zeros(0))
    colors: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __len__(self):
        return self.tokens.shape[0]

    def is_won(self):
        return self.reward > 3

    def astuple(self):
        return astuple(self)

    def reshape(self, shape: tuple) -> 'Batch':
        tokens = self.tokens.reshape((*shape, self.tokens.shape[-2], game.TOKEN_SIZE))
        policy = self.policy.reshape((*shape, self.policy.shape[-1]))
        reward = self.reward.reshape((*shape, 1))
        colors = self.colors.reshape((*shape, 8))

        return Batch(tokens, policy, reward, colors)

    @classmethod
    def stack(cls, samples: list['Batch']) -> 'Batch':
        tokens = np.stack([sample.tokens for sample in samples])
        policy = np.stack([sample.policy for sample in samples])
        reward = np.stack([sample.reward for sample in samples])
        colors = np.stack([sample.colors for sample in samples])

        return Batch(tokens, policy, reward, colors)

    def to_npz(self, path):
        save_dict = {
            't': self.tokens,
            'p': self.policy,
            'r': self.reward,
            'c': self.colors,
        }
        np.savez(path, **save_dict)

    @classmethod
    def from_npz(cls, path, shuffle=False):
        with np.load(path) as data:
            tokens = data['t']
            policy = data['p']
            reward = data['r']
            colors = data['c']

            if shuffle:
                indices = np.arange(len(tokens))
                np.random.shuffle(indices)

                tokens = tokens[indices]
                policy = policy[indices]
                reward = reward[indices]
                colors = colors[indices]

            return Batch(tokens, policy, reward, colors)

    def split(self, r: float) -> tuple['Batch', 'Batch']:
        assert 0 < r < 1

        n = int(len(self) * r)

        indices = np.arange(len(self))

        b1 = self.create_batch_from_indices(indices[:n])
        b2 = self.create_batch_from_indices(indices[n:])

        return b1, b2

    def divide(self, batch_size: int) -> list["Batch"]:
        n = len(self) // batch_size

        indices = np.arange(n * batch_size)
        indices = indices.reshape(n, -1)

        return [self.create_batch_from_indices(i) for i in indices]

    def create_minibatch(self, batch_size: int) -> "Batch":
        indices = np.random.choice(range(self.tokens.shape[0]), size=batch_size)
        return self.create_batch_from_indices(indices)

    def create_batch_from_indices(self, indices) -> "Batch":
        tokens = self.tokens[indices]
        policy = self.policy[indices]
        reward = self.reward[indices]
        colors = self.colors[indices]

        return Batch(tokens, policy, reward, colors)

    def save(self, file_name: str, append: bool):
        tokens = self.tokens
        policy = self.policy
        reward = self.reward
        colors = self.colors

        if append and os.path.isfile(file_name):
            with np.load(file_name) as data:
                tokens = np.concatenate([data['t'], tokens])
                policy = np.concatenate([data['p'], policy])
                reward = np.concatenate([data['r'], reward])
                colors = np.concatenate([data['c'], colors])

        save_dict = {
            't': tokens,
            'p': policy,
            'r': reward,
            'c': colors,
        }
        np.savez(file_name, **save_dict)


class ReplayBuffer:
    def __init__(
            self,
            buffer_size: int,
            sample_shape: tuple,
            seq_length: int
            ):
        self.buffer_size = buffer_size
        self.index = 0
        self.n_samples = 0

        self.tokens = np.zeros((buffer_size, *sample_shape, seq_length, game.TOKEN_SIZE), dtype=np.uint8)
        self.policy = np.zeros((buffer_size, *sample_shape, seq_length), dtype=np.uint8)
        self.reward = np.zeros((buffer_size, *sample_shape, 1), dtype=np.int8)
        self.colors = np.zeros((buffer_size, *sample_shape, 8), dtype=np.uint8)

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

    def get_all_batch(self, shuffle: bool):
        indices = np.arange(self.n_samples)
        if shuffle:
            np.random.shuffle(indices)

        return self.create_batch_from_indices(indices)

    def create_batch_from_indices(self, indices):
        tokens = self.tokens[indices]
        policy = self.policy[indices]
        reward = self.reward[indices]
        colors = self.colors[indices]

        return Batch(tokens, policy, reward, colors)

    def add_sample(self, sample: Batch):
        self.tokens[self.index] = sample.tokens
        self.policy[self.index] = sample.policy
        self.reward[self.index] = sample.reward
        self.colors[self.index] = sample.colors

        self.n_samples = max(self.n_samples, self.index + 1)
        self.index = (self.index + 1) % self.buffer_size

    def save(self, file_name: str, append: bool, indices: np.ndarray = None):
        if indices is not None:
            tokens = self.tokens[indices]
            policy = self.policy[indices]
            reward = self.reward[indices]
            colors = self.colors[indices]
        else:
            tokens = self.tokens
            policy = self.policy
            reward = self.reward
            colors = self.colors

        if append and os.path.isfile(file_name):
            with np.load(file_name) as data:
                tokens = np.concatenate([data['t'], tokens])
                policy = np.concatenate([data['p'], policy])
                reward = np.concatenate([data['r'], reward])
                colors = np.concatenate([data['c'], colors])

        save_dict = {
            't': tokens,
            'p': policy,
            'r': reward,
            'c': colors,
        }
        np.savez(file_name, **save_dict)

    def load(self, file_name: str):
        assert file_name.endswith('.npz')

        with np.load(file_name) as data:
            tokens = data['t']
            policy = data['p']
            reward = data['r']
            colors = data['c']

        n_samples = min(len(tokens), self.buffer_size)

        self.tokens[:n_samples] = tokens[-n_samples:]
        self.policy[:n_samples] = policy[-n_samples:]
        self.reward[:n_samples] = reward[-n_samples:]
        self.colors[:n_samples] = colors[-n_samples:]

        self.n_samples = n_samples
        self.index = self.n_samples % self.buffer_size
