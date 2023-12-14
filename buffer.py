import os
from dataclasses import dataclass, astuple, field

import numpy as np
import geister as game


@dataclass
class Sample:
    tokens: np.ndarray = field(default_factory=lambda: np.zeros(0))
    policy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    reward: int = 0
    colors: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class Batch:
    tokens: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mask: np.ndarray = field(default_factory=lambda: np.zeros(0))
    policy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    reward: np.ndarray = field(default_factory=lambda: np.zeros(0))
    colors: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def astuple(self):
        return astuple(self)

    def to_npz(self, path):
        save_dict = {
            't': self.tokens,
            'm': self.mask,
            'p': self.policy,
            'r': self.reward,
            'c': self.colors,
        }
        np.savez(path, **save_dict)

    @classmethod
    def from_npz(cls, path):
        with np.load(path) as data:
            return Batch(data['t'], data['m'], data['p'], data['r'], data['c'])

    def create_batch_from_indices(self, indices):
        tokens = self.tokens[indices]
        policy = self.policy[indices]
        reward = self.reward[indices]
        colors = self.colors[indices]

        mask = np.any(tokens != 0, axis=2)

        return Batch(tokens, mask, policy, reward, colors)


class ReplayBuffer:
    def __init__(self, buffer_size: int, seq_length: int, file_name: str = None):
        self.buffer_size = buffer_size
        self.file_name = file_name
        self.index = 0
        self.n_samples = 0

        self.tokens_buffer = np.zeros((buffer_size, seq_length, game.TOKEN_SIZE), dtype=np.uint8)
        self.policy_buffer = np.zeros((buffer_size, seq_length), dtype=np.uint8)
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.int8)
        self.colors_buffer = np.zeros((buffer_size, 8), dtype=np.uint8)

    def __len__(self):
        return self.n_samples

    def get_last_minibatch(self, batch_size):
        indices = (self.index - batch_size + np.arange(batch_size)) % self.buffer_size
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
        tokens = self.tokens_buffer[indices]
        policy = self.policy_buffer[indices]
        reward = self.reward_buffer[indices]
        colors = self.colors_buffer[indices]

        mask = np.any(tokens != 0, axis=2)

        return Batch(tokens, mask, policy, reward, colors)

    def add_sample(self, sample: Sample):
        self.tokens_buffer[self.index] = sample.tokens
        self.policy_buffer[self.index] = sample.policy
        self.reward_buffer[self.index] = sample.reward
        self.colors_buffer[self.index] = sample.colors

        if self.file_name is not None and self.index == 0:
            self.save(self.file_name, append=True)

        self.n_samples = max(self.n_samples, self.index + 1)
        self.index = (self.index + 1) % self.buffer_size

    def save(self, file_name: str, append: bool):
        tokens = self.tokens_buffer
        policy = self.policy_buffer
        reward = self.reward_buffer
        colors = self.colors_buffer

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

        mask = np.any(colors != 0, axis=1)
        indices = np.arange(len(colors))[mask]

        n_samples = min(len(indices), self.buffer_size)

        indices = indices[-n_samples:]

        self.tokens_buffer[:n_samples] = tokens[indices]
        self.policy_buffer[:n_samples] = policy[indices]
        self.reward_buffer[:n_samples] = reward[indices]
        self.colors_buffer[:n_samples] = colors[indices]

        self.n_samples = n_samples
        self.index = self.n_samples % self.buffer_size


def load_batch(file_names, shuffle: bool):
    for i, file_name in enumerate(file_names):
        with np.load(file_name) as data:
            if i == 0:
                tokens = data['t']
                policy = data['p']
                reward = data['r']
                colors = data['c']
            else:
                tokens = np.concatenate([data['t'], tokens])
                policy = np.concatenate([data['p'], policy])
                reward = np.concatenate([data['r'], reward])
                colors = np.concatenate([data['c'], colors])

    indices = np.arange(tokens.shape[0])
    if shuffle:
        np.random.shuffle(indices)

    tokens = tokens[indices]
    policy = policy[indices]
    reward = reward[indices]
    colors = colors[indices]

    mask = np.any(tokens != 0, axis=2)

    return Batch(tokens, mask, policy, reward, colors)


def save_batch(batch: Batch, file_name):
    save_dict = {
        't': batch.tokens,
        'p': batch.policy,
        'r': batch.reward,
        'c': batch.colors,
    }
    np.savez(file_name, **save_dict)
