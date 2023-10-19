import os
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

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
    policy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    reward: np.ndarray = field(default_factory=lambda: np.zeros(0))
    colors: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def create_inputs(self, create_x_state: bool):
        mask = np.any(self.tokens != 0, axis=2)

        if create_x_state:
            pos_history = [create_pos_history_from_tokens(self.tokens[i]) for i in range(self.tokens.shape[0])]
            pos_history = jnp.array(pos_history)
            x_states = pos_history_to_states(pos_history)
        else:
            x_states = None

        return self.tokens, x_states, mask, self.policy, self.reward, self.colors


def pos_history_to_states(pos_history: np.ndarray) -> jnp.ndarray:
    x_states = jax.nn.one_hot(pos_history, 37, axis=-2)
    x_states = x_states[..., :36, :]
    x_states = x_states.reshape(-1, 200, 6, 6, 16)

    return x_states


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

    def get_last__minibatch(self, batch_size):
        i = (self.index - batch_size) % self.buffer_size
        indices = np.arange(i, i + batch_size)

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

        tokens[:, :, game.Token.T] = np.clip(tokens[:, :, game.Token.T], 0, 199)

        return Batch(tokens, policy, reward, colors)

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

        n_samples = int(np.sum(np.any(colors != 0, axis=1)))
        n_samples = min(n_samples, self.buffer_size)

        self.tokens_buffer[:n_samples] = tokens[:n_samples]
        self.policy_buffer[:n_samples] = policy[:n_samples]
        self.reward_buffer[:n_samples] = reward[:n_samples]
        self.colors_buffer[:n_samples] = colors[:n_samples]

        self.n_samples = n_samples
        self.index = self.n_samples % self.buffer_size
