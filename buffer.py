from dataclasses import dataclass
import numpy as np
import geister as game


@dataclass
class Sample:
    tokens: np.ndarray
    policy: np.ndarray
    reward: int
    pieces: np.ndarray


class ReplayBuffer:
    buffer_size: int
    index: int
    n_samples: int

    tokens_buffer: np.ndarray
    policy_buffer: np.ndarray
    reward_buffer: np.ndarray
    pieces_buffer: np.ndarray

    def __init__(self, buffer_size, seq_length):
        self.buffer_size = buffer_size
        self.index = 0
        self.n_samples = 0

        self.tokens_buffer = np.zeros((buffer_size, seq_length, game.TOKEN_SIZE), dtype=np.uint8)
        self.policy_buffer = np.zeros((buffer_size, seq_length), dtype=np.uint8)
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.int8)
        self.pieces_buffer = np.zeros((buffer_size, 8), dtype=np.uint8)

    def __len__(self):
        return self.n_samples

    def get_minibatch(self, batch_size):
        indices = np.random.choice(range(self.n_samples), size=batch_size)

        tokens = self.tokens_buffer[indices]
        policy = self.policy_buffer[indices]
        reward = self.reward_buffer[indices]
        pieces = self.pieces_buffer[indices]

        return tokens, policy, reward, pieces

    def add_sample(self, sample: Sample):
        self.tokens_buffer[self.index] = sample.tokens
        self.policy_buffer[self.index] = sample.policy
        self.reward_buffer[self.index] = sample.reward
        self.pieces_buffer[self.index] = sample.pieces

        self.n_samples = max(self.n_samples, self.index + 1)
        self.index = (self.index + 1) % self.buffer_size

    def save(self, save_dir):
        np.save(save_dir + '/tokens.npy', self.tokens_buffer)
        np.save(save_dir + '/policy.npy', self.policy_buffer)
        np.save(save_dir + '/reward.npy', self.reward_buffer)
        np.save(save_dir + '/pieces.npy', self.pieces_buffer)

    def load(self, save_dir):
        self.tokens_buffer = np.load(save_dir + '/tokens.npy')
        self.policy_buffer = np.load(save_dir + '/policy.npy')
        self.reward_buffer = np.load(save_dir + '/reward.npy')
        self.pieces_buffer = np.load(save_dir + '/pieces.npy')
