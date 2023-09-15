from dataclasses import dataclass, astuple
import numpy as np
import geister as game


@dataclass
class Sample:
    tokens: np.ndarray
    mask: np.ndarray
    policy: np.ndarray
    reward: int
    pieces: np.ndarray


@dataclass
class Batch:
    tokens: np.ndarray
    mask: np.ndarray
    policy: np.ndarray
    reward: np.ndarray
    pieces: np.ndarray

    def astuple(self):
        return astuple(self)


class ReplayBuffer:
    def __init__(self, buffer_size, seq_length):
        self.buffer_size = buffer_size
        self.index = 0
        self.n_samples = 0

        self.tokens_buffer = np.zeros((buffer_size, seq_length, game.TOKEN_SIZE), dtype=np.uint8)
        self.mask_buffer = np.zeros((buffer_size, seq_length), dtype=np.uint8)
        self.policy_buffer = np.zeros((buffer_size, seq_length), dtype=np.uint8)
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.int8)
        self.pieces_buffer = np.zeros((buffer_size, 8), dtype=np.uint8)

    def __len__(self):
        return self.n_samples

    def get_last__minibatch(self, batch_size):
        i = (self.index - batch_size) % self.buffer_size

        tokens = self.tokens_buffer[i: i+batch_size]
        mask = self.mask_buffer[i: i+batch_size]
        policy = self.policy_buffer[i: i+batch_size]
        reward = self.reward_buffer[i: i+batch_size]
        pieces = self.pieces_buffer[i: i+batch_size]

        return Batch(tokens, mask, policy, reward, pieces)

    def get_minibatch(self, batch_size):
        indices = np.random.choice(range(self.n_samples), size=batch_size)

        tokens = self.tokens_buffer[indices]
        mask = self.mask_buffer[indices]
        policy = self.policy_buffer[indices]
        reward = self.reward_buffer[indices]
        pieces = self.pieces_buffer[indices]

        return Batch(tokens, mask, policy, reward, pieces)

    def add_sample(self, sample: Sample):
        self.tokens_buffer[self.index] = sample.tokens
        self.mask_buffer[self.index] = sample.mask
        self.policy_buffer[self.index] = sample.policy
        self.reward_buffer[self.index] = sample.reward
        self.pieces_buffer[self.index] = sample.pieces

        self.n_samples = max(self.n_samples, self.index + 1)
        self.index = (self.index + 1) % self.buffer_size

    def save(self, save_dir):
        np.save(save_dir + '/tokens.npy', self.tokens_buffer)
        np.save(save_dir + '/mask.npy', self.mask_buffer)
        np.save(save_dir + '/policy.npy', self.policy_buffer)
        np.save(save_dir + '/reward.npy', self.reward_buffer)
        np.save(save_dir + '/pieces.npy', self.pieces_buffer)

    def load(self, save_dir):
        self.tokens_buffer = np.load(save_dir + '/tokens.npy')
        self.mask_buffer = np.load(save_dir + '/mask.npy')
        self.policy_buffer = np.load(save_dir + '/policy.npy')
        self.reward_buffer = np.load(save_dir + '/reward.npy')
        self.pieces_buffer = np.load(save_dir + '/pieces.npy')

        self.n_samples = int(np.sum(self.mask_buffer[:, 0]))
        self.index = int(self.n_samples % self.buffer_size)
