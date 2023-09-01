import collections
from typing import List
from dataclasses import dataclass
import numpy as np


@dataclass
class Sample:
    tokens: np.ndarray
    mcts_policy: np.ndarray
    player: int
    reward: int
    pieces: List[int]


class ReplayBuffer:
    buffer: List[Sample]

    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def get_minibatch(self, batch_size):
        indices = np.random.choice(range(len(self.buffer)), size=batch_size)

        samples = [self.buffer[idx] for idx in indices]

        tokens = np.array([s.tokens for s in samples], dtype=np.uint8)
        mcts_policy = np.array([s.mcts_policy for s in samples], dtype=np.uint8)
        rewards = np.array([s.reward for s in samples], dtype=np.int8)
        pieces = np.array([s.pieces for s in samples], dtype=np.uint8)

        return tokens, mcts_policy, rewards, pieces

    def add_record(self, record):
        self.buffer.append(record)
