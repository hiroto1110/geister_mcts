import collections
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class Sample:
    tokens: List[Tuple]
    mcts_policy: List[float]
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

        tokens = np.array([s.tokens for s in samples])
        mcts_policy = np.array([s.mcts_policy for s in samples])
        rewards = np.array([s.reward for s in samples])
        pieces = np.array([s.pieces for s in samples])

        return tokens, mcts_policy, rewards, pieces

    def add_record(self, record):
        for sample in record:
            self.buffer.append(sample)
