from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class MatchResult:
    agent_id: int
    win: bool


class FSP:
    def __init__(self, n_agents: int, match_buffer_size=1000, p=2) -> None:
        self.n_agents = n_agents
        self.match_buffer_size = match_buffer_size
        self.agent_match_deques = [deque(maxlen=match_buffer_size) for _ in range(n_agents)]

        self.score_func = lambda x: (1 - x) ** p

    def next_match(self) -> int:
        if self.n_agents == 1:
            return 0

        for i in range(self.n_agents):
            if len(self.agent_match_deques[i]) < self.match_buffer_size:
                return i

        win_rate = [np.mean(match_deque) for match_deque in self.agent_match_deques]
        win_rate = np.array(win_rate)
        score = self.score_func(win_rate)

        score = score / score.sum()

        return np.random.choice(range(self.n_agents), p=score)

    def apply_match_result(self, agent_id: int, win: bool):
        self.agent_match_deques[agent_id].append(int(win))

    def is_winning_all_agents(self, win_rate_threshold: float) -> bool:
        win_rate = [np.mean(match_deque) for match_deque in self.agent_match_deques]
        win_rate = np.array(win_rate)

        for match_deque in self.agent_match_deques:
            if len(match_deque) < self.match_buffer_size:
                return False, win_rate

        return np.all(win_rate > win_rate_threshold), win_rate

    def add_agent(self):
        self.n_agents += 1
        self.agent_match_deques.append(deque(maxlen=self.match_buffer_size))


def main():
    true_p = np.array([0.9, 0.7, 0.61, 0.59, 0.58])

    fsp = FSP(n_agents=true_p.shape[0], match_buffer_size=500, p=4)

    for i in range(2000):
        id = fsp.next_match()
        win = true_p[id] > np.random.random()

        fsp.apply_match_result(id, win)
    print(fsp.is_winning_all_agents(0.55))

    true_p = np.concatenate([true_p, [0.5]])
    fsp.add_agent()

    for i in range(3000):
        id = fsp.next_match()
        win = true_p[id] > np.random.random()

        fsp.apply_match_result(id, win)
    print(fsp.is_winning_all_agents(0.55))


if __name__ == "__main__":
    main()
