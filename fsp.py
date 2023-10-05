from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta


@dataclass
class MatchResult:
    agent_id: int
    win: bool


def underestimation(w, n, q=0.05):
    p = beta.ppf(q, w + 1, (n - w) + 1)
    return p


class FSP:
    def __init__(self, n_agents: int, match_buffer_size=1000, p=2) -> None:
        self.n = np.zeros(n_agents)
        self.w = np.zeros(n_agents)
        self.n_agents = n_agents
        self.match_buffer_size = match_buffer_size

        maxlen = match_buffer_size * n_agents
        self.match_deque = deque(maxlen=maxlen)
        self.agent_match_deques = [deque(maxlen=match_buffer_size) for _ in range(n_agents)]

        self.score_func = lambda x: (1 - x) ** p

    def next_match(self) -> int:
        win_rate = underestimation(self.w, self.n)
        score = self.score_func(win_rate)

        score = score / score.sum()

        return np.random.choice(range(self.n_agents), p=score)

    def apply_match_result(self, agent_id: int, win: bool):
        self.n[agent_id] += 1
        self.w[agent_id] += int(win)

        if len(self.match_deque) == self.match_deque.maxlen:
            result = self.match_deque[0]
            self.n[result.agent_id] -= 1
            self.w[result.agent_id] -= int(result.win)

        self.match_deque.append(MatchResult(agent_id, win))

        self.agent_match_deques[agent_id].append(int(win))

    def is_winning_all_agents(self, win_rate_threshold: float) -> bool:
        win_rate = [np.mean(match_deque) for match_deque in self.agent_match_deques]
        win_rate = np.array(win_rate)

        for matchs in self.agent_match_deques:
            if len(matchs) < self.match_buffer_size:
                return False, win_rate

        return np.all(win_rate > win_rate_threshold), win_rate

    def add_agent(self):
        self.w = np.concatenate([self.w, [0]])
        self.n = np.concatenate([self.n, [0]])
        self.n_agents += 1

        maxlen = self.n_agents * self.match_buffer_size
        self.match_deque = deque(self.match_deque, maxlen=maxlen)

        self.agent_match_deques.append(deque(maxlen=self.match_buffer_size))


def main():
    true_p = np.array([0.9, 0.7, 0.61, 0.59, 0.58])

    fsp = FSP(n_agents=true_p.shape[0], match_buffer_size=500, p=6)

    for i in range(2000):
        id = fsp.next_match()
        win = true_p[id] > np.random.random()

        fsp.apply_match_result(id, win)

        print(fsp.n)
    r = fsp.is_winning_all_agents(0.55)
    print(underestimation(fsp.w, fsp.n))

    true_p = np.concatenate([true_p, [0.5]])
    fsp.add_agent()

    for i in range(2000):
        id = fsp.next_match()
        win = true_p[id] > np.random.random()

        fsp.apply_match_result(id, win)

        print(fsp.n)

    is_w, r = fsp.is_winning_all_agents(0.55)
    print(r)


if __name__ == "__main__":
    main()
