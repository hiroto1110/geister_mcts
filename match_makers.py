from collections import deque
from dataclasses import dataclass
import numpy as np
import scipy.stats


@dataclass
class MatchResult:
    agent_id: int
    win: bool


SELFPLAY_ID = -1


class MatchMakingMethod:
    def next_match(self, won: np.ndarray, won_individual: np.ndarray) -> int:
        pass


class MatchMaker:
    def __init__(self, method: MatchMakingMethod, n_agents: int, selfplay_p=0.5, match_buffer_size=1000) -> None:
        self.match_making_method = method
        self.n_agents = n_agents
        self.match_buffer_size = match_buffer_size
        self.selfplay_p = selfplay_p

        self.match_deques_individual = [deque([0], maxlen=match_buffer_size) for _ in range(n_agents)]
        self.match_deque = deque([], maxlen=match_buffer_size * n_agents)
        self.won = np.zeros(n_agents, dtype=np.int32)
        self.lst = np.zeros(n_agents, dtype=np.int32)

    def next_match(self) -> int:
        if self.n_agents == 0:
            return SELFPLAY_ID

        if np.random.random() < self.selfplay_p:
            return SELFPLAY_ID

        if self.n_agents == 1:
            return 0

        for i in range(self.n_agents):
            if len(self.match_deques_individual[i]) < self.match_buffer_size:
                return i

        return self.match_making_method.next_match()

    def apply_match_result(self, agent_id: int, is_won: bool):
        if agent_id == SELFPLAY_ID:
            return

        self.match_deques_individual[agent_id].append(int(is_won))

        if len(self.match_deque) == self.match_deque.maxlen:
            self._delete_last_match()

        self.match_deque.appendleft((agent_id, is_won))

        if is_won:
            self.won[agent_id] += 1
        else:
            self.lst[agent_id] += 1

    def _delete_last_match(self):
        agent_id, is_won = self.match_deque.pop()

        if is_won:
            self.won[agent_id] -= 1
        else:
            self.lst[agent_id] -= 1

    def has_enough_matches(self) -> bool:
        for match_deque in self.match_deques_individual:
            if len(match_deque) < self.match_buffer_size:
                return False

        return True

    def get_win_rates(self) -> np.ndarray:
        win_rate = [np.mean(match_deque) for match_deque in self.match_deques_individual]
        return np.array(win_rate)

    def add_agent(self):
        self.n_agents += 1
        self.match_deques_individual.append(deque([0], maxlen=self.match_buffer_size))

        self.match_deque = deque(
            self.match_deque,
            maxlen=self.match_buffer_size * self.n_agents
        )

        self.won = np.concatenate([self.won, [0]], dtype=np.int32)
        self.lst = np.concatenate([self.lst, [0]], dtype=np.int32)


@dataclass
class MatchMakingMethodThompsonSampling(MatchMakingMethod):
    def next_match(self, n: int, won: np.ndarray, won_individual: np.ndarray) -> int:
        lost = n - won
        score = scipy.stats.beta.rvs(won, lost)
        # print(score, np.argmin(score))
        return np.argmin(score)


@dataclass
class MatchMakingMethodPFSP(MatchMakingMethod):
    p: float

    def next_match(self, n: int, won: np.ndarray, won_individual: np.ndarray) -> int:
        win_rate = won_individual / n
        score = (1 - win_rate) ** self.p

        score = score / score.sum()

        agent_id = np.random.choice(range(self.n_agents), p=score)
        return int(agent_id)


def main():
    true_p = np.array([0.9, 0.7, 0.61, 0.59, 0.58])

    match_maker = MatchMakerPFSPThompsonSampling(n_agents=true_p.shape[0], selfplay_p=0, match_buffer_size=1000)

    for i in range(5000):
        id = match_maker.next_match()
        win = true_p[id] > np.random.random()

        match_maker.apply_match_result(id, win)
    print(match_maker.get_win_rates())
    print(match_maker.won + match_maker.lst)

    true_p = np.concatenate([true_p, [0.5]])
    match_maker.add_agent()

    for i in range(5000):
        id = match_maker.next_match()
        win = true_p[id] > np.random.random()

        match_maker.apply_match_result(id, win)
    print(match_maker.get_win_rates())
    print(match_maker.won + match_maker.lst)


if __name__ == "__main__":
    main()
