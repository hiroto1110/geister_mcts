from collections import deque
from dataclasses import dataclass
import numpy as np
import scipy.stats


@dataclass
class MatchResult:
    agent_id: int
    win: bool


SELFPLAY_ID = -1


class MatchMaker:
    def next_match(self) -> int:
        pass

    def apply_match_result(self, agent_id: int, is_won: bool):
        pass

    def has_enough_matches(self) -> bool:
        pass

    def get_win_rates(self) -> np.ndarray:
        pass

    def add_agent(self):
        pass


class MatchMakerPFSPThompsonSampling(MatchMaker):
    def __init__(self, n_agents: int, selfplay_p=0.5, match_buffer_size=1000) -> None:
        self.n_agents = n_agents
        self.match_buffer_size = match_buffer_size
        self.selfplay_p = selfplay_p

        self.agent_match_deques = deque([], maxlen=match_buffer_size * n_agents)
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
            if (self.won[i] + self.lst[i]) < 100:
                return i

        score = scipy.stats.beta.rvs(self.won + 1, self.lst + 1)
        # print(score, np.argmin(score))
        return np.argmin(score)

    def _delete_last_match(self):
        agent_id, is_won = self.agent_match_deques.pop()

        if is_won:
            self.won[agent_id] -= 1
        else:
            self.lst[agent_id] -= 1

    def apply_match_result(self, agent_id: int, is_won: bool):
        if agent_id == SELFPLAY_ID:
            return

        if len(self.agent_match_deques) == self.agent_match_deques.maxlen:
            self._delete_last_match()

        self.agent_match_deques.appendleft((agent_id, is_won))

        if is_won:
            self.won[agent_id] += 1
        else:
            self.lst[agent_id] += 1

    def has_enough_matches(self) -> bool:
        return True

    def get_win_rates(self) -> np.ndarray:
        return scipy.stats.beta.ppf(
            q=0.2,
            a=self.won + 1,
            b=self.lst + 1
        )

    def add_agent(self):
        self.n_agents += 1
        self.agent_match_deques = deque(
            self.agent_match_deques,
            maxlen=self.match_buffer_size * self.n_agents
        )

        self.won = np.concatenate([self.won, [0]], dtype=np.int32)
        self.lst = np.concatenate([self.lst, [0]], dtype=np.int32)


class MatchMakerFSP(MatchMaker):
    def __init__(self, n_agents: int, selfplay_p=0.5, match_buffer_size=1000, p=2) -> None:
        self.n_agents = n_agents
        self.match_buffer_size = match_buffer_size
        self.selfplay_p = selfplay_p
        self.agent_match_deques = [deque([0], maxlen=match_buffer_size) for _ in range(n_agents)]
        self.p = p

    def next_match(self) -> int:
        if self.n_agents == 0:
            return SELFPLAY_ID

        if np.random.random() < self.selfplay_p:
            return SELFPLAY_ID

        if self.n_agents == 1:
            return 0

        for i in range(self.n_agents):
            if len(self.agent_match_deques[i]) < self.match_buffer_size:
                return i

        win_rate = [np.mean(match_deque) for match_deque in self.agent_match_deques]
        win_rate = np.array(win_rate)
        score = (1 - win_rate) ** self.p

        score = score / score.sum()

        agent_id = np.random.choice(range(self.n_agents), p=score)
        return int(agent_id)

    def apply_match_result(self, agent_id: int, is_won: bool):
        if agent_id == SELFPLAY_ID:
            return

        self.agent_match_deques[agent_id].append(int(is_won))

    def has_enough_matches(self) -> bool:
        for match_deque in self.agent_match_deques:
            if len(match_deque) < self.match_buffer_size:
                return False

        return True

    def get_win_rates(self) -> np.ndarray:
        win_rate = [np.mean(match_deque) for match_deque in self.agent_match_deques]
        return np.array(win_rate)

    def add_agent(self):
        self.n_agents += 1
        self.agent_match_deques.append(deque([0], maxlen=self.match_buffer_size))


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
