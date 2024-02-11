from typing import TypeVar, Generic
from collections import deque
from dataclasses import dataclass

import numpy as np
import scipy.stats


T = TypeVar('T')


@dataclass
class MatchMakingMethodBase:
    def next_match(self, latest_games: np.ndarray, individual_latest_games: np.ndarray) -> int:
        pass


class MatchMaker(Generic[T]):
    def __init__(self, method: MatchMakingMethodBase, match_buffer_size=1000) -> None:
        self.match_making_method = method
        self.match_buffer_size = match_buffer_size

        self.agents: list[T] = []
        self.match_deques_individual: list[deque] = []

        self.match_deque = deque([], maxlen=0)
        self.won = np.zeros(0, dtype=np.int32)
        self.lst = np.zeros(0, dtype=np.int32)

    def next_match(self) -> T:
        if len(self.agents) == 0:
            raise RuntimeError("MatchMaker has no agents")

        if len(self.agents) == 1:
            return self.agents[0]

        for i in range(len(self.agents)):
            if len(self.match_deques_individual[i]) < self.match_buffer_size:
                return self.agents[i]

        latest_games = np.stack([self.won, self.lst])

        won = np.array([np.sum(game) for game in self.match_deques_individual])
        individual_latest_games = np.stack([won, self.match_buffer_size - won])

        agent_id = self.match_making_method.next_match(latest_games, individual_latest_games)

        return self.agents[agent_id]

    def apply_match_result(self, agent: T, is_won: bool):
        agent_id = self.agents.index(agent)

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

    def add_agent(self, agent: T):
        self.agents.append(agent)

        self.match_deques_individual.append(deque([0], maxlen=self.match_buffer_size))

        self.match_deque = deque(
            self.match_deque,
            maxlen=self.match_buffer_size * len(self.agents)
        )

        self.won = np.concatenate([self.won, [0]], dtype=np.int32)
        self.lst = np.concatenate([self.lst, [0]], dtype=np.int32)


@dataclass
class ThompsonSampling(MatchMakingMethodBase):
    def next_match(self, latest_games: np.ndarray, individual_latest_games: np.ndarray) -> int:
        score = scipy.stats.beta.rvs(latest_games[0] + 1, latest_games[1] + 1)
        # print(score, np.argmin(score))
        return np.argmin(score)


@dataclass
class PFSP(MatchMakingMethodBase):
    p: float

    def next_match(self, latest_games: np.ndarray, individual_latest_games: np.ndarray) -> int:
        win_rate = individual_latest_games[0] / individual_latest_games.sum(axis=0)
        score = (1 - win_rate) ** self.p

        score = score / score.sum()

        agent_id = np.random.choice(range(len(score)), p=score)
        return int(agent_id)


MatchMakingMethod = ThompsonSampling | PFSP


def main():
    true_p = np.array([0.8, 0.7, 0.61, 0.59, 0.58])

    match_making = ThompsonSampling()
    # match_making = PFSP(p=6)
    match_maker = MatchMaker(match_making, n_agents=true_p.shape[0], selfplay_p=0, match_buffer_size=500)

    for i in range(10000):
        id = match_maker.next_match()
        win = true_p[id] > np.random.random()

        match_maker.apply_match_result(id, win)
    print(match_maker.get_win_rates())
    print(match_maker.won + match_maker.lst)

    true_p = np.concatenate([true_p, [0.5]])
    match_maker.add_agent()

    for i in range(10000):
        id = match_maker.next_match()
        win = true_p[id] > np.random.random()

        match_maker.apply_match_result(id, win)
    print(match_maker.get_win_rates())
    print(match_maker.won + match_maker.lst)


if __name__ == "__main__":
    main()
