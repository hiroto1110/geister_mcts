from dataclasses import dataclass, asdict
import time
import socket
import threading
import queue

from tqdm import tqdm
import click

import numpy as np
import jax
from sklearn.linear_model import LinearRegression

import wandb

from config import RunConfig, AgentConfig
from communication import EncryptedCommunicator
from messages import (
    MessageActorInitClient, MessageActorInitServer,
    MessageLeanerInitServer, LearningJob, MessageLearningRequest, LearningJobResult, MessageLearningJobResult,
    MessageMatchResult, MessageNextMatch, MatchInfo, SnapshotInfo, SNAPSHOT_INFO_NAOTTI
)

from match_makers import MatchMaker
from batch import save, is_won, get_reward
from network.checkpoints import Checkpoint, CheckpointManager


@dataclass
class StepSummary:
    is_league_member: bool
    log: dict


@dataclass(eq=False)
class Snapshot:
    name: str
    ckpt: Checkpoint

    def __eq__(self, __value: "Snapshot") -> bool:
        return __value.name == self.name and __value.ckpt.step == self.ckpt.step

    @property
    def info(self):
        return SnapshotInfo(self.name, self.ckpt.step)


class Agent:
    def __init__(self, config: AgentConfig, run_config: RunConfig) -> None:
        self.config = config
        self.run_config = run_config

        self.ckpt_manager = CheckpointManager(self.agent_dir, run_config.ckpt_options)
        self.match_maker: MatchMaker[SnapshotInfo] = config.create_match_maker()
        self.replay_buffer = run_config.create_replay_buffer()

        model, params = config.init_params.create_model_and_params()

        self.current = Snapshot(self.name, Checkpoint(0, model, params))
        self.snapshots: list[Snapshot] = [self.current]

        self.lastest_games: dict[SnapshotInfo, list[np.ndarray]] = {}

        if self.name in config.opponent_names:
            self.match_maker.add_agent(self.current.info)

        if SNAPSHOT_INFO_NAOTTI.name in config.opponent_names:
            self.match_maker.add_agent(SNAPSHOT_INFO_NAOTTI)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def opponent_names(self) -> list[str]:
        return self.config.opponent_names

    @property
    def agent_dir(self) -> str:
        return f'{self.run_config.project_dir}/{self.name}'

    @property
    def replay_buffer_path(self) -> str:
        return f'{self.run_config.project_dir}/{self.name}/replay.npy'

    def next_match(self) -> MatchInfo:
        return MatchInfo(
            player=self.current.info,
            opponent=self.match_maker.next_match()
        )

    def should_do_add_to_replay_buffer(self, match: MatchInfo) -> bool:
        if match.player.name not in self.config.replay_buffer_sharing:
            return False

        return self.config.replay_buffer_sharing[match.player.name] >= np.random.random()

    def apply_match_result(self, match: MatchInfo, samples: np.ndarray):
        if self.should_do_add_to_replay_buffer(match):
            self.replay_buffer.add_sample(samples)

        if match.player.name == self.name:
            if match.opponent not in self.lastest_games:
                self.lastest_games[match.opponent] = []

            self.lastest_games[match.opponent].append(samples)

            for sample in samples:
                self.match_maker.apply_match_result(match.opponent, is_won(sample))

    def add_snaphost(self, snapshot: Snapshot):
        if snapshot.name == self.name:
            self.snapshots.append(self.current)

        if snapshot.name in self.opponent_names:
            self.match_maker.add_agent(snapshot.info)

    def next_step(self) -> StepSummary:
        win_rates = self.match_maker.get_win_rates()
        is_league_member = self.config.condition_for_keeping_snapshots.is_league_member(
            win_rates, self.current.ckpt.step
        )

        lastest_games = np.stack(sum(self.lastest_games.values(), start=[]), axis=0)
        lastest_games = lastest_games.astype(np.uint8)

        save(self.replay_buffer_path, lastest_games, append=True)

        log = {}

        for opponent in self.lastest_games:
            label = f'{opponent.name}-{opponent.step}'
            lastest_games_opponent = np.stack(self.lastest_games[opponent]).astype(np.uint8)

            count = len(lastest_games_opponent) / len(lastest_games)
            log[f'game_count/{label}'] = count

            won_in_series = np.mean(is_won(lastest_games_opponent), axis=0)

            lr = LinearRegression()
            lr.fit(np.arange(len(won_in_series)).reshape(-1, 1), won_in_series)

            log[f'win_rate_coefficient/{label}'] = lr.coef_[0]
            log[f'win_rate_intercept/{label}'] = lr.intercept_

        for i in range(7):
            log[f'game_result/{i}'] = np.mean(get_reward(lastest_games) == i)

        for i in range(len(win_rates)):
            if win_rates[i] == 0:
                continue

            opponent = self.match_maker.agents[i]
            label = f'{opponent.name}-{opponent.step}'
            log[f'win_rate/{label}'] = win_rates[i]

        self.lastest_games.clear()

        return StepSummary(is_league_member, log)

    @property
    def minibatch_size(self) -> int:
        return self.config.training.batch_size * self.config.training.num_batches

    def has_enough_samples(self) -> bool:
        return len(self.replay_buffer) > self.minibatch_size

    def create_learning_job(self) -> LearningJob:
        return LearningJob(self.name, self.replay_buffer.get_minibatch(self.minibatch_size))

    def apply_learning_job_result(self, result: LearningJobResult):
        self.current = Snapshot(self.name, result.ckpt)
        self.ckpt_manager.save(result.ckpt)


class AgentManager:
    def __init__(self, config: RunConfig):
        self.config = config

        self.agents: dict[str, Agent] = {agent.name: Agent(agent, config) for agent in config.agents}

    def create_init_learner_message(self) -> MessageLeanerInitServer:
        ckpts = {
            name: agent.current.ckpt
            for name, agent in self.agents.items()
        }
        return MessageLeanerInitServer(self.config, ckpts)

    def create_init_actor_message(self, n_processes: int) -> MessageActorInitServer:
        snapshots = {
            name: [s.ckpt for s in agent.snapshots] + [agent.current.ckpt]
            for name, agent in self.agents.items()
        }

        n_matches = n_processes + 10

        allocations_list = [[agent] * int(n_matches * agent.processes_allocation_ratio) for agent in self.config.agents]
        allocations = sum(allocations_list, start=[])

        return MessageActorInitServer(
            self.config,
            snapshots=snapshots,
            matches=[self.init_match(agent.name) for agent in allocations]
        )

    def init_match(self, agent_name: str) -> MatchInfo:
        return self.agents[agent_name].next_match()

    def next_match(self, prev_match: MatchInfo) -> MatchInfo:
        return self.agents[prev_match.player.name].next_match()

    def apply_match_result(self, match: MatchInfo, samples: np.ndarray):
        for agent in self.agents.values():
            agent.apply_match_result(match, samples)

    def create_learning_jobs(self) -> list[LearningJob]:
        return [agent.create_learning_job() for agent in self.agents.values() if agent.has_enough_samples()]

    def apply_learning_job_result(self, result: LearningJobResult) -> dict:
        self.agents[result.agent_name].apply_learning_job_result(result)

        return {
            f'{result.agent_name}/loss': result.loss,
            f'{result.agent_name}/loss policy': result.loss_policy,
            f'{result.agent_name}/loss value': result.loss_value,
            f'{result.agent_name}/loss color': result.loss_color,
        }

    def add_snapshot(self, snapshot: Snapshot):
        for agent in self.agents.values():
            agent.add_snaphost(snapshot)

    def next_step(self) -> dict:
        log = {}

        for agent in self.agents.values():
            summary = agent.next_step()

            log |= {f'{agent.name}/{k}': v for k, v in summary.log.items()}

            if summary.is_league_member:
                self.add_snapshot(agent.current)

        return log


def handle_client_actor(
    sock: socket.socket,
    communicator: EncryptedCommunicator,
    match_result_queue: queue.Queue,
    agent_manager: AgentManager,
):
    try:
        _handle_client_actor(sock, communicator, match_result_queue, agent_manager)
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        sock.close()
        print("Actor client is disconnected")


def _handle_client_actor(
    sock: socket.socket,
    communicator: EncryptedCommunicator,
    match_result_queue: queue.Queue,
    agent_manager: AgentManager,
):
    init_msg_client = communicator.recv_json_obj(sock, MessageActorInitClient)

    init_msg_server = agent_manager.create_init_actor_message(init_msg_client.n_processes)
    communicator.send_json_obj(sock, init_msg_server)

    sent_steps = {name: [s.step for s in snapshots] for name, snapshots in init_msg_server.snapshots.items()}

    while result_msg := communicator.recv_json_obj(sock, MessageMatchResult):
        match_result_queue.put(result_msg)

        next_match = agent_manager.next_match(prev_match=result_msg.match)

        updated = {name: [] for name in agent_manager.agents}

        for info in [next_match.player, next_match.opponent]:
            if info.name not in sent_steps:
                continue
            if info.step in sent_steps[info.name]:
                continue

            sent_steps[info.name].append(info.step)
            updated[info.name].append(agent_manager.agents[info.name].ckpt_manager.load(info.step))

        communicator.send_json_obj(sock, MessageNextMatch(next_match, updated))


def wait_accept(
    server: socket.socket,
    communicator: EncryptedCommunicator,
    match_result_queue: queue.Queue,
    agent_manager: AgentManager,
):
    print("Waiting actor client...")

    while True:
        client_sock, address = server.accept()
        print(f"Actor client from {address}")

        args = client_sock, communicator, match_result_queue, agent_manager
        thread = threading.Thread(target=handle_client_actor, args=args)
        thread.start()


def start(
    server: socket.socket,
    config_path: str,
    password: str,
):
    config = RunConfig.from_json_file(config_path)
    communicator = EncryptedCommunicator(password)

    print("Waiting learner client...")
    learner, address = server.accept()
    print(f"Learner client from {address}")

    print("Initializing agents")
    agent_manager = AgentManager(config)

    print("Sending a init message to a learner client")
    communicator.send_json_obj(learner, agent_manager.create_init_learner_message())

    match_result_queue = queue.Queue()

    print("Initilizing actor client hub")
    args = server, communicator, match_result_queue, agent_manager
    thread = threading.Thread(target=wait_accept, args=args)
    thread.start()

    is_waiting_parameter_update = False

    if config.wandb_log:
        wandb.init(project=config.project_name, config=asdict(config))

    for s in range(100000000):
        # recv_match_result
        for i in tqdm(range(config.update_period), desc='Collecting'):
            while match_result_queue.empty():
                time.sleep(0.1)

            result: MessageMatchResult = match_result_queue.get()
            agent_manager.apply_match_result(result.match, result.samples)

        log = agent_manager.next_step()

        if is_waiting_parameter_update:
            # recv_updated_params
            learning_result_msg = communicator.recv_json_obj(learner, MessageLearningJobResult)

            for learning_result in learning_result_msg.results:
                log |= agent_manager.apply_learning_job_result(learning_result)

        if config.wandb_log:
            wandb.log(log)

        # send_training_minibatch
        jobs = agent_manager.create_learning_jobs()

        if len(jobs) > 0:
            communicator.send_json_obj(learner, MessageLearningRequest(jobs))
            is_waiting_parameter_update = True


@click.command()
@click.argument('host', type=str)
@click.argument('port', type=int)
@click.argument('password', type=str)
@click.argument('config_path', type=str)
def main(host: str, port: int, config_path: str, password: str):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        server.bind((host, port))
        server.listen()

        jax.config.update('jax_platform_name', 'cpu')
        with jax.default_device(jax.devices("cpu")[0]):
            start(server, config_path, password)

    except Exception as e:
        server.close()
        raise e


if __name__ == "__main__":
    main()
