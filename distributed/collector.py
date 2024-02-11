from dataclasses import dataclass
import time
import socket
import threading
import queue

from tqdm import tqdm
import click

import numpy as np
import jax

import wandb

from config import RunConfig, AgentConfig
from communication import EncryptedCommunicator
from messages import (
    MessageActorInitClient, MessageActorInitServer,
    MessageLeanerInitServer, LearningJob, MessageLearningRequest, LearningJobResult, MessageLearningJobResult,
    MessageMatchResult, MessageNextMatch, MatchInfo, SnapshotInfo, SNAPSHOT_INFO_SELFPLAY
)

from match_makers import MatchMaker
from batch import save, is_won
from network.checkpoints import Checkpoint, CheckpointManager
import training_logger


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
        self.replay_buffer = config.create_replay_buffer(run_config.series_length, run_config.tokens_length)

        model, params = config.init_params.create_model_and_params()

        self.current = Snapshot(self.name, Checkpoint(0, model, params))
        self.snapshots: list[Snapshot] = [self.current]

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

    def next_match(self, prev_match: MatchInfo = None) -> MessageNextMatch:
        next_match = MatchInfo(
            player=self.current.info,
            opponent=self.match_maker.next_match()
        )

        if prev_match is not None and prev_match.player.step != self.current.step:
            return MessageNextMatch(next_match, self.current.ckpt)
        else:
            return MessageNextMatch(next_match, None)

    def apply_match_result(self, match: MatchInfo, samples: np.ndarray):
        self.replay_buffer.add_sample(samples)

        for sample in samples:
            self.match_maker.apply_match_result(match.opponent, is_won(sample))

    def add_snaphost(self, snapshot: Snapshot):
        if snapshot.name == self.name:
            self.snapshots.append(self.current)

        if snapshot.name in self.opponent_names:
            self.match_maker.add_agent(snapshot.info)

    def next_step(self) -> StepSummary:
        win_rates = self.match_maker.get_win_rates()
        is_league_member = self.config.condition_for_keeping_snapshots.is_league_member(win_rates)

        last_batch = self.replay_buffer.get_last_minibatch(self.run_config.update_period)
        save(self.replay_buffer_path, last_batch, append=True)

        log = training_logger.create_log(win_rates, last_batch)
        for i in range(len(win_rates)):
            opponent = self.match_maker.agents[i]
            label = f'{opponent.name}-{opponent.step}'
            log[f'win_rate/{label}'] = win_rates[i]

        return StepSummary(is_league_member, log)

    @property
    def minibatch_size(self) -> int:
        return self.config.training.batch_size * self.config.training.num_batches

    def has_enough_samples(self) -> bool:
        return len(self.replay_buffer) > self.minibatch_size

    def create_learning_job(self) -> LearningJob:
        return LearningJob(self.name, self.replay_buffer.get_minibatch(self.minibatch_size))

    def apply_learning_job_result(self, result: LearningJobResult):
        self.current = result.ckpt
        self.ckpt_manager.save(result.ckpt)


class AgentManager:
    def __init__(self, config: RunConfig):
        self.config = config

        self.agents: dict[str, Agent] = {agent.name: Agent(agent, config) for agent in config.agents}

    def init_match(self, agent_name: str) -> MessageNextMatch:
        return self.agents[agent_name].next_match()

    def next_match(self, prev_match: MatchInfo) -> MessageNextMatch:
        return self.agents[prev_match.player.name].next_match(prev_match)

    def apply_match_result(self, match: MatchInfo, samples: np.ndarray):
        if match.player.name == SNAPSHOT_INFO_SELFPLAY.name:
            return

        self.agents[match.player.name].apply_match_result(match, samples)

    def create_learning_jobs(self) -> list[LearningJob]:
        return [agent.create_learning_job() for agent in self.agents.values() if agent.has_enough_samples()]

    def apply_learning_job_result(self, result: LearningJobResult):
        self.agents[result.agent_name].apply_learning_job_result(result)

    def add_snapshot(self, snapshot: Snapshot):
        for agent in self.agents.values():
            agent.add_snaphost(snapshot)

    def next_step(self) -> dict:
        log = {}

        for agent in self.agents.values():
            summary = agent.next_step()

            log[agent.name] = summary.log

            if summary.is_league_member:
                self.add_snapshot(agent.current)

        return log


def handle_client_actor(
    sock: socket.socket,
    communicator: EncryptedCommunicator,
    match_result_queue: queue.Queue,
    agent_manager: AgentManager,
    config: RunConfig
):
    try:
        _handle_client_actor(sock, communicator, match_result_queue, agent_manager, config)
    except Exception as e:
        sock.close()
        print(e)


def _handle_client_actor(
    sock: socket.socket,
    communicator: EncryptedCommunicator,
    match_result_queue: queue.Queue,
    agent_manager: AgentManager,
    config: RunConfig
):
    init_msg_client = communicator.recv_json_obj(sock, MessageActorInitClient)

    init_msg_server = MessageActorInitServer(
        config,
        agent_manager.ckpt,
        agent_manager.agents,
        matches=[MatchInfo(agent_manager.init_match()) for i in range(init_msg_client.n_processes)]
    )
    communicator.send_json_obj(sock, init_msg_server)

    while result_msg := communicator.recv_json_obj(sock, MessageMatchResult):
        match_result_queue.put(result_msg)

        next_msg = agent_manager.next_match(prev_match=result_msg.match)

        communicator.send_json_obj(sock, next_msg)
    sock.close()


def wait_accept(
    server: socket.socket,
    communicator: EncryptedCommunicator,
    match_result_queue: queue.Queue,
    agent_manager: AgentManager,
    config: RunConfig
):
    print("Waiting actor client...")

    while True:
        client_sock, address = server.accept()
        print(f"Actor client from {address}")

        args = client_sock, communicator, match_result_queue, agent_manager, config
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
    client, address = server.accept()
    print(f"Learner client from {address}")

    print("Initializing agents")
    agent_manager = AgentManager(config)

    print("Sending a init message to a learner client")
    communicator.send_json_obj(client, MessageLeanerInitServer(config, agent_manager.ckpt))

    match_result_queue = queue.Queue()

    print("Starting actor client hub")
    args = server, communicator, match_result_queue, agent_manager, config
    thread = threading.Thread(target=wait_accept, args=args)
    thread.start()

    is_waiting_parameter_update = False

    if config.wandb_log:
        print("Initilizing wandb project")
        wandb.init(project=config.project_name)

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
            learning_result_msg = communicator.recv_json_obj(client, MessageLearningJobResult)

            for learning_result in learning_result_msg.results:
                agent_manager.apply_learning_job_result(learning_result)

                log[f'train/{learning_result.agent_name}/loss'] = learning_result.loss
                log[f'train/{learning_result.agent_name}/loss policy'] = learning_result.loss_policy
                log[f'train/{learning_result.agent_name}/loss value'] = learning_result.loss_color
                log[f'train/{learning_result.agent_name}/loss color'] = learning_result.loss_value

            if config.wandb_log:
                wandb.log(log)

        # send_training_minibatch
        jobs = agent_manager.create_learning_jobs()

        if len(jobs) > 0:
            communicator.send_json_obj(client, MessageLearningRequest(jobs))
            is_waiting_parameter_update = True


@click.command()
@click.argument('host', type=str)
@click.argument('port', type=int)
@click.argument('config_path', type=str)
@click.argument('password', type=str)
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
