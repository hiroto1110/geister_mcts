import os
from dataclasses import dataclass
import time
import pickle
import socket
import threading
import queue

from tqdm import tqdm
import click

import numpy as np
import jax
import orbax.checkpoint

import wandb

from distributed.config import RunConfig
from distributed.socket_util import EncryptedCommunicator

from batch import ReplayBuffer, save, is_won
from network.train import Checkpoint
import match_makers
import training_logger


@dataclass
class MatchResult:
    samples: list[np.ndarray]
    agent_id: int


@dataclass
class MatchInfo:
    agent_id: int

    def __post_init__(self):
        self.agent_id = int(self.agent_id)


@dataclass
class MessageMatchResult:
    result: MatchResult
    step: int


@dataclass
class MessageNextMatch:
    next_match: MatchInfo
    ckpt: Checkpoint


@dataclass
class MessageLearningRequest:
    ckpt: Checkpoint
    minibatch: np.ndarray


@dataclass
class MessageLearningResult:
    ckpt: Checkpoint
    log: dict


class AgentManager:
    def __init__(
        self,
        match_maker: match_makers.MatchMaker,
        fsp_threshold: float,
        ckpt: Checkpoint
    ) -> None:
        self.match_maker = match_maker
        self.fsp_threshold = fsp_threshold
        self.ckpt = ckpt

        self.agents: list[Checkpoint] = [ckpt]

    def next_match(self):
        agent_index = self.match_maker.next_match()

        if agent_index >= 0:
            return self.agents[agent_index].step
        else:
            return agent_index

    def apply_match_result(self, result: MatchResult):
        if result.agent_id == match_makers.SELFPLAY_ID:
            return

        for i, ckpt in enumerate(self.agents):
            if result.agent_id == ckpt.step:
                agent_index = i
                break
        else:
            raise RuntimeError(f'Checkpoint(step={result.agent_id}) is not found in snapshots')

        for sample in result.samples:
            self.match_maker.apply_match_result(agent_index, is_won(sample))

    def next_step(self) -> np.ndarray:
        win_rates = self.match_maker.get_win_rates()

        if np.all(win_rates > self.fsp_threshold):
            self.match_maker.add_agent()
            self.agents.append(self.ckpt)

        return win_rates


def handle_client_actor(
    sock: socket.socket,
    communicator: EncryptedCommunicator,
    match_result_queue: queue.Queue,
    agent_manager: AgentManager,
    config: RunConfig
):
    communicator.send_bytes(sock, pickle.dumps(config))
    communicator.send_bytes(sock, pickle.dumps(agent_manager.ckpt))

    communicator.send_bytes(sock, pickle.dumps(len(agent_manager.agents)))
    for ckpt_i in agent_manager.agents:
        communicator.send_bytes(sock, pickle.dumps(ckpt_i))

    n_processes: int = pickle.loads(communicator.recv_bytes(sock))

    init_matches = [MatchInfo(agent_manager.next_match()) for i in range(n_processes + 4)]

    communicator.send_bytes(sock, pickle.dumps(init_matches))

    while data := communicator.recv_bytes(sock):
        result_msg: MessageMatchResult = pickle.loads(data)

        match_result_queue.put(result_msg.result)

        next_match = MatchInfo(agent_manager.next_match())

        if result_msg.step == agent_manager.ckpt.step:
            msg = MessageNextMatch(next_match, ckpt=None)
        else:
            msg = MessageNextMatch(next_match, ckpt=agent_manager.ckpt)

        communicator.send_bytes(sock, pickle.dumps(msg))
    sock.close()


def wait_accept(
    server: socket.socket,
    communicator: EncryptedCommunicator,
    match_result_queue: queue.Queue,
    agent_manager: AgentManager,
    config: RunConfig
):
    while True:
        client_sock, address = server.accept()
        print(f"New client from {address}")

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

    client, address = server.accept()
    print(f"Learner client from {address}")
    communicator.send_bytes(client, pickle.dumps(config))

    buffer = ReplayBuffer(
        buffer_size=config.buffer_size,
        sample_shape=(config.series_length,),
        seq_length=config.tokens_length
    )
    buffer.load(config.load_replay_buffer_path)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        os.path.abspath(config.ckpt_dir),
        checkpointer,
        options=config.ckpt_options)

    model, params = config.init_params.create_model_and_params()

    match_maker = match_makers.MatchMaker(
        method=config.match_making,
        n_agents=1,
        selfplay_p=config.selfplay_p,
        match_buffer_size=config.match_maker_buffer_size
    )
    agent_manager = AgentManager(
        match_maker,
        config.fsp_threshold,
        ckpt=Checkpoint(step=0, params=params, model=model),
    )
    agent_manager.ckpt.save(checkpoint_manager)

    match_result_queue = queue.Queue()

    args = server, communicator, match_result_queue, agent_manager, config
    thread = threading.Thread(target=wait_accept, args=args)
    thread.start()

    is_waiting_parameter_update = False

    if config.wandb_log:
        wandb.init(project=config.project_name)

    for s in range(100000000):
        # recv_match_result
        for i in tqdm(range(config.update_period), desc='Collecting'):
            while match_result_queue.empty():
                time.sleep(0.1)

            result: MatchResult = match_result_queue.get()
            agent_manager.apply_match_result(result)
            buffer.add_sample(np.stack(result.samples))

        last_batch = buffer.get_last_minibatch(config.update_period)
        save(config.save_replay_buffer_path, last_batch, append=True)

        win_rate = agent_manager.next_step()
        print(win_rate)

        if is_waiting_parameter_update:
            # recv_updated_params
            msg: MessageLearningResult = pickle.loads(communicator.recv_bytes(client))
            agent_manager.ckpt = msg.ckpt
            agent_manager.ckpt.save(checkpoint_manager)

            if config.wandb_log:
                log_dict = msg.log | training_logger.create_log(
                    win_rates=win_rate,
                    last_games=last_batch
                )
                wandb.log(log_dict)

        if len(buffer) >= config.batch_size * config.num_batches:
            # send_training_minibatch
            train_batch = buffer.get_minibatch(config.batch_size * config.num_batches)
            communicator.send_bytes(client, pickle.dumps(MessageLearningRequest(agent_manager.ckpt, train_batch)))
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
