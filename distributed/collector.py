from dataclasses import dataclass, asdict
import time
import socket
import threading
import queue

from tqdm import tqdm
import click

import numpy as np
import jax

import wandb

from config import RunConfig
from communication import EncryptedCommunicator
from messages import (
    MessageActorInitClient, MessageActorInitServer,
    MessageLeanerInitServer, MessageLearningRequest, MessageLearningResult,
    MessageMatchResult, MessageNextMatch, MatchInfo, MatchResult
)

from batch import ReplayBuffer, save, is_won
from network.checkpoints import Checkpoint, CheckpointManager
import match_makers
import training_logger


@dataclass
class Agent:
    name: str
    ckpt: Checkpoint

    def __eq__(self, __value: "Agent") -> bool:
        return __value.name == self.name and __value.ckpt.step == self.ckpt.step


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
        matches=[MatchInfo(agent_manager.next_match()) for i in range(init_msg_client.n_processes)]
    )
    communicator.send_json_obj(sock, init_msg_server)

    while result_msg := communicator.recv_json_obj(sock, MessageMatchResult):
        match_result_queue.put(result_msg.result)

        next_match = MatchInfo(agent_manager.next_match())

        if result_msg.step == agent_manager.ckpt.step:
            next_msg = MessageNextMatch(next_match, ckpt=None)
        else:
            next_msg = MessageNextMatch(next_match, ckpt=agent_manager.ckpt)

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

    print(f"Loading ReplayBuffer from {config.load_replay_buffer_path}")
    buffer = ReplayBuffer(
        buffer_size=config.buffer_size,
        sample_shape=(config.series_length,),
        seq_length=config.tokens_length
    )
    buffer.load(config.load_replay_buffer_path)

    checkpoint_manager = CheckpointManager(config.ckpt_dir, options=config.ckpt_options)

    print("Preparing model and params")
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
    checkpoint_manager.save(agent_manager.ckpt)

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

            result: MatchResult = match_result_queue.get()
            agent_manager.apply_match_result(result)
            buffer.add_sample(np.stack(result.samples))

        last_batch = buffer.get_last_minibatch(config.update_period)
        save(config.save_replay_buffer_path, last_batch, append=True)

        win_rate = agent_manager.next_step()
        print(win_rate)

        if is_waiting_parameter_update:
            # recv_updated_params
            result_msg = communicator.recv_json_obj(client, MessageLearningResult)
            agent_manager.ckpt = result_msg.ckpt
            checkpoint_manager.save(agent_manager.ckpt)

            if config.wandb_log:
                log_dict = asdict(result_msg.losses) | training_logger.create_log(
                    win_rates=win_rate,
                    last_games=last_batch
                )
                wandb.log(log_dict)

        if len(buffer) >= config.batch_size * config.num_batches:
            # send_training_minibatch
            train_batch = buffer.get_minibatch(config.batch_size * config.num_batches)
            communicator.send_json_obj(client, MessageLearningRequest(train_batch))
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
