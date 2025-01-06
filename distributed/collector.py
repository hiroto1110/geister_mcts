from __future__ import annotations

from dataclasses import asdict
import time
import socket
import threading
import multiprocessing
import queue

from tqdm import tqdm
import click

import numpy as np
import jax

import wandb

from distributed.config import RunConfig, AgentConfig
from distributed.communication import EncryptedCommunicator
from distributed.messages import (
    MessageActorInitClient, MessageActorInitServer,
    MessageLeanerInitServer, MessageLearningRequest, MessageLearningResult,
    MessageMatchResult, MessageNextMatch, MatchInfo
)
from distributed.actor import start_selfplay_process

from players import PlayerMCTSConfig
from players.base import PlayerConfig
from players.naotti import PlayerNaotti2020Config
from match_makers import MatchMaker
from batch import ReplayBuffer, save, FORMAT_X5_PVC
from network.checkpoints import Checkpoint, CheckpointManager

ctx = multiprocessing.get_context('spawn')


def batch_to_reward(batch: np.ndarray) -> np.ndarray:
    return FORMAT_X5_PVC.get_feature(batch, FORMAT_X5_PVC.indices.V)


class TestProcess:
    def __init__(self, config: RunConfig, num_init_matches: int = 100):
        self.config = config
        self.num_init_matches = num_init_matches

        self.test_match_request_queue = ctx.Queue()
        self.test_match_result_queue = ctx.Queue()

        self.current_player: PlayerMCTSConfig = None
    
    def set_current_player_config(self, player: PlayerMCTSConfig):
        self.current_player = player
        self.test_match = MatchInfo(
            self.current_player,
            PlayerNaotti2020Config(depth_min=4, depth_max=7, num_random_ply=6)
        )

        while not self.test_match_request_queue.empty():
            self.test_match_request_queue.get()

        for i in range(self.num_init_matches):
            self.test_match_request_queue.put(self.test_match)

    def get_game_results(self) -> list[MessageMatchResult]:
        results = []

        while not self.test_match_result_queue.empty():
            results.append(self.test_match_result_queue.get())

        return results

    def start(self):
        process = ctx.Process(
            target=start_selfplay_process,
            args=(
                self.test_match_request_queue,
                self.test_match_result_queue,
                self.config.project_dir,
                self.config.tokens_length,
                0
            )
        )
        process.start()


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        test_process: TestProcess,
        ckpt_manager: CheckpointManager,
        replay_buffer: ReplayBuffer,
        replay_buffer_path: str,
    ) -> None:

        self.config = config
        self.test_process = test_process
        self.ckpt_manager = ckpt_manager
        self.replay_buffer = replay_buffer
        self.replay_buffer_path = replay_buffer_path

        self.match_maker: MatchMaker[PlayerConfig] = config.create_match_maker()

        model, params = config.init_params.create_model_and_params()

        self.current = Checkpoint(0, model, params)
        self.snapshots: list[Checkpoint] = [self.current]

        self.ckpt_manager.save(self.current)

        self.lastest_games: dict[PlayerConfig, list[np.ndarray]] = {}

        self.add_current_ckpt_to_matching_pool()

        self.test_process.set_current_player_config(
            self.create_current_player_config()
        )
        self.test_process.start()

    @property
    def name(self) -> str:
        return "main"

    def create_current_player_config(self) -> PlayerMCTSConfig:
        return PlayerMCTSConfig(
            base_name=self.name,
            step=self.current.step,
            mcts_params=self.config.mcts_params,
        )

    def add_current_ckpt_to_matching_pool(self):
        config = self.create_current_player_config()
        self.match_maker.add_agent(config)

    def next_match(self) -> MatchInfo:
        config = self.create_current_player_config()

        return MatchInfo(
            player=config,
            opponent=self.match_maker.next_match()
        )

    def apply_match_result(self, result: MessageMatchResult):
        match = result.match

        self.replay_buffer.add_sample(result.sample_p)
        self.replay_buffer.add_sample(result.sample_o)

        if match.opponent not in self.lastest_games:
            self.lastest_games[match.opponent] = []

        self.lastest_games[match.opponent].append(result.sample_p)

        reward = batch_to_reward(result)
        is_won = reward > 3

        self.match_maker.apply_match_result(match.opponent, is_won)

    def next_step(self) -> dict[str, float]:
        win_rates = self.match_maker.get_win_rates()
        is_league_member = self.config.condition_for_keeping_snapshots.is_league_member(
            win_rates, self.current.step
        )

        if is_league_member:
            self.add_current_ckpt_to_matching_pool()

        lastest_games = np.concatenate(sum(self.lastest_games.values(), start=[]), axis=0)
        lastest_games = lastest_games.astype(np.uint8)
        lastest_games = lastest_games.reshape(-1, lastest_games.shape[-1])

        save(self.replay_buffer_path, lastest_games, append=True)

        log = {}

        reward = batch_to_reward(lastest_games)

        for i in range(7):
            log[f'game_result/train_{i}'] = np.mean(reward == i)
        
        for i in range(len(win_rates)):
            if win_rates[i] == 0:
                continue

            opponent = self.match_maker.agents[i]
            label = opponent.name

            log[f'win_rate/{label}'] = win_rates[i]

            if opponent not in self.lastest_games:
                log[f'game_count/{label}'] = 0
            else:
                lastest_games_opponent = np.stack(self.lastest_games[opponent]).astype(np.uint8)
                log[f'game_count/{label}'] = len(lastest_games_opponent) / len(lastest_games)  

        self.lastest_games.clear()

        self.test_process.set_current_player_config(self.create_current_player_config())

        test_results = self.test_process.get_game_results()
        test_games = np.stack([result.sample_p for result in test_results]).astype(np.uint8)

        reward = batch_to_reward(test_games)

        for i in range(7):
            log[f'game_result/test_{i}'] = np.mean(reward == i)

        log[f'win_rate/test'] = np.mean(reward > 3)
        log[f'game_count/test'] = len(test_games)

        return log

    @property
    def minibatch_size(self) -> int:
        return self.config.training.batch_size * self.config.training.num_batches

    def has_enough_samples(self) -> bool:
        return len(self.replay_buffer) > self.minibatch_size

    def create_learning_request(self) -> MessageLearningRequest:
        return MessageLearningRequest(self.replay_buffer.get_minibatch(self.minibatch_size))

    def apply_learning_result(self, result: MessageLearningResult) -> dict[str: float]:
        self.current = result.ckpt
        self.ckpt_manager.save(result.ckpt)

        return {
            f'loss': result.loss,
            f'loss policy': result.loss_policy,
            f'loss value': result.loss_value,
            f'loss color': result.loss_color,
        }


def handle_client_actor(
    sock: socket.socket,
    communicator: EncryptedCommunicator,
    match_result_queue: queue.Queue,
    agent: Agent,
    config: RunConfig,
):
    try:
        _handle_client_actor(sock, communicator, match_result_queue, agent, config)
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
    agent: Agent,
    config: RunConfig,
):
    init_msg_client = communicator.recv_json_obj(sock, MessageActorInitClient)

    init_msg_server = MessageActorInitServer(
        tokens_length=config.tokens_length,
        snapshots=agent.snapshots + [agent.current],
        matches=[agent.next_match() for _ in range(init_msg_client.n_processes + 10)]
    )

    communicator.send_json_obj(sock, init_msg_server)

    sent_steps = []

    while result_msg := communicator.recv_json_obj(sock, MessageMatchResult):
        match_result_queue.put(result_msg)

        next_match = agent.next_match()

        updated = []

        for player_config in [next_match.player, next_match.opponent]:
            step = player_config.necessary_checkpoint_step
            if (step is None) or (step in sent_steps):
                continue

            sent_steps.append(step)

            ckpt = agent.ckpt_manager.load(step)
            if ckpt is not None:
                updated.append(ckpt)

        communicator.send_json_obj(sock, MessageNextMatch(next_match, updated))


def wait_accept(
    server: socket.socket,
    communicator: EncryptedCommunicator,
    match_result_queue: queue.Queue,
    agent: Agent,
    config: RunConfig,
):
    print("Waiting actor client...")

    while True:
        client_sock, address = server.accept()
        print(f"Actor client from {address}")

        args = client_sock, communicator, match_result_queue, agent, config
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
    agent = Agent(
        config.agent,
        test_process=TestProcess(config),
        ckpt_manager=CheckpointManager(config.project_dir, config.ckpt_options),
        replay_buffer=config.create_replay_buffer(),
        replay_buffer_path=f"{config.project_dir}/replay_buffer.npy",
    )

    print("Sending a init message to a learner client")
    communicator.send_json_obj(learner, MessageLeanerInitServer(config, agent.current))

    match_result_queue = queue.Queue()

    print("Initilizing actor client hub")
    args = server, communicator, match_result_queue, agent, config
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
            agent.apply_match_result(result)

        log = agent.next_step()

        if is_waiting_parameter_update:
            # recv_updated_params
            learning_result_msg = communicator.recv_json_obj(learner, MessageLearningResult)

            log |= agent.apply_learning_result(learning_result_msg)

        if config.wandb_log:
            wandb.log(log)

        # send_training_minibatch
        if agent.has_enough_samples():
            communicator.send_json_obj(learner, agent.create_learning_request())
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
        start(server, config_path, password)

    except Exception as e:
        server.close()
        raise e


if __name__ == "__main__":
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'

    main()
