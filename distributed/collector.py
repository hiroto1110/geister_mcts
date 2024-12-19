from dataclasses import asdict
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

from distributed.config import RunConfig, AgentConfig
from distributed.communication import EncryptedCommunicator
from distributed.messages import (
    MessageActorInitClient, MessageActorInitServer,
    MessageLeanerInitServer, MessageLearningRequest, MessageLearningResult,
    MessageMatchResult, MessageNextMatch, MatchInfo
)

from players import PlayerMCTSConfig
from players.base import PlayerConfig
from players.strategy import Random as StrategyRandom
from match_makers import MatchMaker
from batch import ReplayBuffer, save, FORMAT_X7_ST_PVC
from network.checkpoints import Checkpoint, CheckpointManager


def batch_to_reward(batch: np.ndarray) -> np.ndarray:
    return FORMAT_X7_ST_PVC.get_feature(batch, FORMAT_X7_ST_PVC.indices.V)


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        ckpt_manager: CheckpointManager,
        replay_buffer: ReplayBuffer,
        replay_buffer_path: str,
    ) -> None:
        self.config = config
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
    
    @property
    def name(self) -> str:
        return "main"

    def create_current_player_config(
        self,
        strategy_p: tuple[float, float, float]
    ) -> PlayerMCTSConfig:
        
        strategy_p = tuple([float(i) for i in strategy_p])

        return PlayerMCTSConfig(
            base_name=self.name,
            step=self.current.step,
            mcts_params=self.config.mcts_params,
            strategy_factory=StrategyRandom(p=strategy_p)
        )

    def add_current_ckpt_to_matching_pool(self):
        config = self.create_current_player_config(strategy_p=[0.35, 0.35, 0.3])

        self.match_maker.add_agent(config)

    def create_init_actor_message(self, n_processes: int) -> MessageActorInitServer:
        snapshots = self.snapshots + [self.current]

        return MessageActorInitServer(
            self.config,
            snapshots=snapshots,
            matches=[self.next_match() for _ in range(n_processes + 10)]
        )

    def next_match(self) -> MatchInfo:
        config = self.create_current_player_config(strategy_p=[0.0, 0.0, 1.0])

        return MatchInfo(
            player=config,
            opponent=self.match_maker.next_match()
        )

    def apply_match_result(self, match: MatchInfo, samples: np.ndarray):
        for i in range(len(samples)):
            self.replay_buffer.add_sample(samples[i])

        if match.opponent not in self.lastest_games:
            self.lastest_games[match.opponent] = []

        self.lastest_games[match.opponent].append(samples)

        reward = batch_to_reward(samples)
        is_won = reward > 3

        for i in range(len(samples)):
            self.match_maker.apply_match_result(match.opponent, is_won[i])

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

        for opponent in self.lastest_games:
            label = opponent.name
            lastest_games_opponent = np.stack(self.lastest_games[opponent]).astype(np.uint8)

            count = len(lastest_games_opponent) / len(lastest_games)
            log[f'game_count/{label}'] = count

            reward = batch_to_reward(lastest_games_opponent)
            is_won = reward > 3

            won_in_series = np.mean(is_won, axis=0)

            lr = LinearRegression()
            lr.fit(np.arange(len(won_in_series)).reshape(-1, 1), won_in_series)

            log[f'win_rate_coefficient/{label}'] = lr.coef_[0]
            log[f'win_rate_intercept/{label}'] = lr.intercept_

        reward = batch_to_reward(lastest_games)

        for i in range(7):
            log[f'game_result/{i}'] = np.mean(reward == i)

        for i in range(len(win_rates)):
            if win_rates[i] == 0:
                continue

            opponent = self.match_maker.agents[i]
            label = opponent.name
            log[f'win_rate/{label}'] = win_rates[i]

        self.lastest_games.clear()

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
        series_length=config.series_length,
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
            agent.apply_match_result(result.match, result.samples)

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
