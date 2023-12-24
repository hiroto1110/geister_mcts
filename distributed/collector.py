from dataclasses import dataclass
import pickle
import multiprocessing
import socket
import threading
import queue

from tqdm import tqdm
import numpy as np
import jax
import orbax.checkpoint

from distributed.config import RunConfig
import distributed.socket_util as socket_util

from buffer import ReplayBuffer, Batch
from network.train import Checkpoint
import mcts
import match_makers
import training_logger


@dataclass
class MatchResult:
    samples: list[Batch]
    agent_id: int


@dataclass
class MatchInfo:
    agent_id: int


@dataclass
class MessageMatchResult:
    result: MatchResult
    step: int


@dataclass
class MessageNextMatch:
    next_match: MatchInfo
    ckpt: Checkpoint


class AgentManager:
    def __init__(
            self,
            match_maker: match_makers.MatchMaker,
            series_length: int,
            fsp_threshold: float
    ) -> None:

        self.match_maker = match_maker
        self.series_length = series_length
        self.fsp_threshold = fsp_threshold

        self.agents: list[int] = [0]

    def next_match(self):
        agent_index = self.match_maker.next_match()
        if agent_index >= 0:
            return self.agents[agent_index]
        else:
            return agent_index

    def apply_match_result(self, result: MatchResult):
        for sample in result.samples:
            self.match_maker.apply_match_result(result.agent_id, sample.is_won())

    def next_step(self, step: int) -> np.ndarray:
        win_rates = self.match_maker.get_win_rates()

        if np.all(win_rates > self.fsp_threshold):
            self.match_maker.add_agent()
            self.agents.append(step)

        return win_rates


class CheckpointHolder:
    def __init__(self):
        self.ckpt: Checkpoint = None

    def get_current_step(self) -> int:
        return self.ckpt.step


def handle_client(
        sock: socket.socket,
        match_result_queue: queue.Queue,
        ckpt_holder: CheckpointHolder,
        agent_manager: AgentManager,
        mcts_params: mcts.SearchParameters
        ):
    socket_util.send_msg(sock, pickle.dumps(mcts_params))
    socket_util.send_msg(sock, pickle.dumps(ckpt_holder.ckpt))
    socket_util.send_msg(sock, pickle.dumps(agent_manager.series_length))

    n_processes: int = pickle.loads(socket_util.recv_msg(sock))

    init_matches = [MatchInfo(agent_manager.next_match()) for i in range(n_processes + 4)]

    socket_util.send_msg(sock, pickle.dumps(init_matches))

    while data := socket_util.recv_msg(sock):
        recv_msg: MessageMatchResult = pickle.loads(data)

        match_result_queue.put(recv_msg.result)

        next_match = MatchInfo(agent_manager.next_match())

        if recv_msg.step == ckpt_holder.get_current_step():
            msg = MessageNextMatch(next_match, ckpt=None)
        else:
            msg = MessageNextMatch(next_match, ckpt=ckpt_holder.ckpt)

        socket_util.send_msg(sock, pickle.dumps(msg))
    sock.close()


def wait_accept(
        server: socket.socket,
        match_result_queue: queue.Queue,
        ckpt_holder: CheckpointHolder,
        agent_manager: AgentManager,
        mcts_params: mcts.SearchParameters):
    while True:
        client_sock, address = server.accept()
        print(f"New client from {address}")

        args = client_sock, match_result_queue, ckpt_holder, agent_manager, mcts_params
        thread = threading.Thread(target=handle_client, args=args)
        thread.start()


def start(
        server: socket.socket,
        config: RunConfig,
        learner_update_queue: multiprocessing.Queue,
        learner_request_queue: multiprocessing.Queue,
):
    buffer = ReplayBuffer(
        buffer_size=config.buffer_size,
        sample_shape=(config.series_length,),
        seq_length=220
    )
    buffer.load('./data/replay_buffer/run-3.npz')

    match_maker = config.match_maker.create_match_maker()
    agent_manager = AgentManager(match_maker, config.series_length, config.fsp_threshold)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(config.ckpt_dir, checkpointer)

    match_result_queue = queue.Queue()

    ckpt_holder = CheckpointHolder()
    ckpt_holder.ckpt = Checkpoint.load(checkpoint_manager, checkpoint_manager.latest_step())

    args = server, match_result_queue, ckpt_holder, agent_manager, config.mcts_params
    thread = threading.Thread(target=wait_accept, args=args)
    thread.start()

    is_waiting_parameter_update = False

    for s in range(1000000):
        # recv_match_result
        for i in tqdm(range(config.update_period), desc='Collecting'):
            result: MatchResult = match_result_queue.get()
            agent_manager.apply_match_result(result)
            buffer.add_sample(Batch.stack(result.samples))

        last_batch = buffer.get_last_minibatch(config.update_period)
        last_batch.save(
            file_name='./data/replay_buffer/run-3.npz',
            append=True
        )

        win_rate = agent_manager.next_step(ckpt_holder.ckpt.step)
        print(win_rate)

        if is_waiting_parameter_update:
            # recv_updated_params
            step: int = learner_update_queue.get()
            ckpt_holder.ckpt = Checkpoint.load(checkpoint_manager, step)

        if len(buffer) >= config.batch_size:
            log_dict = training_logger.create_log(
                win_rates=win_rate,
                last_games=last_batch
            )

            # send_training_minibatch
            train_batch = buffer.get_minibatch(config.batch_size * config.num_batches)
            train_batch.to_npz(config.minibatch_temp_path)
            learner_request_queue.put(log_dict)

            is_waiting_parameter_update = True


def main(
        host: str,
        port: int,
        config: RunConfig,
        learner_update_queue: multiprocessing.Queue,
        learner_request_queue: multiprocessing.Queue,
):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        server.bind((host, port))
        server.listen()

        jax.config.update('jax_platform_name', 'cpu')
        with jax.default_device(jax.devices("cpu")[0]):
            start(server, config, learner_update_queue, learner_request_queue)

    except Exception as e:
        server.close()
        raise e
