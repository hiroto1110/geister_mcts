from typing import Union
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


@dataclass
class MatchResult:
    sample: Batch
    series_id: int
    agent_id: int

    def is_won(self):
        return self.sample.reward > 3


@dataclass
class MatchInfo:
    series_id: int
    agent_id: int


@dataclass
class MessageMatchResult:
    result: MatchResult
    step: int


@dataclass
class MessageNextMatch:
    next_match: MatchInfo
    ckpt: Checkpoint


class MatchSeries:
    def __init__(self, series_id: int, agent_id: int, length: int) -> None:
        self.series_id = series_id
        self.agent_id = agent_id
        self.length = length

        self.samples: list[Batch] = []
        self.assigned_client_id = -1

    def is_ready(self):
        return self.assigned_client_id == -1

    def assign_client(self, client_id: int):
        # print(f"assign {self.series_id}: {client_id}")
        self.assigned_client_id = client_id

    def apply_result(self, sample: Batch):
        self.samples.append(sample)
        # print(f"apply_result {self.series_id} (n={len(self.samples)}): {self.assigned_client_id}")
        self.assigned_client_id = -1

    def is_finished(self) -> bool:
        return len(self.samples) == self.length

    def create_batch(self) -> Batch:
        assert self.is_finished()

        return Batch.stack(self.samples)


class MatchSeriesManager:
    def __init__(
            self,
            match_maker: match_makers.MatchMaker,
            series_length: int,
            fsp_threshold: float) -> None:

        self.match_maker = match_maker
        self.series_length = series_length
        self.fsp_threshold = fsp_threshold

        self.agents: list[int] = [0]
        self.series_list: list[MatchSeries] = []

    def _get_ready_series(self) -> Union[MatchSeries, None]:
        for series in self.series_list:
            if series.is_ready():
                return series
        return None

    def _get_next_match_agent_id(self):
        agent_index = self.match_maker.next_match()
        if agent_index >= 0:
            return self.agents[agent_index]
        else:
            return -1

    def next_match(self, client_id: int) -> MatchSeries:
        series = self._get_ready_series()

        if series is None:
            agent_id = self._get_next_match_agent_id()
            series_id = len(self.series_list)

            series = MatchSeries(series_id, agent_id, self.series_length)
            self.series_list.append(series)

        series.assign_client(client_id)

        return series

    def apply_match_result(self, result: MatchResult) -> Union[Batch, None]:
        self.match_maker.apply_match_result(result.agent_id, result.is_won())

        series = self.series_list[result.series_id]
        series.apply_result(result.sample)

        if not series.is_finished():
            return None

        batch = series.create_batch()

        agent_id = self.match_maker.next_match()
        self.series_list[result.series_id] = MatchSeries(result.series_id, agent_id, self.series_length)

        return batch

    def client_disconnected(self, client_id: int):
        print(f"dissconnected: {client_id}")

        for series in self.series_list:
            if series.assigned_client_id == client_id:
                series.assigned_client_id = -1

    def next_step(self, step: int) -> tuple[np.ndarray, bool]:
        win_rates = self.match_maker.get_win_rates()

        if np.all(win_rates > self.fsp_threshold):
            self.match_maker.add_agent()
            self.agents.append(step)

        return win_rates


class TcpClient:
    def __init__(
            self,
            match_series_manager: MatchSeriesManager,
            match_result_queue: queue.Queue,
            mcts_params: mcts.SearchParameters
    ):
        self.match_series_manager = match_series_manager
        self.match_result_queue = match_result_queue
        self.mcts_params = mcts_params
        self.threading_lock = threading.Lock()
        self.ckpt: Checkpoint = None

        self.n_clients = 0

    def add_client(self):
        self.n_clients += 1
        return self.n_clients

    def get_current_step(self) -> int:
        return self.ckpt.step


def handle_client(client: TcpClient, sock: socket.socket):
    socket_util.send_msg(sock, pickle.dumps(client.mcts_params))
    socket_util.send_msg(sock, pickle.dumps(client.ckpt))

    n_processes: int = pickle.loads(socket_util.recv_msg(sock))

    client.threading_lock.acquire()
    client_id = client.add_client()
    init_matches = [client.match_series_manager.next_match(client_id) for i in range(n_processes + 4)]
    client.threading_lock.release()

    socket_util.send_msg(sock, pickle.dumps(init_matches))

    while data := socket_util.recv_msg(sock):
        recv_msg: MessageMatchResult = pickle.loads(data)

        client.match_result_queue.put(recv_msg.result)

        client.threading_lock.acquire()
        series = client.match_series_manager.next_match(client_id)
        client.threading_lock.release()

        next_match = MatchInfo(series.series_id, series.agent_id)

        if recv_msg.step == client.get_current_step():
            msg = MessageNextMatch(next_match, ckpt=None)
        else:
            msg = MessageNextMatch(next_match, ckpt=client.ckpt)

        socket_util.send_msg(sock, pickle.dumps(msg))
    sock.close()

    client.threading_lock.acquire()
    series = client.match_series_manager.client_disconnected(client_id)
    client.threading_lock.release()


def wait_accept(server: socket.socket, client: TcpClient):
    while True:
        client_sock, address = server.accept()
        print(f"New client from {address}")

        thread = threading.Thread(target=handle_client, args=(client, client_sock))
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
        seq_length=200
    )
    buffer.load('./data/replay_buffer/run-3.npz')

    match_maker = config.match_maker.create_match_maker()
    match_series_manager = MatchSeriesManager(match_maker, config.series_length, config.fsp_threshold)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(config.ckpt_dir, checkpointer)

    match_result_queue = queue.Queue()

    client = TcpClient(match_series_manager, match_result_queue, config.mcts_params)
    client.ckpt = Checkpoint.load(checkpoint_manager, checkpoint_manager.latest_step())

    thread = threading.Thread(target=wait_accept, args=(server, client))
    thread.start()

    is_waiting_parameter_update = False

    for s in range(1000000):
        # recv_match_result
        for i in tqdm(range(config.update_period), desc='Collecting'):
            series_batch = None

            while series_batch is None:
                result: MatchResult = match_result_queue.get()
                series_batch = match_series_manager.apply_match_result(result)

            buffer.add_sample(series_batch)

        buffer.save(
            file_name='./data/replay_buffer/run-3.npz',
            append=True,
            indices=buffer.get_last_indices(config.update_period)
        )

        if is_waiting_parameter_update:
            # recv_updated_params
            step: int = learner_update_queue.get()
            client.ckpt = Checkpoint.load(checkpoint_manager, step)

        win_rate = match_series_manager.next_step(client.ckpt.step)
        print(win_rate)

        if len(buffer) >= config.batch_size:
            log_dict = {}
            for i in range(len(win_rate)):
                if win_rate[i] > 0:
                    log_dict[f'fsp/win_rate_{i}'] = win_rate[i]

            # send_training_minibatch
            batch = buffer.get_minibatch(config.batch_size * config.num_batches)
            batch.to_npz(config.minibatch_temp_path)
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
