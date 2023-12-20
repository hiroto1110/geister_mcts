from typing import Union
from dataclasses import dataclass
import pickle
import multiprocessing
import socket
import threading
import queue

from tqdm import tqdm
import numpy as np
import orbax.checkpoint

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
        self.memory = None
        self.is_waiting_result = True

    def is_finished(self) -> bool:
        return len(self.samples) == self.length

    def add_sample(self, sample: Batch):
        self.samples.append(sample)

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

        self.series_list: list[MatchSeries] = []

    def _get_ready_series(self) -> Union[MatchSeries, None]:
        for series in self.series_list:
            if not series.is_waiting_result:
                return series
        return None

    def next_match(self) -> MatchSeries:
        series = self._get_ready_series()

        if series is None:
            agent_id = self.match_maker.next_match()
            series_id = len(self.series_list)

            series = MatchSeries(series_id, agent_id, self.series_length)
            self.series_list.append(series)

        series.is_waiting_result = True

        return series

    def apply_match_result(self, result: MatchResult) -> Union[Batch, None]:
        self.match_maker.apply_match_result(result.agent_id, result.is_won())

        series = self.series_list[result.series_id]

        series.add_sample(result.sample)
        series.is_waiting_result = False

        if not series.is_finished():
            return None

        batch = series.create_batch()

        agent_id = self.match_maker.next_match()
        self.series_list[result.series_id] = MatchSeries(result.series_id, agent_id, self.series_length)

        return batch

    def next_step(self) -> tuple[np.ndarray, bool]:
        win_rates = self.match_maker.get_win_rates()
        is_league_member = np.all(win_rates > self.fsp_threshold)

        if is_league_member:
            self.match_maker.add_agent()

        return win_rates, is_league_member


class TcpClient:
    def __init__(
            self,
            match_maker: match_makers.MatchMaker,
            match_result_queue: queue.Queue,
            mcts_params: mcts.SearchParameters
    ):
        self.match_maker = match_maker
        self.match_result_queue = match_result_queue
        self.mcts_params = mcts_params
        self.ckpt: Checkpoint = None

    def get_current_step(self) -> int:
        return self.ckpt.state.epoch


def handle_client(client: TcpClient, sock: socket.socket):
    socket_util.send_msg(sock, pickle.dumps(client.mcts_params))
    socket_util.send_msg(sock, pickle.dumps(client.ckpt))

    while data := socket_util.recv_msg(sock):
        recv_msg: MessageMatchResult = pickle.loads(data)

        client.match_result_queue.put(recv_msg.result)

        next_match = MatchInfo(client.match_maker.next_match())

        if recv_msg.step == client.get_current_step():
            msg = MessageNextMatch(next_match, ckpt=None)
        else:
            msg = MessageNextMatch(next_match, ckpt=client.ckpt)

        socket_util.send_msg(sock, pickle.dumps(msg))
    sock.close()


def wait_accept(server: socket.socket, client: TcpClient):
    while True:
        client_sock, address = server.accept()
        print(f"New client from {address}")

        thread = threading.Thread(target=handle_client, args=(client, client_sock))
        thread.start()


def start(
        server: socket.socket,
        series_length: int,
        buffer_size: int,
        batch_size: int,
        update_period: int,
        save_period: int,
        match_maker: match_makers.MatchMaker,
        fsp_threshold: float,
        mcts_params: mcts.SearchParameters,
        ckpt_dir: str,
        minibatch_temp_path: str,
        learner_update_queue: multiprocessing.Queue,
        learner_request_queue: multiprocessing.Queue,
):
    buffer = ReplayBuffer(
        buffer_size,
        sample_shape=(series_length,),
        seq_length=200
    )
    # buffer.load('./data/replay_buffer/189.npz')

    match_series_manager = MatchSeriesManager(match_maker, series_length, fsp_threshold)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, checkpointer)

    match_result_queue = queue.Queue()

    client = TcpClient(match_maker, match_result_queue, mcts_params)
    client.ckpt = Checkpoint.load(checkpoint_manager, checkpoint_manager.latest_step())

    thread = threading.Thread(target=wait_accept, args=(server, client))
    thread.start()

    is_waiting_parameter_update = False

    for s in range(1000000):
        # recv_match_result
        for i in tqdm(range(update_period), desc='Collecting'):
            series_batch = None

            while series_batch is None:
                result: MatchResult = match_result_queue.get()
                series_batch = match_series_manager.apply_match_result(result.sample, result.series_id)

            buffer.add_sample(series_batch)

        if s % save_period == 0:
            buffer.save(append=True)

        if is_waiting_parameter_update:
            # recv_updated_params
            step: int = learner_update_queue.get()
            client.ckpt = Checkpoint.load(checkpoint_manager, step)

        is_league_member, win_rate = match_series_manager.next_step()
        print(win_rate)

        if len(buffer) > 128:
            log_dict = {}
            for i in range(len(win_rate)):
                if win_rate[i] > 0:
                    log_dict[f'fsp/win_rate_{i}'] = win_rate[i]

            # send_training_minibatch
            batch = buffer.get_minibatch(batch_size)
            batch.to_npz(minibatch_temp_path)
            learner_request_queue.put((log_dict, is_league_member))

            is_waiting_parameter_update = True


def main(
    host: str,
    port: int,
    buffer_size: int,
    batch_size: int,
    update_period: int,
    match_maker: match_makers.MatchMaker,
    fsp_threshold: float,
    mcts_params: mcts.SearchParameters,
    ckpt_dir: str,
    minibatch_temp_path: str,
    learner_update_queue: multiprocessing.Queue,
    learner_request_queue: multiprocessing.Queue,
):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        server.bind((host, port))
        server.listen()

        start(
            server,
            buffer_size,
            batch_size,
            update_period,
            match_maker,
            fsp_threshold,
            mcts_params,
            ckpt_dir,
            minibatch_temp_path,
            learner_update_queue,
            learner_request_queue
        )

    except Exception as e:
        server.close()
        raise e
