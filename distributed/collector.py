from dataclasses import dataclass
import pickle
import multiprocessing
import socket
import threading
import queue

from tqdm import tqdm
import orbax.checkpoint

import distributed.socket_util as socket_util

from buffer import ReplayBuffer, Sample
from network.train import Checkpoint
import mcts
import match_makers


@dataclass
class MatchResult:
    sample: Sample
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
    buffer = ReplayBuffer(buffer_size, seq_length=200)
    buffer.load('./data/replay_buffer/189.npz')

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, checkpointer)

    match_result_queue = queue.Queue()

    client = TcpClient(match_maker, match_result_queue, mcts_params)
    client.ckpt = Checkpoint.load(checkpoint_manager, checkpoint_manager.latest_step())

    thread = threading.Thread(target=wait_accept, args=(server, client))
    thread.start()

    while True:
        is_league_member, win_rate = match_maker.is_winning_all_agents(fsp_threshold)
        if is_league_member:
            match_maker.add_agent()

        print(win_rate)

        log_dict = {}

        for i in range(len(win_rate)):
            if win_rate[i] > 0:
                log_dict[f'fsp/win_rate_{i}'] = win_rate[i]

        # send_training_minibatch
        batch = buffer.get_minibatch(batch_size)
        batch.to_npz(minibatch_temp_path)
        learner_request_queue.put(log_dict)

        # recv_match_result
        for i in tqdm(range(update_period), desc='Collecting'):
            result: MatchResult = match_result_queue.get()
            buffer.add_sample(result.sample)

            match_maker.apply_match_result(result.agent_id, result.is_won())

        # recv_updated_params
        step: int = learner_update_queue.get()
        client.ckpt = Checkpoint.load(checkpoint_manager, step)


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
