import pickle
import multiprocessing
import socket
import asyncio
from dataclasses import dataclass

from tqdm import tqdm
import orbax.checkpoint

import fsp
from buffer import ReplayBuffer, Sample


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
class MessageUpdatedParameters:
    ckpt: dict
    is_league_member: bool
    step: int


@dataclass
class MessageNextMatch:
    next_match: MatchInfo
    updated_message: MessageUpdatedParameters


class TcpClient:
    def __init__(self, match_maker: fsp.FSP, match_result_queue: asyncio.Queue):
        self.match_maker = match_maker
        self.match_result_queue = match_result_queue
        self.updated_msg = MessageUpdatedParameters(None, False, 0)

    async def handle_client(self, client, loop):
        while data := await loop.sock_recv(client, 1024):
            recv_msg: MessageMatchResult = pickle.loads(data)

            print(recv_msg)

            self.match_result_queue.put(recv_msg.result)

            next_match = MatchInfo(self.match_maker.next_match())

            if recv_msg.step == self.updated_msg.step:
                msg = MessageNextMatch(next_match, updated_message=None)
            else:
                msg = MessageNextMatch(next_match, updated_message=self.updated_msg)

            await loop.sock_sendall(client, pickle.dumps(msg))
        client.close()


async def start(
        host: str,
        port: int,
        buffer_size: int,
        batch_size: int,
        update_period: int,
        match_maker: fsp.FSP,
        fsp_threshold: float,
        ckpt_dir: str,
        minibatch_temp_path: str,
        learner_update_queue: multiprocessing.Queue,
        learner_request_queue: multiprocessing.Queue,
):
    buffer = ReplayBuffer(buffer_size, seq_length=200)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, checkpointer)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen()
    loop = asyncio.get_event_loop()

    clients: list[TcpClient] = []

    match_result_queue = asyncio.Queue()

    current_updated_msg = MessageUpdatedParameters(None, False, 0)

    while True:
        # recv_match_result
        for i in tqdm(range(update_period), desc='Collecting'):
            while match_result_queue.empty():
                try:
                    client_sock, address = await asyncio.wait_for(loop.sock_accept(server), 0.1)
                    client_sock.setblocking(False)
                    print(f"New client from {address}")

                    client = TcpClient(match_maker, match_result_queue)
                    asyncio.create_task(client.handle_client(client_sock, loop))

                    clients.append(client)
                except asyncio.TimeoutError:
                    pass

            result = match_result_queue.get()
            buffer.add_sample(result.sample)

            match_maker.apply_match_result(result.agent_id, result.is_won())

        is_league_member, win_rate = match_maker.is_winning_all_agents(fsp_threshold)
        if is_league_member:
            match_maker.add_agent()

        log_dict = {}

        for i in range(len(win_rate)):
            if win_rate[i] > 0:
                log_dict[f'fsp/win_rate_{i}'] = win_rate[i]

        # recv_updated_params
        step = learner_update_queue.get()
        ckpt = checkpoint_manager.restore(step)

        current_updated_msg = MessageUpdatedParameters(ckpt, is_league_member, step)
        for client in clients:
            client.updated_msg = current_updated_msg

        # send_training_minibatch
        batch = buffer.get_minibatch(batch_size)
        batch.to_npz(minibatch_temp_path)
        learner_request_queue.put(log_dict)


def main(
    host: str,
    port: int,
    buffer_size: int,
    batch_size: int,
    update_period: int,
    match_maker: fsp.FSP,
    fsp_threshold: float,
    ckpt_dir: str,
    minibatch_temp_path: str,
    learner_update_queue: multiprocessing.Queue,
    learner_request_queue: multiprocessing.Queue,
):
    asyncio.run(start(
        host,
        port,
        buffer_size,
        batch_size,
        update_period,
        match_maker,
        fsp_threshold,
        ckpt_dir,
        minibatch_temp_path,
        learner_update_queue,
        learner_request_queue
    ))
