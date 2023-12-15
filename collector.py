from dataclasses import dataclass
import pickle
import multiprocessing
import socket
import asyncio
import nest_asyncio

from tqdm import tqdm
import orbax.checkpoint

import socket_util

from buffer import ReplayBuffer, Sample
import mcts
import fsp

nest_asyncio.apply()


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
    def __init__(self, match_maker: fsp.FSP, match_result_queue: asyncio.Queue, mcts_params: mcts.SearchParameters):
        self.match_maker = match_maker
        self.match_result_queue = match_result_queue
        self.mcts_params = mcts_params
        self.updated_msg = MessageUpdatedParameters(None, False, 0)

    async def handle_client(self, client: socket.socket, loop: asyncio.AbstractEventLoop):
        print("handle_client")
        await socket_util.send_msg_async(loop, client, pickle.dumps(self.mcts_params))
        await socket_util.send_msg_async(loop, client, pickle.dumps(self.updated_msg))

        while data := await socket_util.recv_msg_async(loop, client):
            recv_msg: MessageMatchResult = pickle.loads(data)

            self.match_result_queue.put(recv_msg.result)

            next_match = MatchInfo(self.match_maker.next_match())

            if recv_msg.step == self.updated_msg.step:
                msg = MessageNextMatch(next_match, updated_message=None)
            else:
                msg = MessageNextMatch(next_match, updated_message=self.updated_msg)

            await socket_util.send_msg_async(loop, client, pickle.dumps(msg))
        client.close()


async def wait_accept(server: socket.socket, client: TcpClient, loop: asyncio.AbstractEventLoop):
    while True:
        client_sock, address = await loop.sock_accept(server)

        client_sock.setblocking(False)
        print(f"New client from {address}")

        loop.create_task(client.handle_client(client_sock, loop))
        await asyncio.sleep(2)


async def start(
        server: socket.socket,
        buffer_size: int,
        batch_size: int,
        update_period: int,
        match_maker: fsp.FSP,
        fsp_threshold: float,
        mcts_params: mcts.SearchParameters,
        ckpt_dir: str,
        minibatch_temp_path: str,
        learner_update_queue: multiprocessing.Queue,
        learner_request_queue: multiprocessing.Queue,
):
    buffer = ReplayBuffer(buffer_size, seq_length=200)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, checkpointer)

    ckpt = checkpoint_manager.restore(checkpoint_manager.latest_step())

    loop = asyncio.get_event_loop()

    match_result_queue = asyncio.Queue()
    client = TcpClient(match_maker, match_result_queue, mcts_params)
    client.updated_msg = MessageUpdatedParameters(ckpt, False, ckpt['state']['epoch'])

    loop.create_task(wait_accept(server, client, loop))

    while True:
        # recv_match_result
        for i in tqdm(range(update_period), desc='Collecting'):
            while match_result_queue.empty():
                await asyncio.sleep(0.1)

            result: MatchResult = match_result_queue.get()
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

        client.updated_msg = MessageUpdatedParameters(ckpt, is_league_member, step)

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

        asyncio.run(start(
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
        ))

    except Exception as e:
        server.close()
        raise e
