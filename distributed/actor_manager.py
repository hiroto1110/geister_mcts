import multiprocessing
import socket
import pickle
import click
import tempfile

from tqdm import tqdm

import numpy as np
import jax
import orbax.checkpoint

from socket_util import EncryptedCommunicator

from network.train import Checkpoint

import actor
import collector
from collector import MatchInfo, MessageNextMatch
from config import RunConfig


@click.command()
@click.argument('ip', type=str)
@click.argument('port', type=int)
@click.argument("n_clients", type=int)
@click.argument("password", type=str)
def main(
    ip: str,
    port: int,
    n_clients: int,
    password: str
):
    jax.config.update('jax_platform_name', 'cpu')

    with jax.default_device(jax.devices("cpu")[0]):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            start_actor_manager(ip, port, n_clients, str(ckpt_dir), password)


def start_actor_manager(
    ip: str,
    port: int,
    n_clients: int,
    ckpt_dir: str,
    password: str
):
    print(ckpt_dir)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))

    communicator = EncryptedCommunicator(password)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, checkpointer)

    config: RunConfig = pickle.loads(communicator.recv_bytes(sock))
    print(config)

    ckpt: Checkpoint = pickle.loads(communicator.recv_bytes(sock))
    ckpt.save(checkpoint_manager)

    n_snapshots: int = pickle.loads(communicator.recv_bytes(sock))
    snapshots: list[Checkpoint] = []

    for i in tqdm(range(n_snapshots), desc="Receiving Snapshots"):
        ckpt_i: Checkpoint = pickle.loads(communicator.recv_bytes(sock))
        ckpt_i.save(checkpoint_manager)
        snapshots.append(ckpt_i)

    ctx = multiprocessing.get_context('spawn')
    match_request_queue = ctx.Queue(100)
    match_result_queue = ctx.Queue(100)
    ckpt_queues = [ctx.Queue(100) for _ in range(n_clients)]

    communicator.send_bytes(sock, pickle.dumps(n_clients))
    matches: list[MatchInfo] = pickle.loads(communicator.recv_bytes(sock))

    for match in matches:
        match_request_queue.put(match)

    for i in range(n_clients):
        seed = np.random.randint(0, 10000)
        args = (match_request_queue,
                match_result_queue,
                ckpt_queues[i],
                ckpt_dir,
                seed,
                config.mcts_params_min,
                config.mcts_params_max,
                config.series_length,
                config.tokens_length)

        process = ctx.Process(target=actor.start_selfplay_process, args=args)
        process.start()

    while True:
        result: collector.MatchResult = match_result_queue.get()
        result_msg = collector.MessageMatchResult(result, ckpt.step)

        communicator.send_bytes(sock, pickle.dumps(result_msg))

        msg: MessageNextMatch = pickle.loads(communicator.recv_bytes(sock))

        match_request_queue.put(msg.next_match)

        if msg.ckpt is not None:
            ckpt = msg.ckpt

            ckpt.save(checkpoint_manager)

            for ckpt_queue in ckpt_queues:
                ckpt_queue.put(ckpt.step)


if __name__ == '__main__':
    main()
