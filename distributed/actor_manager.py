import multiprocessing
import socket
import pickle
import click
import tempfile

import numpy as np
import jax
import orbax.checkpoint

import distributed.socket_util as socket_util

from network.train import Checkpoint

import actor
import collector
from config import RunConfig


@click.command()
@click.argument('ip', type=str)
@click.argument('port', type=int)
@click.argument("n_clients", type=int)
def main(
    ip: str,
    port: int,
    n_clients: int
):
    jax.config.update('jax_platform_name', 'cpu')

    with jax.default_device(jax.devices("cpu")[0]):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            start_actor_manager(ip, port, n_clients, str(ckpt_dir))


def start_actor_manager(
    ip: str,
    port: int,
    n_clients: int,
    ckpt_dir: str
):
    print(ckpt_dir)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))

    data = socket_util.recv_msg(sock)
    config: RunConfig = pickle.loads(data)

    data = socket_util.recv_msg(sock)
    ckpt: Checkpoint = pickle.loads(data)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, checkpointer)

    ckpt.save(checkpoint_manager)

    ctx = multiprocessing.get_context('spawn')
    match_request_queue = ctx.Queue(100)
    match_result_queue = ctx.Queue(100)
    ckpt_queues = [ctx.Queue(100) for _ in range(n_clients)]

    socket_util.send_msg(sock, pickle.dumps(n_clients))
    data = socket_util.recv_msg(sock)
    matches: list[collector.MatchInfo] = pickle.loads(data)

    for match in matches:
        match_request_queue.put(match)

    for i in range(n_clients):
        seed = np.random.randint(0, 10000)
        args = (match_request_queue,
                match_result_queue,
                ckpt_queues[i],
                ckpt_dir,
                seed,
                config.mcts_params,
                config.series_length,
                config.tokens_length)

        process = ctx.Process(target=actor.start_selfplay_process, args=args)
        process.start()

    while True:
        result: collector.MatchResult = match_result_queue.get()
        result_msg = collector.MessageMatchResult(result, ckpt.step)

        socket_util.send_msg(sock, pickle.dumps(result_msg))

        data = socket_util.recv_msg(sock)
        msg: collector.MessageNextMatch = pickle.loads(data)

        match_request_queue.put(msg.next_match)

        if msg.ckpt is not None:
            ckpt = msg.ckpt

            ckpt.save(checkpoint_manager)

            for ckpt_queue in ckpt_queues:
                ckpt_queue.put(ckpt.step)


if __name__ == '__main__':
    main()
