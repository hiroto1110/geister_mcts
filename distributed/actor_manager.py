import multiprocessing
import socket
import click
import tempfile

import numpy as np
import jax

import distributed.actor as actor
from distributed.communication import EncryptedCommunicator
from distributed.messages import (
    MessageActorInitClient, MessageActorInitServer,
    MessageMatchResult, MessageNextMatch
)

from network.checkpoints import CheckpointManager


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

    communicator.send_json_obj(sock, MessageActorInitClient(n_clients))

    init_msg = communicator.recv_json_obj(sock, MessageActorInitServer)

    checkpoint_manager = CheckpointManager(ckpt_dir)

    for ckpt_i in init_msg.snapshots:
        checkpoint_manager.save(ckpt_i)

    ctx = multiprocessing.get_context('spawn')
    match_request_queue = ctx.Queue(100)
    match_result_queue = ctx.Queue(100)

    for match in init_msg.matches:
        match_request_queue.put(match)

    for i in range(n_clients):
        seed = np.random.randint(0, 10000)
        args = (
            match_request_queue,
            match_result_queue,
            ckpt_dir,
            init_msg.tokens_length,
            seed,
        )

        process = ctx.Process(target=actor.start_selfplay_process, args=args)
        process.start()

    while True:
        result: MessageMatchResult = match_result_queue.get()
        communicator.send_json_obj(sock, result)

        msg = communicator.recv_json_obj(sock, MessageNextMatch)

        for ckpt in msg.ckpts:
            checkpoint_manager.save(ckpt)

        match_request_queue.put(msg.match)


@click.command()
@click.argument('ip', type=str)
@click.argument('port', type=int)
@click.argument("password", type=str)
@click.argument("n_clients", type=int)
def main(
    ip: str,
    port: int,
    password: str,
    n_clients: int,
):
    jax.config.update('jax_platform_name', 'cpu')

    with tempfile.TemporaryDirectory() as ckpt_dir:
        start_actor_manager(ip, port, n_clients, str(ckpt_dir), password)


if __name__ == '__main__':
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'

    main()
