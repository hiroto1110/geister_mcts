import multiprocessing
import socket
import click
import tempfile

import numpy as np
import jax

from distributed.communication import EncryptedCommunicator

from network.checkpoints import CheckpointManager

import actor
from messages import (
    MessageActorInitClient, MessageActorInitServer,
    MessageMatchResult, MessageNextMatch
)


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

    communicator.send_json_obj(sock, MessageActorInitClient(n_clients))

    init_msg = communicator.recv_json_obj(sock, MessageActorInitServer)

    config = init_msg.config
    print(config)

    checkpoint_managers = {
        name: CheckpointManager(f'{ckpt_dir}/{name}')
        for name in init_msg.snapshots
    }

    for name, snapshots in init_msg.snapshots.items():
        for ckpt_i in snapshots:
            checkpoint_managers[name].save(ckpt_i)

    ctx = multiprocessing.get_context('spawn')
    match_request_queue = ctx.Queue(100)
    match_result_queue = ctx.Queue(100)

    for match in init_msg.matches:
        match_request_queue.put(match)

    mcts_params_ranges = {agent.name: agent.mcts_params for agent in config.agents}

    for i in range(n_clients):
        seed = np.random.randint(0, 10000)
        args = (
            match_request_queue,
            match_result_queue,
            ckpt_dir,
            mcts_params_ranges,
            config.series_length,
            config.tokens_length,
            seed,
        )

        process = ctx.Process(target=actor.start_selfplay_process, args=args)
        process.start()

    while True:
        result: MessageMatchResult = match_result_queue.get()
        communicator.send_json_obj(sock, result)

        msg = communicator.recv_json_obj(sock, MessageNextMatch)

        for name, ckpt_list in msg.ckpts.items():
            for i in range(len(ckpt_list)):
                checkpoint_managers[name].save(ckpt_list[i])

        match_request_queue.put(msg.match)


if __name__ == '__main__':
    main()
