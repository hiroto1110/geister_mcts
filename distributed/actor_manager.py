import multiprocessing
import socket
import click
import tempfile

import numpy as np
import jax

from distributed.communication import EncryptedCommunicator

from network.checkpoints import CheckpointManager

import actor
import collector


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

    checkpoint_manager = CheckpointManager(ckpt_dir)

    communicator.send_json_obj(sock, collector.MessageActorInitClient(n_clients))

    init_msg = communicator.recv_json_obj(sock, collector.MessageActorInitServer)

    config = init_msg.config
    print(config)

    ckpt = init_msg.current_ckpt
    checkpoint_manager.save(ckpt)

    for ckpt_i in init_msg.snapshots:
        checkpoint_manager.save(ckpt_i)

    ctx = multiprocessing.get_context('spawn')
    match_request_queue = ctx.Queue(100)
    match_result_queue = ctx.Queue(100)
    ckpt_queues = [ctx.Queue(100) for _ in range(n_clients)]

    for match in init_msg.matches:
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
        communicator.send_json_obj(sock, collector.MessageMatchResult(result, ckpt.step))

        msg = communicator.recv_json_obj(sock, collector.MessageNextMatch)

        match_request_queue.put(msg.next_match)

        if msg.ckpt is not None:
            ckpt = msg.ckpt

            checkpoint_manager.save(ckpt)

            for ckpt_queue in ckpt_queues:
                ckpt_queue.put(ckpt.step)


if __name__ == '__main__':
    main()
