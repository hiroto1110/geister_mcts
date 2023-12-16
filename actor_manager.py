import multiprocessing
import socket
import pickle
import click

import numpy as np
import orbax.checkpoint

import socket_util
import mcts

import network_transformer as network

import actor
import collector


@click.command()
@click.argument('ip', type=str)
@click.argument('port', type=int)
@click.option(
        "--n_clients", "-n",
        type=int,
        default=15,
)
@click.option(
        "--ckpt_dir", "-d",
        type=str,
        default="/home/kuramitsu/lab/geister/checkpoints/test-3/"
)
def main(
    ip: str,
    port: int,
    n_clients: int,
    ckpt_dir: str,
):
    print(ckpt_dir)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))

    data = socket_util.recv_msg(sock)
    mcts_params: mcts.SearchParameters = pickle.loads(data)
    print(mcts_params)

    data = socket_util.recv_msg(sock)
    updated_msg: collector.MessageUpdatedParameters = pickle.loads(data)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, checkpointer)

    network.save_ckpt(updated_msg.step, updated_msg.ckpt, checkpoint_manager)

    ctx = multiprocessing.get_context('spawn')
    match_request_queue = ctx.Queue(100)
    match_result_queue = ctx.Queue(100)
    ckpt_queues = [ctx.Queue(100) for _ in range(n_clients)]

    for i in range(n_clients * 2):
        match_request_queue.put(0)

    for i in range(n_clients):
        seed = np.random.randint(0, 10000)
        args = (match_request_queue,
                match_result_queue,
                ckpt_queues[i],
                ckpt_dir,
                seed,
                mcts_params)

        process = ctx.Process(target=actor.start_selfplay_process, args=args)
        process.start()

    while True:
        result: collector.MatchResult = match_result_queue.get()
        result_msg = collector.MessageMatchResult(result, updated_msg.step)

        socket_util.send_msg(sock, pickle.dumps(result_msg))

        data = socket_util.recv_msg(sock)
        msg: collector.MessageNextMatch = pickle.loads(data)

        match_request_queue.put(msg.next_match.agent_id)

        if msg.updated_message is not None:
            updated_msg = msg.updated_message

            checkpoint_manager.save(updated_msg.step, updated_msg.ckpt)
            for ckpt_queue in ckpt_queues:
                ckpt_queue.put((updated_msg.step, updated_msg.is_league_member))


if __name__ == '__main__':
    main()
