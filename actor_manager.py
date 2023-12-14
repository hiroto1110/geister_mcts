import multiprocessing
import socket
import pickle

import numpy as np
import orbax.checkpoint

import actor
import collector


def main(
        ip: str,
        port: int,
        n_clients: int,
        ckpt_dir: str,
        num_mcts_sim: int,
        dirichlet_alpha: float
):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, checkpointer)

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
                num_mcts_sim,
                dirichlet_alpha)

        process = ctx.Process(target=actor.start_selfplay_process, args=args)
        process.start()

    prev_step = 0

    while True:
        result: collector.MatchResult = match_result_queue.get()
        result_msg = collector.MessageMatchResult(result, prev_step)

        data = pickle.dumps(result_msg)
        sock.send(data)

        data = sock.recv()
        msg: collector.MessageNextMatch = pickle.loads(data)

        match_request_queue.put(msg.next_match.agent_id)

        updated_msg = msg.updated_message

        if updated_msg is not None:
            checkpoint_manager.save(updated_msg.step, updated_msg.ckpt)
            ckpt_queues.put((updated_msg.step, updated_msg.is_league_member))


if __name__ == '__main__':
    main(
        ip='localhost',
        port=23001,
        n_clients=15,
        ckpt_dir='./checkpoints/run-3',
        num_mcts_sim=50,
        dirichlet_alpha=0.3
    )
