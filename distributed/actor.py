import os
import time
import multiprocessing


def start_selfplay_process(
        match_request_queue: multiprocessing.Queue,
        match_result_queue: multiprocessing.Queue,
        ckpt_queue: multiprocessing.Queue,
        ckpt_dir: str,
        seed: int,
        mcts_params,
        series_length: int
):
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import numpy as np
    import jax
    import orbax.checkpoint

    from network.train import Checkpoint
    import mcts
    import match_makers
    import collector

    jax.config.update('jax_platform_name', 'cpu')

    np.random.seed(seed)

    with jax.default_device(jax.devices("cpu")[0]):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer)

        step = checkpoint_manager.latest_step()
        ckpt = Checkpoint.load(checkpoint_manager, step, is_caching_model=True)

        params_checkpoints = {-1: None, 0: None}

        while True:
            start_t = time.perf_counter()

            match: collector.MatchInfo = match_request_queue.get()
            if match.agent_id not in params_checkpoints:
                ckpt = Checkpoint.load(checkpoint_manager, step, is_caching_model=True)
                params_checkpoints[match.agent_id] = ckpt.params

            elapsed_t = time.perf_counter() - start_t
            print(f"assigned: (elapsed={elapsed_t:.3f}s, agent={match.agent_id})")

            samples = []

            for i in range(series_length):
                player1 = mcts.PlayerMCTS(ckpt.params, ckpt.model, mcts_params)

                if match.agent_id == match_makers.SELFPLAY_ID:
                    player2 = mcts.PlayerMCTS(ckpt.params, ckpt.model, mcts_params)
                elif match.agent_id == 0:
                    player2 = mcts.PlayerNaotti2020(depth_min=4, depth_max=6)
                else:
                    player2 = mcts.PlayerMCTS(params_checkpoints[match.agent_id], ckpt.model, mcts_params)

                if np.random.random() > 0.5:
                    actions, color1, color2 = mcts.play_game(player1, player2)
                else:
                    actions, color2, color1 = mcts.play_game(player2, player1)

                sample = player1.create_sample(actions, color2)
                samples.append(sample)

            match_result_queue.put(collector.MatchResult(samples, match.agent_id))

            if not ckpt_queue.empty():
                step = ckpt_queue.get()
                ckpt = Checkpoint.load(checkpoint_manager, step, is_caching_model=True)
                print(f'update: {step}')
