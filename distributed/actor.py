import os
import time
import multiprocessing


def start_selfplay_process(
        match_request_queue: multiprocessing.Queue,
        match_result_queue: multiprocessing.Queue,
        ckpt_queue: multiprocessing.Queue,
        ckpt_dir: str,
        seed: int,
        mcts_params_min,
        mcts_params_max,
        series_length: int,
        tokens_length: int,
):
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import numpy as np
    import jax
    import orbax.checkpoint

    from network.train import Checkpoint
    import mcts
    from match_makers import SELFPLAY_ID
    import collector

    jax.config.update('jax_platform_name', 'cpu')

    np.random.seed(seed)

    def load(step: int) -> Checkpoint:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer)

        if step == -1:
            step = checkpoint_manager.latest_step()

        return Checkpoint.load(checkpoint_manager, step, is_caching_model=True)

    with jax.default_device(jax.devices("cpu")[0]):
        ckpt = load(step=-1)

        model = ckpt.model

        params_checkpoints = {-2: None, SELFPLAY_ID: ckpt.params}

        while True:
            start_t = time.perf_counter()

            match: collector.MatchInfo = match_request_queue.get()
            if match.agent_id not in params_checkpoints:
                params_checkpoints[match.agent_id] = load(step=match.agent_id).params
                print(match.agent_id, type(match.agent_id))

            elapsed_t = time.perf_counter() - start_t
            print(f"assigned: (elapsed={elapsed_t:.3f}s, agent={match.agent_id})")

            samples = []

            mcts_params = mcts.SearchParameters.interpolate(mcts_params_min, mcts_params_max, p=np.random.random())
            print(mcts_params)
            player1 = mcts.PlayerMCTS(params_checkpoints[SELFPLAY_ID], model, mcts_params, tokens_length)

            if match.agent_id == SELFPLAY_ID:
                player2 = mcts.PlayerMCTS(params_checkpoints[SELFPLAY_ID], model, mcts_params, tokens_length)
            elif match.agent_id == -2:
                player2 = mcts.PlayerNaotti2020(depth_min=3, depth_max=6)
            else:
                player2 = mcts.PlayerMCTS(params_checkpoints[match.agent_id], model, mcts_params, tokens_length)

            for i in range(series_length):
                if np.random.random() > 0.5:
                    actions, color1, color2 = mcts.play_game(player1, player2)
                else:
                    actions, color2, color1 = mcts.play_game(player2, player1)

                sample = player1.create_sample(actions, color2)
                samples.append(sample)

            match_result_queue.put(collector.MatchResult(samples, match.agent_id))

            if ckpt_queue.empty():
                continue

            while not ckpt_queue.empty():
                step = ckpt_queue.get()

            params_checkpoints[SELFPLAY_ID] = load(step).params
            print(f'update: {step}')
