import os
import multiprocessing


def start_selfplay_process(
        match_request_queue: multiprocessing.Queue,
        match_result_queue: multiprocessing.Queue,
        ckpt_queue: multiprocessing.Queue,
        ckpt_dir: str,
        seed: int,
        mcts_params
):
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import numpy as np
    import jax
    import orbax.checkpoint

    import network_transformer as network
    import mcts
    import fsp
    import collector

    jax.config.update('jax_platform_name', 'cpu')

    np.random.seed(seed)

    with jax.default_device(jax.devices("cpu")[0]):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer)

        ckpt = checkpoint_manager.restore(checkpoint_manager.latest_step())

        model = network.TransformerDecoderWithCache(**ckpt['model'])
        current_params = ckpt['state']['params']

        params_checkpoints = [current_params]

        while True:
            print("start")
            agent_id = match_request_queue.get()
            if agent_id >= len(params_checkpoints):
                agent_id = len(params_checkpoints) - 1
                print("agent_id is out of range")
            print("agent:", agent_id)

            player1 = mcts.PlayerMCTS(current_params, model, mcts_params)

            if agent_id == fsp.SELFPLAY_ID:
                player2 = mcts.PlayerMCTS(current_params, model, mcts_params)
            elif agent_id == 0:
                player2 = mcts.PlayerNaotti2020(depth_min=4, depth_max=6)
            else:
                player2 = mcts.PlayerMCTS(params_checkpoints[agent_id], model, mcts_params)
            
            print("play game")

            if np.random.random() > 0.5:
                actions, color1, color2 = mcts.play_game(player1, player2)
            else:
                actions, color2, color1 = mcts.play_game(player2, player1)
            print("finished game")

            sample1 = player1.create_sample(actions, color2)

            match_result_queue.put(collector.MatchResult(sample1, agent_id))

            if not ckpt_queue.empty():
                step, is_league_member = ckpt_queue.get()
                if is_league_member:
                    params_checkpoints.append(current_params)

                # print(f'update: {step}')
                ckpt = checkpoint_manager.restore(step)
                current_params = ckpt['state']['params']
