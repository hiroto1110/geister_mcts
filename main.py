import multiprocessing as mp

import optax
from flax.training import checkpoints
import jax
import numpy as np

from tqdm import tqdm
import wandb

from buffer import ReplayBuffer, Sample
import network_transformer as network
import geister as game
import mcts


def start_selfplay_process(queue: mp.Queue, n_updates,
                           num_mcts_simulations: int, dirichlet_alpha):
    with jax.default_device(jax.devices("cpu")[0]):
        # print(f"device: {jax.devices()}")

        model = create_model()

        ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, prefix=PREFIX, target=None)
        pred_state = mcts.PredictState(model.apply, ckpt['params'])

        last_n_updates = n_updates.value

        while True:
            # start = time.perf_counter()
            sample = selfplay(pred_state, model, num_mcts_simulations, dirichlet_alpha)
            # end = time.perf_counter()
            # print(f"game is done: {end - start}")

            queue.put(sample)

            if last_n_updates != n_updates.value:
                ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, prefix=PREFIX, target=None)
                pred_state = mcts.PredictState(model.apply, ckpt['params'])

                last_n_updates = n_updates.value


def selfplay(pred_state: mcts.PredictState,
             model: network.TransformerDecoderWithCache,
             num_mcts_simulations: int, dirichlet_alpha):

    state = game.get_initial_state()

    player = 1

    node1 = mcts.create_root_node(state, pred_state, model, 1)
    node2 = mcts.create_root_node(state, pred_state, model, -1)

    policies = np.zeros((201, 144))

    for i in range(200):
        policy, node1, node2 = mcts.step(node1, node2,
                                         state, player,
                                         pred_state,
                                         num_mcts_simulations,
                                         dirichlet_alpha)

        policies[i] = policy

        if game.is_done(state, player):
            break

        player = -player

    record_player = np.random.choice([1, -1])

    reward = game.get_result(state, record_player)
    tokens = game.get_tokens(state, record_player, 200)
    color = state.color_o if record_player == 1 else state.color_p

    policies = policies[tokens[:, 4]]

    return Sample(tokens, policies, record_player, reward, color)


def testplay(train_state, model, num_mcts_simulations, dirichlet_alpha=None, n_testplay=10):
    result = 0

    for _ in range(n_testplay):
        alphazero = np.random.choice([1, -1])
        player = 1

        state = game.get_initial_state()

        node1 = mcts.create_root_node(state, train_state, model, 1)
        node2 = mcts.create_root_node(state, train_state, model, -1)

        for i in range(200):
            if player == alphazero:
                policy, node1, node2 = mcts.step(node1, node2, state, player, train_state, num_mcts_simulations)
            else:
                action = game.greedy_action(state, player, epsilon=1)
                state.step(action, player)

            if game.is_done(state, player):
                break

            player = -player

        result += state.winner

    return result / n_testplay


CKPT_DIR = './checkpoints/'
PREFIX = 'geister_'


def create_model():
    return network.TransformerDecoderWithCache(num_heads=8, embed_dim=128, num_hidden_layers=2)


def main(num_cpus, n_episodes=20000, buffer_size=40000,
         batch_size=32, epochs_per_update=4,
         num_mcts_simulations=25,
         update_period=200, test_period=100,
         n_testplay=5,
         dirichlet_alpha=0.2):

    wandb.init(project="geister-zero",
               config={"dirichlet_alpha": 0.2})

    model = network.TransformerDecoder(num_heads=8, embed_dim=128, num_hidden_layers=2)

    ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, prefix=PREFIX, target=None)
    state = network.TrainState.create(
        apply_fn=model.apply,
        params=ckpt['params'],
        tx=optax.adam(learning_rate=0.00005),
        dropout_rng=ckpt['dropout_rng'],
        epoch=ckpt['epoch'])

    sample_queue = mp.Queue()
    n_updates = mp.Value('i', 0)
    replay = ReplayBuffer(buffer_size=buffer_size)

    def create_process(q, n):
        args = q, n, num_mcts_simulations, dirichlet_alpha
        return mp.Process(target=start_selfplay_process, args=args)

    processes = [create_process(sample_queue, n_updates) for _ in range(16)]
    for process in processes:
        process.start()

    while True:
        for i in tqdm(range(update_period)):
            while sample_queue.empty():
                pass
            sample = sample_queue.get()
            replay.add_record([sample])

        num_iters = epochs_per_update * (len(replay) // batch_size)
        for i in range(num_iters):
            batch = replay.get_minibatch(batch_size=batch_size)

            state, loss, info = network.train_step(state, *batch, eval=False)

        wandb.log({"loss": loss,
                   "loss policy": info[0],
                   "loss reward": info[1],
                   "loss pieces": info[2],
                   "acc pieces": info[3]
                   })

        checkpoints.save_checkpoint(
            ckpt_dir=CKPT_DIR, prefix=PREFIX,
            target=state, step=n_updates.value, overwrite=True, keep=5)

        n_updates.value += 1


if __name__ == "__main__":
    try:
        main(num_cpus=31)

    except Exception:
        import traceback
        with open('error.log', 'w') as f:
            traceback.print_exc(file=f)
