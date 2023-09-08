import multiprocessing as mp

import optax
from flax.training import checkpoints
import jax
import numpy as np

from tqdm import tqdm
import wandb

from buffer import ReplayBuffer, Sample
from multiprocessing_util import MultiSenderPipe
import network_transformer as network
import geister as game
import mcts


def start_selfplay_process(sender, n_updates, num_mcts_sim: int, dirichlet_alpha: float):
    with jax.default_device(jax.devices("cpu")[0]):
        model = create_model()

        ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, prefix=PREFIX, target=None)
        pred_state = mcts.PredictState(model.apply, ckpt['params'])

        last_n_updates = n_updates.value

        while True:
            # num_mcts_simu1, num_mcts_simu2 = np.random.randint(num_mcts_sim // 2, num_mcts_sim, size=2)
            sample = selfplay(pred_state, model, num_mcts_sim, num_mcts_sim, dirichlet_alpha)

            sender.send(sample)

            if last_n_updates != n_updates.value:
                ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, prefix=PREFIX, target=None)
                pred_state = mcts.PredictState(model.apply, ckpt['params'])

                last_n_updates = n_updates.value


def selfplay(pred_state: mcts.PredictState,
             model: network.TransformerDecoderWithCache,
             num_mcts_sim1: int, num_mcts_sim2: int,
             dirichlet_alpha):

    state, actions = mcts.play_game(pred_state, model,
                                    num_mcts_sim1, num_mcts_sim2, dirichlet_alpha)

    record_player = np.random.choice([1, -1])

    tokens = state.get_tokens(record_player)
    actions = actions[tokens[:, 4]]
    reward = int(state.winner * state.win_type.value * record_player) + 3
    color = state.color_o if record_player == 1 else state.color_p

    return Sample(tokens, actions, reward, color)


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
    return network.TransformerDecoderWithCache(num_heads=8, embed_dim=128, num_hidden_layers=3)


def main(n_clients=30,
         buffer_size=40000,
         batch_size=128,
         epochs_per_update=1,
         update_period=200,
         num_mcts_sim=100,
         dirichlet_alpha=0.3):

    wandb.init(project="geister-zero",
               config={"dirichlet_alpha": dirichlet_alpha})

    model = network.TransformerDecoder(num_heads=8, embed_dim=128, num_hidden_layers=3)

    ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, prefix=PREFIX, target=None)
    state = network.TrainState.create(
        apply_fn=model.apply,
        params=ckpt['params'],
        tx=optax.adam(learning_rate=0.00005),
        dropout_rng=ckpt['dropout_rng'],
        epoch=ckpt['epoch'])

    pipe = MultiSenderPipe(n_clients)
    n_updates = mp.Value('i', 0)

    for i in range(n_clients):
        sender = pipe.get_sender(i)
        args = sender, n_updates, num_mcts_sim, dirichlet_alpha
        process = mp.Process(target=start_selfplay_process, args=args)
        process.start()

    replay = ReplayBuffer(buffer_size=buffer_size, seq_length=game.MAX_TOKEN_LENGTH)

    while True:
        for i in tqdm(range(update_period)):
            while not pipe.poll():
                pass

            sample = pipe.recv()
            replay.add_sample(sample)

        replay.save('replay_buffer')

        num_iters = epochs_per_update * (len(replay) // batch_size)
        info = np.zeros((num_iters, 4, game.MAX_TOKEN_LENGTH))
        loss = 0

        for i in range(num_iters):
            batch = replay.get_minibatch(batch_size=batch_size)
            state, loss_i, info_i = network.train_step(state, *batch, eval=False)

            loss += loss_i
            info[i] = info_i

        n_div = 4
        info = info.reshape(num_iters, 4, n_div, -1)
        info = info.mean(axis=(0, 3))

        log_dict = {"loss": loss / num_iters,
                    "value": wandb.Histogram(replay.reward_buffer[replay.index - update_period: replay.index]),
                    "n_ply": replay.tokens_buffer[replay.index - update_period: replay.index, :, 4].max(axis=1).mean(),
                    "num updates": n_updates.value}

        for i in range(n_div):
            log_dict.update({f"{i}/loss policy": info[0, i],
                             f"{i}/loss reward": info[1, i],
                             f"{i}/loss pieces": info[2, i],
                             f"{i}/acc pieces": info[3, i]})

        wandb.log(log_dict)

        state = state.replace(epoch=state.epoch + 1)
        checkpoints.save_checkpoint(
            ckpt_dir=CKPT_DIR, prefix=PREFIX,
            target=state, step=state.epoch, overwrite=True, keep=50)

        n_updates.value += 1


if __name__ == "__main__":
    try:
        main()

    except Exception:
        import traceback
        with open('error.log', 'w') as f:
            traceback.print_exc(file=f)
