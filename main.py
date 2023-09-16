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


def start_selfplay_process(sender, n_updates, seed: int, num_mcts_sim: int, dirichlet_alpha: float):
    np.random.seed(seed)

    with jax.default_device(jax.devices("cpu")[0]):
        model = create_model()

        ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, prefix=PREFIX, target=None)
        params = ckpt['params']

        last_n_updates = n_updates.value

        while True:
            # num_mcts_simu1, num_mcts_simu2 = np.random.randint(num_mcts_sim // 2, num_mcts_sim, size=2)
            sample = selfplay(model, params, params, num_mcts_sim, num_mcts_sim, dirichlet_alpha)

            sender.send(sample)

            if last_n_updates != n_updates.value:
                ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, prefix=PREFIX, target=None)
                params = ckpt['params']

                last_n_updates = n_updates.value


def selfplay(model: network.TransformerDecoderWithCache,
             params1, params2,
             num_mcts_sim1: int, num_mcts_sim2: int,
             dirichlet_alpha):

    record_player = np.random.choice([1, -1])

    tokens_ls, actions, reward, color = mcts.play_game(model,
                                                       params1, params2,
                                                       num_mcts_sim1, num_mcts_sim2,
                                                       dirichlet_alpha,
                                                       record_player)

    tokens = np.zeros((200, 5), dtype=np.uint8)
    tokens[:min(200, len(tokens_ls))] = tokens_ls[:200]

    mask = np.zeros(200, dtype=np.uint8)
    mask[:len(tokens_ls)] = 1

    actions = actions[tokens[:, 4]]
    reward = reward + 3

    return Sample(tokens, mask, actions, reward, color)


def start_testplay_process(queue: mp.Queue, num_games, num_mcts_sim, dirichlet_alpha):
    with jax.default_device(jax.devices("cpu")[0]):
        model = create_model()

        ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, prefix=PREFIX, target=None)
        params = ckpt['params']

        ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, prefix=PREFIX_BEST, target=None)
        best_params = ckpt['params']

        win_count = 0

        for i in range(num_games):
            if i % 2 == 0:
                _, _, reward, _ = mcts.play_game(model, params, best_params,
                                                 num_mcts_sim, num_mcts_sim, dirichlet_alpha, 1)
            else:
                _, _, reward, _ = mcts.play_game(model, best_params, params,
                                                 num_mcts_sim, num_mcts_sim, dirichlet_alpha, -1)

            if reward > 0:
                win_count += 1

        queue.put(win_count / num_games)


CKPT_DIR = './checkpoints/'
PREFIX = 'geister_'
PREFIX_BEST = 'geister_best_'
PREFIX_BACKUP = 'geister_backup_'


def create_model():
    return network.TransformerDecoderWithCache(num_heads=8, embed_dim=128, num_hidden_layers=2)


def main(n_clients=30,
         buffer_size=100000,
         batch_size=256,
         update_period=400,
         num_mcts_sim=50,
         dirichlet_alpha=0.3):

    wandb.init(project="geister-zero",
               config={"dirichlet_alpha": dirichlet_alpha})

    model = network.TransformerDecoder(num_heads=8, embed_dim=128, num_hidden_layers=2)

    ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, prefix=PREFIX, target=None)
    state = network.TrainState.create(
        apply_fn=model.apply,
        params=ckpt['params'],
        tx=optax.adam(learning_rate=0.00005),
        dropout_rng=ckpt['dropout_rng'],
        epoch=ckpt['epoch'])

    result_testplay_queue = mp.Queue()
    pipe = MultiSenderPipe(n_clients)
    n_updates = mp.Value('i', 0)

    for i in range(n_clients):
        sender = pipe.get_sender(i)
        seed = np.random.randint(0, 10000)
        args = sender, n_updates, seed, num_mcts_sim, dirichlet_alpha

        process = mp.Process(target=start_selfplay_process, args=args)
        process.start()

    replay_buffer = ReplayBuffer(buffer_size=buffer_size, seq_length=game.MAX_TOKEN_LENGTH)
    # replay_buffer.load('replay_buffer')

    while True:
        for i in tqdm(range(update_period)):
            while not pipe.poll():
                pass

            sample = pipe.recv()
            replay_buffer.add_sample(sample)

        replay_buffer.save('replay_buffer')

        state = train_and_log(state, replay_buffer, batch_size, update_period)

        save_checkpoint(state)

        if (n_updates.value % 10 == 0) and not result_testplay_queue.empty():
            result = result_testplay_queue.get()

            wandb.log({"step": state.epoch,
                       "test_play": result})

            if result > 0.55:
                checkpoints.save_checkpoint(
                    ckpt_dir=CKPT_DIR, prefix=PREFIX_BEST,
                    target=state, step=state.epoch, overwrite=True, keep=1000)

            args = result_testplay_queue, 100, num_mcts_sim, dirichlet_alpha
            process = mp.Process(target=start_testplay_process, args=args)
            process.start()

        n_updates.value += 1


def train_and_log(state: network.TrainState,
                  buffer: ReplayBuffer,
                  train_batch_size: int,
                  test_batch_size: int):

    num_iters = (len(buffer) // train_batch_size) // 4
    if num_iters <= 0:
        return state

    info = np.zeros((num_iters, 3))
    loss = 0

    for i in range(num_iters):
        train_batch = buffer.get_minibatch(batch_size=train_batch_size)
        state, loss_i, info_i = network.train_step(state, *train_batch.astuple(), eval=False)

        loss += loss_i
        info[i] = info_i

    info = info.mean(axis=0)

    test_batch = buffer.get_last__minibatch(batch_size=test_batch_size)
    _, test_loss, test_info = network.train_step(state, *test_batch.astuple(), eval=True)

    n_ply = test_batch.tokens[:, :, game.Token.T].max(axis=1)

    log_dict = {"step": state.epoch,
                "train/loss": loss / num_iters,
                "train/loss policy": info[0],
                "train/loss value": info[1],
                "train/loss color": info[2],

                "test/loss": test_loss,
                "test/loss policy": test_info[0],
                "test/loss value": test_info[1],
                "test/loss color": test_info[2],

                "n_ply": n_ply.mean(),
                "n_ply histgram": wandb.Histogram(n_ply, num_bins=200)}

    value = test_batch.reward.flatten()
    value_count = np.bincount(np.abs(value - 3), minlength=4)

    log_dict.update({f'value/{i}': value_count[i] for i in range(4)})

    wandb.log(log_dict)

    return state.replace(epoch=state.epoch + 1)


def save_checkpoint(state: network.TrainState):
    checkpoints.save_checkpoint(
            ckpt_dir=CKPT_DIR, prefix=PREFIX,
            target=state, step=state.epoch, overwrite=True, keep=50)

    if state.epoch % 100 == 0:
        checkpoints.save_checkpoint(
            ckpt_dir=CKPT_DIR, prefix=PREFIX_BACKUP,
            target=state, step=state.epoch, overwrite=True, keep=500)


if __name__ == "__main__":
    try:
        main()

    except Exception:
        import traceback
        with open('error.log', 'w') as f:
            traceback.print_exc(file=f)
