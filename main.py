import multiprocessing as mp

import optax
from flax.training import checkpoints
import jax
import numpy as np

from tqdm import tqdm
import wandb

from buffer import ReplayBuffer
from multiprocessing_util import MultiSenderPipe
import network_transformer as network
import geister as game
import mcts


def start_selfplay_process(sender, n_updates, seed: int, num_mcts_sim: int, dirichlet_alpha: float):
    np.random.seed(seed)

    with jax.default_device(jax.devices("cpu")[0]):
        model = create_model()

        ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, prefix=PREFIX, target=None)

        last_n_updates = n_updates.value

        player1 = mcts.PlayerMCTS(ckpt['params'], model, num_mcts_sim, dirichlet_alpha)
        player2 = mcts.PlayerMCTS(ckpt['params'], model, num_mcts_sim, dirichlet_alpha)

        weight_v_default = np.array([-1, -1, -1, 0, 1, 1, 1])

        while True:
            player1.weight_v = np.random.normal(weight_v_default, scale=0.3)
            player2.weight_v = np.random.normal(weight_v_default, scale=0.3)

            sample1, sample2 = selfplay(player1, player2)

            sender.send(sample1)
            sender.send(sample2)

            if last_n_updates != n_updates.value:
                ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, prefix=PREFIX, target=None)

                player1.update_params(ckpt['params'])
                player2.update_params(ckpt['params'])

                last_n_updates = n_updates.value


def selfplay(player1: mcts.PlayerMCTS, player2: mcts.PlayerMCTS):
    actions, color1, color2 = mcts.play_game(player1, player2)

    sample1 = player1.create_sample(actions, color2)
    sample2 = player2.create_sample(actions, color1)

    return sample1, sample2


CKPT_DIR = './checkpoints/'
CKPT_BACKUP_DIR = './checkpoints_backup/'
PREFIX = 'geister_'


def create_model():
    return network.TransformerDecoderWithCache(num_heads=8, embed_dim=128, num_hidden_layers=4)


def main(n_clients=30,
         buffer_size=200000,
         batch_size=128,
         num_batches=40,
         update_period=200,
         num_mcts_sim=100,
         dirichlet_alpha=0.3):

    wandb.init(project="geister-zero",
               config={"dirichlet_alpha": dirichlet_alpha})

    model = network.TransformerDecoder(num_heads=8, embed_dim=128, num_hidden_layers=4)

    ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, prefix=PREFIX, target=None)
    state = network.TrainState.create(
        apply_fn=model.apply,
        params=ckpt['params'],
        tx=optax.adam(learning_rate=0.0005),
        dropout_rng=ckpt['dropout_rng'],
        epoch=ckpt['epoch'])

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

        if state.epoch % (buffer_size // update_period) == 0:
            replay_buffer.save('replay_buffer', append=n_updates.value > 0)

        log_dict = {"step": state.epoch}

        log_games(replay_buffer, update_period, log_dict)
        state = train_and_log(state, replay_buffer, batch_size, num_batches, log_dict)

        wandb.log(log_dict)

        save_checkpoint(state)

        n_updates.value += 1


def log_games(buffer: ReplayBuffer, num_games: int, log_dict):
    batch = buffer.get_last__minibatch(batch_size=num_games)

    n_ply = batch.tokens[:, :, game.Token.T].max(axis=1)

    log_dict["n_ply"] = n_ply.mean()
    log_dict["n_ply histgram"] = wandb.Histogram(n_ply, num_bins=50)

    value = batch.reward.flatten()
    value_count = np.bincount(np.abs(value - 3), minlength=4)

    for i in range(4):
        log_dict[f'value/{i}'] = value_count[i] / num_games


def train_and_log(state: network.TrainState,
                  buffer: ReplayBuffer,
                  train_batch_size: int,
                  num_batches: int,
                  log_dict: dict):

    if len(buffer) < num_batches * train_batch_size:
        return state

    info = np.zeros((num_batches, 3))
    loss = 0

    for i in range(num_batches):
        train_batch = buffer.get_minibatch(batch_size=train_batch_size)
        state, loss_i, info_i = network.train_step(state, *train_batch.astuple(), eval=False)

        loss += loss_i
        info[i] = info_i

    info = info.mean(axis=0)
    loss /= num_batches

    log_dict["train/loss"] = loss
    log_dict["train/loss policy"] = info[0]
    log_dict["train/loss value"] = info[1]
    log_dict["train/loss color"] = info[2]

    return state.replace(epoch=state.epoch + 1)


def save_checkpoint(state: network.TrainState):
    checkpoints.save_checkpoint(
            ckpt_dir=CKPT_DIR, prefix=PREFIX,
            target=state, step=state.epoch, overwrite=True, keep=50)

    if state.epoch % 100 == 0:
        checkpoints.save_checkpoint(
            ckpt_dir=CKPT_BACKUP_DIR, prefix=PREFIX,
            target=state, step=state.epoch, overwrite=True, keep=500)


if __name__ == "__main__":
    try:
        main()

    except Exception:
        import traceback
        with open('error.log', 'w') as f:
            traceback.print_exc(file=f)
