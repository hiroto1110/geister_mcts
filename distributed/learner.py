import socket
import pickle

from tqdm import tqdm
import click

import numpy as np
import jax
import optax

from network.train import Checkpoint, TrainState, train_step
from batch import astuple

from config import RunConfig
from collector import MessageLearningRequest, MessageLearningResult
from socket_util import EncryptedCommunicator


@click.command()
@click.argument('host', type=str)
@click.argument('port', type=int)
@click.argument('password', type=str)
def main(host: str, port: int, password: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    communicator = EncryptedCommunicator(password)

    config: RunConfig = pickle.loads(communicator.recv_bytes(sock))
    print(config)

    while True:
        request: MessageLearningRequest = pickle.loads(communicator.recv_bytes(sock))

        state = TrainState.create(
            apply_fn=request.ckpt.model.apply,
            params=request.ckpt.params,
            tx=optax.adam(learning_rate=config.learning_rate),
            dropout_rng=jax.random.PRNGKey(0),
            epoch=request.ckpt.step,
            init_memory=TrainState.create_init_memory(request.ckpt.model)
        )

        if request.ckpt.opt_state is not None:
            state = state.replace(opt_state=request.ckpt.opt_state)

        state, train_log_dict = train(state, request.minibatch, config.batch_size)

        result = MessageLearningResult(
            ckpt=Checkpoint(state.epoch, state.params, request.ckpt.model, state.opt_state),
            log=train_log_dict
        )

        communicator.send_bytes(sock, pickle.dumps(result))


def train(
    state: TrainState,
    train_batch: np.ndarray,
    batch_size: int
) -> tuple[TrainState, dict]:
    train_batches = np.split(train_batch, len(train_batch) // batch_size)
    num_batches = len(train_batches)

    loss = 0
    losses = []

    for i in tqdm(range(num_batches), desc=' Training '):
        state, loss_i, losses_i = train_step(state, *astuple(train_batches[i]), num_division_of_segment=4, eval=False)

        loss += loss_i
        losses.append(losses_i)

    loss /= num_batches

    num_division = 2
    losses = np.reshape(losses, (num_batches, num_division, -1, 3))
    losses = np.mean(losses, axis=(0, 2))

    log_dict = {'train/loss': loss}

    for i in range(num_division):
        for j, name in enumerate(['policy', 'value', 'color']):
            log_dict[f'train/loss {name} {i}'] = losses[i, j]

    print(log_dict)

    return state.replace(epoch=state.epoch + 1), log_dict


if __name__ == "__main__":
    try:
        main()

    except Exception:
        import traceback
        traceback.print_exc()

        with open('error.log', 'w') as f:
            traceback.print_exc(file=f)
