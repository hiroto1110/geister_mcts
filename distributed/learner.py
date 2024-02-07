import socket

from tqdm import tqdm
import click

import numpy as np
import jax
import optax

from network.train import TrainState, train_step
from network.checkpoints import Checkpoint, make_pytree_structures_same
from batch import astuple

from config import RunConfig
from collector import MessageLearningRequest, MessageLearningResult, Losses
from distributed.communication import EncryptedCommunicator


@click.command()
@click.argument('host', type=str)
@click.argument('port', type=int)
@click.argument('password', type=str)
def main(host: str, port: int, password: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    communicator = EncryptedCommunicator(password)

    config = communicator.recv_json_obj(sock, RunConfig)
    print(config)

    while True:
        request = communicator.recv_json_obj(sock, MessageLearningRequest)

        model = request.ckpt.model.create_model()

        state = TrainState.create(
            apply_fn=model.apply,
            params=request.ckpt.params,
            tx=optax.adam(learning_rate=config.learning_rate),
            dropout_rng=jax.random.PRNGKey(0),
            epoch=request.ckpt.step,
            init_memory=TrainState.create_init_memory(model)
        )

        if request.ckpt.opt_state is not None:
            opt_state = make_pytree_structures_same(request.ckpt.opt_state, state.opt_state)
            state = state.replace(opt_state=opt_state)

        state, losses = train(state, request.minibatch, config.batch_size)

        result = MessageLearningResult(
            losses=losses,
            ckpt=Checkpoint(int(state.epoch), request.ckpt.model, state.params, state.opt_state),
            # ckpt=Checkpoint(state.epoch, request.ckpt.model, None, None),
        )
        communicator.send_json_obj(sock, result)


def train(
    state: TrainState,
    train_batch: np.ndarray,
    batch_size: int
) -> tuple[TrainState, Losses]:
    train_batches = np.split(train_batch, len(train_batch) // batch_size)
    num_batches = len(train_batches)

    loss = 0
    losses = []

    for i in tqdm(range(num_batches), desc=' Training '):
        state, loss_i, losses_i = train_step(state, *astuple(train_batches[i]), num_division_of_segment=4, eval=False)

        loss += loss_i
        losses.append(losses_i)

    loss /= num_batches

    losses = np.reshape(losses, (-1, 3))
    losses = np.mean(losses, axis=0)

    losses = Losses(
        loss=float(loss),
        loss_policy=float(losses[0]),
        loss_value=float(losses[1]),
        loss_color=float(losses[2]),
    )
    print(losses)

    return state.replace(epoch=state.epoch + 1), losses


if __name__ == "__main__":
    try:
        main()

    except Exception:
        import traceback
        traceback.print_exc()

        with open('error.log', 'w') as f:
            traceback.print_exc(file=f)
