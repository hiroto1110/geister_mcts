import socket

from tqdm import tqdm
import click

import numpy as np
import jax
import optax

from network.transformer import TrainStateTransformer, Transformer
from network.checkpoints import Checkpoint

from distributed.messages import (
    MessageLeanerInitServer,
    MessageLearningRequest, MessageLearningResult
)
from distributed.communication import EncryptedCommunicator
from distributed.config import TrainingConfig


def train(
    state: TrainStateTransformer,
    model: Transformer,
    minibatch: np.ndarray,
    config: TrainingConfig
) -> tuple[TrainStateTransformer, MessageLearningResult]:

    train_batches = np.split(minibatch, len(minibatch) // config.batch_size)
    num_batches = len(train_batches)

    loss = 0
    losses = []

    for i in tqdm(range(num_batches), desc=f' Training '):
        state, loss_i, losses_i = state.train_step(
            train_batches[i], eval=False
        )
        loss += loss_i
        losses.append(losses_i)

    loss /= num_batches * config.batch_size

    losses = np.reshape(losses, (-1, 3))
    losses = np.mean(losses, axis=0)

    state = state.replace(epoch=state.epoch + 1)

    result = MessageLearningResult(
        ckpt=Checkpoint(int(state.epoch), model, state.params),
        loss=float(loss),
        loss_policy=float(losses[0]),
        loss_value=float(losses[1]),
        loss_color=float(losses[2]),
    )
    return state, result


@click.command()
@click.argument('host', type=str)
@click.argument('port', type=int)
@click.argument('password', type=str)
def main(host: str, port: int, password: str):
    start_learner(host, port, password)

def start_learner(host: str, port: int, password: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    communicator = EncryptedCommunicator(password)

    init_msg = communicator.recv_json_obj(sock, MessageLeanerInitServer)
    config = init_msg.config
    print(config)

    model = init_msg.ckpt.model.create_model()

    state = TrainStateTransformer.create(
        apply_fn=model.apply,
        params=init_msg.ckpt.params,
        tx=optax.adam(learning_rate=config.agent.training.learning_rate),
        dropout_rng=jax.random.PRNGKey(0),
        epoch=0
    )

    while True:
        request = communicator.recv_json_obj(sock, MessageLearningRequest)

        state, result = train(state, model, request.minibatch, config.agent.training)

        communicator.send_json_obj(sock, result)


if __name__ == "__main__":
    try:
        main()

    except Exception:
        import traceback
        traceback.print_exc()

        with open('error.log', 'w') as f:
            traceback.print_exc(file=f)
