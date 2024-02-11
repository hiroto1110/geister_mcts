import socket

from tqdm import tqdm
import click

import numpy as np
import jax
import optax

from network.train import TrainState, train_step
from network.checkpoints import Checkpoint
from batch import astuple

from messages import (
    MessageLeanerInitServer,
    MessageLearningRequest, LearningJob,
    MessageLearningJobResult, LearningJobResult
)
from distributed.communication import EncryptedCommunicator
from config import AgentConfig


class Agent:
    def __init__(self, config: AgentConfig, init_ckpt: Checkpoint) -> None:
        self.config = config
        self.model = init_ckpt.model

        model = self.model.create_model()

        self.state = TrainState.create(
            apply_fn=model.apply,
            params=init_ckpt.params,
            tx=optax.adam(learning_rate=config.training.learning_rate),
            dropout_rng=jax.random.PRNGKey(0),
            epoch=0,
            init_memory=TrainState.create_init_memory(model)
        )

    def train(self, job: LearningJob):
        train_batches = np.split(job.minibatch, len(job.minibatch) // self.config.training.batch_size)
        num_batches = len(train_batches)

        loss = 0
        losses = []

        for i in tqdm(range(num_batches), desc=f' Training {self.config.name} '):
            self.state, loss_i, losses_i = train_step(
                self.state,
                *astuple(train_batches[i]),
                num_division_of_segment=4,
                eval=False
            )
            loss += loss_i
            losses.append(losses_i)

        loss /= num_batches * self.config.training.batch_size

        losses = np.reshape(losses, (-1, 3))
        losses = np.mean(losses, axis=0)

        self.state = self.state.replace(epoch=self.state.epoch + 1)

        return LearningJobResult(
            self.config.name,
            ckpt=Checkpoint(self.state.epoch, self.model, self.state.params),
            loss=float(loss),
            loss_policy=float(losses[0]),
            loss_value=float(losses[1]),
            loss_color=float(losses[2]),
        )


@click.command()
@click.argument('host', type=str)
@click.argument('port', type=int)
@click.argument('password', type=str)
def main(host: str, port: int, password: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    communicator = EncryptedCommunicator(password)

    init_msg = communicator.recv_json_obj(sock, MessageLeanerInitServer)
    config = init_msg.config
    print(config)

    agents: dict[str, Agent] = {
        Agent(agent, init_msg.ckpts[agent.name]) for agent in config.agents
    }

    while True:
        request = communicator.recv_json_obj(sock, MessageLearningRequest)

        results = [agents[job.agent_name].train(job) for job in request.jobs]

        communicator.send_json_obj(sock, MessageLearningJobResult(results))


if __name__ == "__main__":
    try:
        main()

    except Exception:
        import traceback
        traceback.print_exc()

        with open('error.log', 'w') as f:
            traceback.print_exc(file=f)
