from dataclasses import dataclass

from serde import serde
from serde.core import InternalTagging
from serde.json import from_json, to_json

import jax
import jax.numpy as jnp

import match_makers
from constants import SearchParameters

from distributed.communication import SerdeJsonSerializable
from network.transformer import TransformerConfig
from network.checkpoints import CheckpointManager, CheckpointManagerOptions


@dataclass
class FromCheckpoint:
    dir_name: str
    step: int

    def create_model_and_params(self) -> tuple[TransformerConfig, dict]:
        checkpoint_manager = CheckpointManager(self.dir_name)
        ckpt = checkpoint_manager.load(self.step)

        return ckpt.model, ckpt.params


@dataclass
class Random:
    model: TransformerConfig

    def create_model_and_params(self) -> tuple[TransformerConfig, dict]:
        init_data = jnp.zeros((1, 200, 5), dtype=jnp.uint8)
        variables = self.model.init(jax.random.PRNGKey(0), init_data)
        params = variables['params']

        return self.model, params


InitModelConfig = Random | FromCheckpoint


@serde(tagging=InternalTagging(tag='type'))
@dataclass
class RunConfig(SerdeJsonSerializable):
    project_name: str
    wandb_log: bool
    series_length: int
    tokens_length: int
    batch_size: int
    num_batches: int
    buffer_size: int
    update_period: int
    learning_rate: float
    selfplay_p: float
    match_maker_buffer_size: int
    match_making: match_makers.MatchMakingMethod
    fsp_threshold: float
    mcts_params_min: SearchParameters
    mcts_params_max: SearchParameters
    ckpt_dir: str
    ckpt_options: CheckpointManagerOptions
    load_replay_buffer_path: str
    save_replay_buffer_path: str
    init_params: InitModelConfig

    @classmethod
    def from_json_file(cls, path) -> 'RunConfig':
        with open(path, mode='r') as f:
            return from_json(cls, f.read())

    def to_json_file(self, path):
        s = to_json(self)

        with open(path, mode='w') as f:
            f.write(s)


def test():
    c1 = RunConfig.from_json_file('./data/run_config.json')
    s = c1.to_json()
    c2 = RunConfig.from_json(s)

    print(c1)
    print(c2)


if __name__ == '__main__':
    test()
