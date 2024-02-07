from dataclasses import dataclass

from serde import serde
from serde.core import InternalTagging
from serde.json import from_json, to_json

import jax
import jax.numpy as jnp

import orbax.checkpoint

import match_makers
import mcts

from network.transformer import Transformer
from network.train import Checkpoint


@dataclass
class FromCheckpoint:
    dir_name: str
    step: int

    def create_model_and_params(self):
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint_manager = orbax.checkpoint.CheckpointManager(self.dir_name, checkpointer)
        ckpt = Checkpoint.load(checkpoint_manager, self.step)

        return ckpt.model, ckpt.params


@dataclass
class Random:
    model: Transformer

    def create_model_and_params(self):
        init_data = jnp.zeros((1, 200, 5), dtype=jnp.uint8)
        variables = self.model.init(jax.random.PRNGKey(0), init_data)
        params = variables['params']

        return self.model, params


InitModelConfig = Random | FromCheckpoint


@serde(tagging=InternalTagging(tag='type'))
@dataclass
class RunConfig:
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
    mcts_params_min: mcts.SearchParameters
    mcts_params_max: mcts.SearchParameters
    ckpt_dir: str
    ckpt_options: orbax.checkpoint.CheckpointManagerOptions
    load_replay_buffer_path: str
    save_replay_buffer_path: str
    init_params: InitModelConfig

    @classmethod
    def from_json(cls, s: str) -> 'RunConfig':
        return from_json(cls, s)

    @classmethod
    def from_json_file(cls, path) -> 'RunConfig':
        with open(path, mode='r') as f:
            return from_json(cls, f.read())

    def to_json(self) -> str:
        return to_json(self)

    def to_json_file(self, path):
        s = to_json(self)

        with open(path, mode='w') as f:
            f.write(s)
