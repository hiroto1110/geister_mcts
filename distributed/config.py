from dataclasses import dataclass, replace

from serde import serde
from serde.core import InternalTagging
from serde.json import from_json

import numpy as np
import jax
import jax.numpy as jnp

import match_makers
from players.config import SearchParametersRange
from batch import ReplayBuffer, FORMAT_X7_ST_PVC

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


@dataclass
class ConditionForKeepingSnapshots:
    win_rate_threshold: float | None = None
    step_period: int | None = None

    def is_league_member(self, win_rates: np.ndarray, step: int):
        if (self.win_rate_threshold is not None) and np.all(win_rates > self.win_rate_threshold):
            return True

        if (self.step_period is not None) and (step % self.step_period) == 0:
            return True

        return False


@dataclass
class TrainingConfig:
    batch_size: int
    num_batches: int
    learning_rate: float


@serde(tagging=InternalTagging(tag='type'))
@dataclass
class MatchMakingConfig:
    mathod: match_makers.MatchMakingMethod
    buffer_size: int


@serde(tagging=InternalTagging(tag='type'))
@dataclass
class AgentConfig(SerdeJsonSerializable):
    init_params: InitModelConfig
    training: TrainingConfig

    match_making: MatchMakingConfig
    condition_for_keeping_snapshots: ConditionForKeepingSnapshots

    mcts_params: SearchParametersRange

    def create_match_maker(self):
        return match_makers.MatchMaker(
            method=self.match_making.mathod,
            match_buffer_size=self.match_making.buffer_size
        )


@serde(tagging=InternalTagging(tag='type'))
@dataclass
class RunConfig(SerdeJsonSerializable):
    project_name: str
    wandb_log: bool

    series_length: int
    tokens_length: int
    update_period: int

    replay_buffer_size: int
    init_replay_buffer: str | None

    agent: AgentConfig

    project_dir: str
    ckpt_options: CheckpointManagerOptions

    def create_replay_buffer(self) -> ReplayBuffer:
        buffer = ReplayBuffer(
            format=FORMAT_X7_ST_PVC,
            buffer_size=self.replay_buffer_size,
            sample_shape=(self.series_length,),
            seq_length=self.tokens_length
        )

        if self.init_replay_buffer is not None:
            buffer.load(self.init_replay_buffer)

        return buffer


def test():
    c1 = RunConfig.from_json_file('./data/run_config.json')
    s = c1.to_json()
    c2 = RunConfig.from_json(s)

    print(c2)
    


if __name__ == '__main__':
    test()
