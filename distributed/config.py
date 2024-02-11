from dataclasses import dataclass, asdict, replace

from serde import serde
from serde.core import InternalTagging
from serde.json import from_json

import numpy as np
import jax
import jax.numpy as jnp

import match_makers
from constants import SearchParametersRange
from batch import ReplayBuffer

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
class WinRate:
    threshold: float

    def is_league_member(self, win_rates: np.ndarray):
        return np.all(win_rates > self.threshold)


ConditionForKeepingSnapshots = WinRate


@dataclass
class TrainingConfig:
    batch_size: int
    num_batches: int
    learning_rate: float


@dataclass
class MatchMakingConfig:
    selfplay_p: float
    mathod: match_makers.MatchMakingMethod
    buffer_size: int


@serde(tagging=InternalTagging(tag='type'))
@dataclass
class AgentConfig(SerdeJsonSerializable):
    name: str
    opponent_names: list[str] = None

    init_replay_buffer: str = None
    buffer_size: int = None

    init_params: InitModelConfig = None
    training: TrainingConfig = None

    match_making: MatchMakingConfig = None
    condition_for_keeping_snapshots: ConditionForKeepingSnapshots = None

    mcts_params: SearchParametersRange = None

    def override(self, other: "AgentConfig") -> "AgentConfig":
        def _override(_base, _other):
            return _other if _other is not None else _base

        base_d = asdict(self)
        other_d = asdict(other)
        return AgentConfig(**{k: _override(base_d[k], other_d[k]) for k in base_d.keys()})

    def create_match_maker(self):
        return match_makers.MatchMaker(
            method=self.match_making.mathod,
            match_buffer_size=self.match_making.buffer_size
        )

    def create_replay_buffer(self, series_length: int, tokens_length: int) -> ReplayBuffer:
        buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            sample_shape=(series_length,),
            seq_length=tokens_length
        )

        if self.init_replay_buffer is not None:
            buffer.load(self.init_replay_buffer)

        return buffer


@serde(tagging=InternalTagging(tag='type'))
@dataclass
class RunConfig(SerdeJsonSerializable):
    project_name: str
    wandb_log: bool

    series_length: int
    tokens_length: int
    update_period: int

    replay_buffer_size: int

    agents: list[AgentConfig]

    project_dir: str
    ckpt_options: CheckpointManagerOptions

    def get_checkpoint_dir(self, agent_name: str) -> str:
        return f'{self.project_dir}/checkpoints/{agent_name}'

    def get_replay_buffer_path(self, agent_name: str) -> str:
        return f'{self.project_dir}/replay_buffer/{agent_name}'

    @classmethod
    def from_json(cls, json_str) -> "RunConfig":
        config = from_json(cls, json_str)
        return replace(config, agents=config._get_agent_configs())

    def _get_agent_configs(self, common_setting_name="common_setting") -> list[AgentConfig]:
        agents_dict = {agent.name: agent for agent in self.agents}

        if common_setting_name not in agents_dict:
            return agents_dict

        for name in agents_dict.keys():
            if name == common_setting_name:
                continue

            agents_dict[name] = agents_dict[common_setting_name].override(agents_dict[name])

        del agents_dict[common_setting_name]

        return list(agents_dict.values())


def test():
    c1 = RunConfig.from_json_file('./data/run_config.json')
    s = c1.to_json()
    c2 = RunConfig.from_json(s)

    print(c1)
    print(c2)


if __name__ == '__main__':
    test()
