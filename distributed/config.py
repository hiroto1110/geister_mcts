from dataclasses import dataclass

from serde import serde
from serde.core import InternalTagging
from serde.json import from_json, to_json

import match_makers
import mcts

from network.transformer import Transformer


@dataclass
class MatchMakerConfig:
    type: str
    parameters: dict

    def create_match_maker(self) -> match_makers.MatchMaker:
        match self.type:
            case 'fsp':
                return match_makers.MatchMakerFSP(n_agents=1, **self.parameters)

            case _:
                raise NotImplementedError()


@dataclass
class InitWithCheckpointConfig:
    dir_name: str
    step: int


@dataclass
class InitWithRandomConfig:
    model: Transformer


InitModelConfig = InitWithCheckpointConfig | InitWithRandomConfig


@serde(tagging=InternalTagging(tag='type'))
@dataclass
class RunConfig:
    project_name: str
    series_length: int
    tokens_length: int
    batch_size: int
    num_batches: int
    buffer_size: int
    update_period: int
    learning_rate: float
    match_maker: MatchMakerConfig
    fsp_threshold: float
    mcts_params: mcts.SearchParameters
    ckpt_dir: str
    load_replay_buffer_path: str
    save_replay_buffer_path: str
    init_params: InitModelConfig
    minibatch_temp_path: str = './data/replay_buffer/minibatch_tmp.npz'

    @classmethod
    def from_json_file(cls, path) -> 'RunConfig':
        with open(path, mode='r') as f:
            return from_json(cls, f.read())

    def to_json_file(self, path):
        s = to_json(self)

        with open(path, mode='w') as f:
            f.write(s)
