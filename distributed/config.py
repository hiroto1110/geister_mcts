from dataclasses import dataclass

from serde import serialize, deserialize
from serde.json import from_json, to_json

import match_makers
import mcts


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


@serialize
@deserialize
@dataclass
class RunConfig:
    series_length: int
    batch_size: int
    num_batches: int
    buffer_size: int
    update_period: int
    save_period: int
    match_maker: MatchMakerConfig
    fsp_threshold: float
    mcts_params: mcts.SearchParameters
    ckpt_dir: str
    minibatch_temp_path: str

    @classmethod
    def from_json_file(cls, path) -> 'RunConfig':
        with open(path, mode='r') as f:
            return from_json(cls, f.read())

    def to_json_file(self, path):
        s = to_json(self)

        with open(path, mode='w') as f:
            f.write(s)
