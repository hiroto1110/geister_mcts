from dataclasses import dataclass

import numpy as np

from distributed.config import RunConfig
from distributed.communication import SerdeJsonSerializable

from network.checkpoints import Checkpoint
from players import Configs


@dataclass(frozen=True)
class MatchInfo(SerdeJsonSerializable):
    player: Configs
    opponent: Configs


@dataclass(frozen=True)
class MessageActorInitClient(SerdeJsonSerializable):
    n_processes: int


@dataclass(frozen=True)
class MessageActorInitServer(SerdeJsonSerializable):
    series_length: int
    tokens_length: int
    snapshots: list[Checkpoint]
    matches: list[MatchInfo]


@dataclass(frozen=True)
class MessageLeanerInitServer(SerdeJsonSerializable):
    config: RunConfig
    ckpt: Checkpoint


@dataclass(frozen=True)
class MessageNextMatch(SerdeJsonSerializable):
    match: MatchInfo
    ckpts: list[Checkpoint]


@dataclass(frozen=True)
class MessageMatchResult(SerdeJsonSerializable):
    match: MatchInfo
    samples: np.ndarray


@dataclass(frozen=True)
class MessageLearningRequest(SerdeJsonSerializable):
    minibatch: np.ndarray


@dataclass(frozen=True)
class MessageLearningResult(SerdeJsonSerializable):
    ckpt: Checkpoint
    loss: float
    loss_policy: float
    loss_value: float
    loss_color: float
