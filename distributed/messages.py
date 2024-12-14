from dataclasses import dataclass

import numpy as np

from distributed.config import RunConfig
from distributed.communication import SerdeJsonSerializable

from network.checkpoints import Checkpoint
from players.base import PlayerConfig


@dataclass(frozen=True)
class SnapshotInfo:
    name: str
    step: int


@dataclass
class MatchInfo(SerdeJsonSerializable):
    player: PlayerConfig
    opponent: PlayerConfig


@dataclass
class MessageActorInitClient(SerdeJsonSerializable):
    n_processes: int


@dataclass
class MessageActorInitServer(SerdeJsonSerializable):
    config: RunConfig
    snapshots: list[Checkpoint]
    matches: list[MatchInfo]


@dataclass
class MessageLeanerInitServer(SerdeJsonSerializable):
    config: RunConfig
    ckpt: Checkpoint


@dataclass
class MessageNextMatch(SerdeJsonSerializable):
    match: MatchInfo
    ckpts: list[Checkpoint]


@dataclass
class MessageMatchResult(SerdeJsonSerializable):
    match: MatchInfo
    samples: np.ndarray


@dataclass
class LearningJob(SerdeJsonSerializable):
    agent_name: str
    minibatch: np.ndarray


@dataclass
class MessageLearningRequest(SerdeJsonSerializable):
    jobs: list[LearningJob]


@dataclass
class LearningJobResult(SerdeJsonSerializable):
    agent_name: str
    ckpt: Checkpoint
    loss: float
    loss_policy: float
    loss_value: float
    loss_color: float


@dataclass
class MessageLearningJobResult(SerdeJsonSerializable):
    results: list[LearningJobResult]
