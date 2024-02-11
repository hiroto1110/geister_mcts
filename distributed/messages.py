from dataclasses import dataclass

import numpy as np

from distributed.config import RunConfig
from distributed.communication import SerdeJsonSerializable

from network.checkpoints import Checkpoint


@dataclass(frozen=True)
class SnapshotInfo:
    name: str
    step: int


SNAPSHOT_INFO_SELFPLAY = SnapshotInfo(name='__selfplay__', step=-1)


@dataclass
class MatchInfo(SerdeJsonSerializable):
    player: SnapshotInfo
    opponent: SnapshotInfo


@dataclass
class MessageActorInitClient(SerdeJsonSerializable):
    n_processes: int


@dataclass
class MessageActorInitServer(SerdeJsonSerializable):
    config: RunConfig
    snapshots: dict[str, list[Checkpoint]]
    matches: list[MatchInfo]


@dataclass
class MessageLeanerInitServer(SerdeJsonSerializable):
    config: RunConfig
    ckpts: dict[str, Checkpoint]


@dataclass
class MessageNextMatch(SerdeJsonSerializable):
    match: MatchInfo
    ckpt: Checkpoint | None


@dataclass
class MessageMatchResult(SerdeJsonSerializable):
    match: MatchInfo
    samples: list[np.ndarray]


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
