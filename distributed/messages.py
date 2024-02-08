from dataclasses import dataclass

import numpy as np

from distributed.config import RunConfig
from distributed.communication import SerdeJsonSerializable

from network.checkpoints import Checkpoint


@dataclass
class MatchResult(SerdeJsonSerializable):
    samples: list[np.ndarray]
    agent_id: int


@dataclass
class MatchInfo(SerdeJsonSerializable):
    agent_id: int


@dataclass
class MessageActorInitClient(SerdeJsonSerializable):
    n_processes: int


@dataclass
class MessageActorInitServer(SerdeJsonSerializable):
    config: RunConfig
    current_ckpt: Checkpoint
    snapshots: list[Checkpoint]
    matches: list[MatchInfo]


@dataclass
class MessageLeanerInitServer(SerdeJsonSerializable):
    config: RunConfig
    init_ckpt: Checkpoint


@dataclass
class MessageMatchResult(SerdeJsonSerializable):
    result: MatchResult
    step: int


@dataclass
class MessageNextMatch(SerdeJsonSerializable):
    next_match: MatchInfo
    ckpt: Checkpoint | None


@dataclass
class MessageLearningRequest(SerdeJsonSerializable):
    minibatch: np.ndarray


@dataclass
class Losses(SerdeJsonSerializable):
    loss: float
    loss_policy: float
    loss_value: float
    loss_color: float


@dataclass
class MessageLearningResult(SerdeJsonSerializable):
    losses: Losses
    ckpt: Checkpoint
