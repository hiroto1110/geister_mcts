from __future__ import annotations

from typing import Any

from jax import numpy as jnp
from flax.training import train_state


class TrainStateBase(train_state.TrainState):
    epoch: int
    dropout_rng: Any

    def train_step(
        self, x: jnp.ndarray, eval: bool
    ) -> tuple[TrainStateBase, jnp.ndarray, jnp.ndarray]:
        pass

    def get_head_names(self) -> list[str]:
        pass
