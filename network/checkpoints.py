import os
import base64
import json
import dataclasses
import glob

import serde

import numpy as np
from jaxlib.xla_extension import ArrayImpl
import jax
from flax.core.frozen_dict import FrozenDict

from distributed.communication import SerdeJsonSerializable
from network.transformer import TransformerConfig


NetworkConfig = TransformerConfig

NAMEDTUPLE_FLAG = "__namedtuple__"
NDARRAY_SHAPE = "__ndarray_shape__"
NDARRAY_DTYPE = "__ndarray_dtype__"
NDARRAY_BYTES = "__ndarray_bytes__"


def post_converted_from_json(obj):
    if obj is None:
        return None

    if isinstance(obj, dict):
        if (NDARRAY_SHAPE in obj) and (NDARRAY_DTYPE in obj) and (NDARRAY_BYTES in obj):
            shape = obj[NDARRAY_SHAPE]
            dtype = obj[NDARRAY_DTYPE]

            if len(shape) > 0 and isinstance(shape[0], str):
                shape = [int(s) for s in shape]

            buffer_str: str = obj[NDARRAY_BYTES]
            buffer = base64.b64decode(buffer_str.encode('utf-8'))

            return np.frombuffer(buffer, dtype).reshape(shape)

        else:
            for k in obj.keys():
                obj[k] = post_converted_from_json(obj[k])
            return obj

    if isinstance(obj, tuple | list):
        return [post_converted_from_json(o_i) for o_i in obj]

    return obj


def pre_converting_to_json(obj):
    if obj is None:
        return None

    if isinstance(obj, dict | FrozenDict):
        return {k: pre_converting_to_json(obj[k]) for k in obj.keys()}

    if isinstance(obj, tuple | list):
        return [pre_converting_to_json(o_i) for o_i in obj]

    if isinstance(obj, jax.Array | ArrayImpl | jax.numpy.ndarray):
        obj = np.asarray(obj)

    if isinstance(obj, np.ndarray):
        return {
            NDARRAY_SHAPE: obj.shape,
            NDARRAY_DTYPE: str(obj.dtype),
            NDARRAY_BYTES: base64.b64encode(obj.tobytes()).decode('utf-8')
        }

    return str(obj)


def deserialize_params(d):
    # print("deserialize_params", type(d))
    if isinstance(d, str):
        d = json.loads(d)

    return post_converted_from_json(d)


@dataclasses.dataclass
class Checkpoint(SerdeJsonSerializable):
    step: int
    model: NetworkConfig
    params: FrozenDict | dict = serde.field(serializer=pre_converting_to_json, deserializer=deserialize_params)

    def __post_init__(self):
        self.step = int(self.step)


@dataclasses.dataclass
class CheckpointManagerOptions:
    max_to_keep: int
    keep_period: int


@dataclasses.dataclass
class CheckpointManager:
    def __init__(self, path: str, options: CheckpointManagerOptions | None = None) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

        self.path = path

        if options is not None:
            self.options = options
        else:
            self.options = CheckpointManagerOptions(
                max_to_keep=2**30,
                keep_period=1,
            )

    def get_path(self, step: int) -> str:
        return f'{self.path}/{step}.json'

    def get_paths(self) -> list[str]:
        return glob.glob(self.path + '/*.json')

    def get_steps(self) -> list[int]:
        paths = self.get_paths()
        return [int(os.path.splitext(os.path.basename(path))[0]) for path in paths]

    def lastest_step(self) -> int:
        return max(self.get_steps())

    def save(self, ckpt: Checkpoint):
        print("save", self.get_path(ckpt.step))
        with open(self.get_path(ckpt.step), mode='w') as f:
            f.write(ckpt.to_json())

        steps = self.get_steps()

        for step in steps:
            if ckpt.step - step <= self.options.max_to_keep:
                continue

            if step % self.options.keep_period == 0:
                continue

            os.remove(self.get_path(step))

    def load(self, step: int) -> Checkpoint:
        if step == -1:
            step = self.lastest_step()

        path = self.get_path(step)

        if not os.path.exists(path):
            FileNotFoundError(f"Checkpoint is not found in {path}")

        with open(path, mode='r') as f:
            json_str = f.read()

        return Checkpoint.from_json(json_str)


def main():
    import time
    import optax

    adam = optax.adam(learning_rate=0.02)

    ckpt = Checkpoint.from_json_file('./data/checkpoints/test-3/8.json')

    state = adam.init(ckpt.params)
    ckpt = Checkpoint(ckpt.step, ckpt.model, ckpt.params, state)

    print(state)

    start = time.perf_counter()
    s = ckpt.to_json()
    print(time.perf_counter() - start)

    start = time.perf_counter()
    msg_ = Checkpoint.from_json(s)
    print(msg_)
    print(time.perf_counter() - start)


if __name__ == "__main__":
    main()
