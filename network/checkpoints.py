import os
import base64
import collections
import json
import dataclasses
import glob

import numpy as np
import jax
import optax
from flax.core.frozen_dict import FrozenDict

from network.transformer import Transformer, TransformerWithCache

NAMEDTUPLE_FLAG = "__namedtuple__"
NDARRAY_FLAG = "__ndarray__"


def post_converted_from_json(obj):
    if obj is None:
        return None

    if isinstance(obj, dict):
        if NDARRAY_FLAG in obj and obj[NDARRAY_FLAG]:
            shape = obj['shape']
            dtype = obj['dtype']

            buffer_str: str = obj['bytes']
            buffer = base64.b64decode(buffer_str.encode('utf-8'))

            return np.frombuffer(buffer, dtype).reshape(shape)

        else:
            for k in obj.keys():
                obj[k] = post_converted_from_json(obj[k])

            if NAMEDTUPLE_FLAG in obj and obj[NAMEDTUPLE_FLAG]:
                namedtuple_type = collections.namedtuple('DecodedNamedTuple', obj.keys())
                return namedtuple_type(**obj)

            return obj

    if isinstance(obj, tuple | list):
        for i in range(len(obj)):
            obj[i] = post_converted_from_json(obj[i])
        return obj

    return obj


def pre_converting_to_json(obj):
    if obj is None:
        return None

    if isinstance(obj, tuple) and hasattr(obj, '_asdict'):
        obj = obj._asdict()
        obj[NAMEDTUPLE_FLAG] = True

    if isinstance(obj, dict | FrozenDict):
        for k in obj.keys():
            obj[k] = pre_converting_to_json(obj[k])
        return obj

    if isinstance(obj, tuple | list):
        for i in range(len(obj)):
            obj[i] = pre_converting_to_json(obj[i])
        return obj

    if isinstance(obj, jax.Array):
        obj = np.asarray(obj)

    if isinstance(obj, np.ndarray):
        return {
            NDARRAY_FLAG: True,
            'shape': obj.shape,
            'dtype': str(obj.dtype),
            'bytes': base64.b64encode(obj.tobytes()).decode('utf-8')
        }

    return str(obj)


@dataclasses.dataclass
class Checkpoint:
    step: int
    params: FrozenDict
    model: Transformer
    opt_state: optax.OptState = None

    def to_dict(self):
        model = dataclasses.asdict(self.model)
        del model['parent']
        del model['name']

        return {
            'step': self.step,
            'params': self.params,
            'model': model,
            'opt_state': self.opt_state
        }

    @classmethod
    def from_dict(cls, ckpt: dict, is_caching_model: bool) -> 'Checkpoint':
        if not is_caching_model:
            model = Transformer(**ckpt['model'])
        else:
            model = TransformerWithCache(**ckpt['model'])

        step = ckpt['step']
        params = ckpt['params']
        opt_state = ckpt['opt_state']

        return Checkpoint(step, params, model, opt_state)

    def to_json(self):
        ckpt = self.to_dict()
        ckpt = pre_converting_to_json(ckpt)

        return json.dumps(ckpt)

    @classmethod
    def from_json(cls, json_str: str, is_caching_model: bool) -> 'Checkpoint':
        ckpt = json.loads(json_str)
        ckpt = post_converted_from_json(ckpt)

        return Checkpoint.from_dict(ckpt, is_caching_model)


@dataclasses.dataclass
class CheckpointManagerOptions:
    max_to_keep: int
    keep_period: int


class CheckpointManager:
    def __init__(self, path: str, options: CheckpointManagerOptions = None) -> None:
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

    def save(self, ckpt: Checkpoint):
        if ckpt.step % self.options.keep_period != 0:
            return

        with open(self.get_path(ckpt.step), mode='w') as f:
            f.write(ckpt.to_json())

        paths = self.get_paths()

        if len(paths) < self.options.max_to_keep:
            for i in range(len(paths) - self.options.max_to_keep):
                os.remove(paths[i])

    def load(self, step: int, is_caching_model: bool) -> Checkpoint:
        path = self.get_path(step)
        if not os.path.exists(path):
            FileNotFoundError(f"Checkpoint is not found in {path}")

        with open(path, mode='r') as f:
            json_str = f.read()

        return Checkpoint.from_json(json_str, is_caching_model)


def main():
    import os
    import orbax.checkpoint as ocp
    import time
    step = 8

    checkpointer = ocp.PyTreeCheckpointer()
    c_old = ocp.CheckpointManager(
        os.path.abspath('./data/checkpoints/rmt_4_256_4'),
        checkpointer
    )

    ckpt_old = c_old.restore(step)

    ckpt = Checkpoint(step, ckpt_old['params'], Transformer(**ckpt_old['model']))
    manager = CheckpointManager('./data/checkpoints/test-3')

    start = time.perf_counter()
    manager.save(ckpt)
    print(time.perf_counter() - start)

    start = time.perf_counter()
    manager.load(8, True)
    print(time.perf_counter() - start)


if __name__ == "__main__":
    main()
