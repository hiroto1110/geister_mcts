from dataclasses import dataclass, replace
import numpy as np


@dataclass
class SearchParameters:
    num_simulations: int
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    n_ply_to_apply_noise: int = 20
    max_duplicates: int = 3
    c_init: float = 1.25
    c_base: int = 25
    depth_search_checkmate_root: int = 7
    depth_search_checkmate_leaf: int = 4
    value_weight: np.ndarray = np.array([-1, -1, -1, 0, 1, 1, 1])
    visibilize_node_graph: bool = False

    def replace(self, **args):
        return replace(self, **args)


@dataclass
class FloatRange:
    min: int
    max: int

    def sample(self):
        p = np.random.random()
        return self.min + (self.max - self.min) * p


@dataclass
class IntRange:
    min: int
    max: int

    def sample(self):
        p = np.random.random()
        return int(np.round(self.min + (self.max - self.min) * p, 0))


@dataclass
class VectorRange:
    min: np.ndarray
    max: np.ndarray

    def sample(self):
        p = np.random.random(size=self.min.shape)
        return self.min + (self.max - self.min) * p


@dataclass
class SearchParametersRange:
    num_simulations: IntRange
    dirichlet_alpha: FloatRange = FloatRange(0.3, 0.3)
    dirichlet_eps: FloatRange = FloatRange(0.25, 0.25)
    n_ply_to_apply_noise: IntRange = IntRange(20, 20)
    max_duplicates: IntRange = IntRange(3, 3)
    c_init: FloatRange = FloatRange(1.25, 1.25)
    c_base: IntRange = IntRange(25, 25)
    depth_search_checkmate_root: IntRange = IntRange(7, 7)
    depth_search_checkmate_leaf: IntRange = IntRange(4, 4)
    value_weight: VectorRange = VectorRange(np.array([-1, -1, -1, 0, 1, 1, 1]), np.array([-1, -1, -1, 0, 1, 1, 1]))
    visibilize_node_graph: bool = False

    def replace(self, **args):
        return replace(self, **args)

    def sample(self) -> SearchParameters:
        return SearchParameters(
            num_simulations=self.num_simulations.sample(),
            dirichlet_alpha=self.dirichlet_alpha.sample(),
            dirichlet_eps=self.dirichlet_eps.sample(),
            n_ply_to_apply_noise=self.n_ply_to_apply_noise.sample(),
            max_duplicates=self.max_duplicates.sample(),
            c_init=self.c_init.sample(),
            c_base=self.c_base.sample(),
            depth_search_checkmate_root=self.depth_search_checkmate_root.sample(),
            depth_search_checkmate_leaf=self.depth_search_checkmate_leaf.sample(),
            value_weight=self.value_weight.sample(),
            visibilize_node_graph=self.visibilize_node_graph
        )
