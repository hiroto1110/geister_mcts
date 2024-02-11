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
    visibilize_node_graph: bool = False

    def replace(self, **args):
        return replace(self, **args)


@dataclass
class FloatRange:
    min: int
    max: int

    def interpolate(self, p: float):
        return self.min + (self.max - self.min) * p


@dataclass
class IntRange:
    min: int
    max: int

    def interpolate(self, p: float):
        return int(np.round(self.min + (self.max - self.min) * p, 0))


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
    visibilize_node_graph: bool = False

    def replace(self, **args):
        return replace(self, **args)

    def sample(self) -> SearchParameters:
        return SearchParameters(
            num_simulations=self.num_simulations.interpolate(np.random.random()),

            dirichlet_alpha=self.dirichlet_alpha.interpolate(np.random.random()),
            dirichlet_eps=self.dirichlet_eps.interpolate(np.random.random()),

            n_ply_to_apply_noise=self.n_ply_to_apply_noise.interpolate(np.random.random()),
            max_duplicates=self.max_duplicates.interpolate(np.random.random()),

            c_init=self.c_init.interpolate(np.random.random()),
            c_base=self.c_base.interpolate(np.random.random()),

            depth_search_checkmate_root=self.depth_search_checkmate_root.interpolate(np.random.random()),
            depth_search_checkmate_leaf=self.depth_search_checkmate_leaf.interpolate(np.random.random()),

            visibilize_node_graph=self.visibilize_node_graph
        )
