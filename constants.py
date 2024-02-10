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
    c_base: int = 19652
    depth_search_checkmate_root: int = 7
    depth_search_checkmate_leaf: int = 4
    visibilize_node_graph: bool = False

    def replace(self, **args):
        return replace(self, **args)

    @classmethod
    def interpolate(cls, p1: "SearchParameters", p2: "SearchParameters", p: float) -> "SearchParameters":
        def interpolate_f(a1, a2):
            return a1 + (a2 - a1) * p

        def interpolate_i(a1, a2):
            return int(np.round(interpolate_f(a1, a2), 0))

        return SearchParameters(
            num_simulations=interpolate_i(p1.num_simulations, p2.num_simulations),
            dirichlet_alpha=interpolate_f(p1.dirichlet_alpha, p2.dirichlet_alpha),
            dirichlet_eps=interpolate_f(p1.dirichlet_eps, p2.dirichlet_eps),
            n_ply_to_apply_noise=interpolate_i(p1.n_ply_to_apply_noise, p2.n_ply_to_apply_noise),
            max_duplicates=interpolate_i(p1.max_duplicates, p2.max_duplicates),
            c_init=interpolate_f(p1.c_init, p2.c_init),
            c_base=interpolate_f(p1.c_base, p2.c_base),
            depth_search_checkmate_root=interpolate_i(p1.depth_search_checkmate_root, p2.depth_search_checkmate_root),
            depth_search_checkmate_leaf=interpolate_i(p1.depth_search_checkmate_leaf, p2.depth_search_checkmate_leaf),
            visibilize_node_graph=p1.visibilize_node_graph and p2.visibilize_node_graph,
        )
