# -*- coding: utf-8 -*-
"""Examples for SymbolicMetric.

Steps used in every example:
1. Define symbolic coordinate variables and differential variables.
2. Build a symbolic map `func_map` for the geometry of interest.
3. Create a geomstats Euclidean space without a default metric.
4. Equip that space with a metric derived from SymbolicMetric.
5. Evaluate metric quantities such as the metric matrix, distances, and curvature.
"""

from time import time

import geomstats.backend as gs
import numpy as np
import sympy as sp
from geomstats.geometry.euclidean import Euclidean
from geomstats.information_geometry.multinomial import MultinomialDistributions

from Geomstats_Sympy import SymbolicMetric


class GenMetric(SymbolicMetric):
    def __init__(self, space, func_map, base_vars, diff_vars):
        super().__init__(
            space=space,
            func_map=func_map,
            base_vars=base_vars,
            diff_vars=diff_vars,
        )


def build_space(dim, func_map, base_vars, diff_vars):
    """Step 3 and step 4: create a space and equip it with the symbolic metric."""
    space = Euclidean(dim=dim, equip=False)
    space.equip_with_metric(
        Metric=GenMetric,
        func_map=func_map,
        base_vars=base_vars,
        diff_vars=diff_vars,
    )
    return space


def l2_metric_example():
    """Identity map example: the pullback metric should match the Euclidean metric."""
    dim = 4
    base_vars = sp.symbols(f"x:{dim}", real=True)
    diff_vars = sp.symbols(f"dx:{dim}")
    func_map = [var for var in base_vars]

    space = build_space(dim, func_map, base_vars, diff_vars)

    base = np.array([1.7, 0.19, 0.8, 2.4])
    point_a = np.array([1.1, 0.39, 0.84, 2.4])
    point_b = np.array([1.3, 0.79, 0.8, 2.0])

    print("L2 metric matrix")
    print(space.metric.metric_matrix(base))
    print("L2 distances")
    print(
        space.metric.dist(point_a, point_b),
        space.metric.dist_linear(point_a, point_b),
        np.linalg.norm(point_a - point_b),
    )
    print("L2 curvature")
    print(space.metric.scalar_curvature(base))
    print(space.metric.ricci_tensor(base))


def simplex_pullback_example():
    """Softmax-like map into the simplex, then evaluate the induced pullback metric."""
    dim = 3
    base_vars = sp.symbols(f"x:{dim}", real=True)
    diff_vars = sp.symbols(f"dx:{dim}")
    exp_vars = [sp.exp(var) for var in base_vars]
    func_map = [exp_v / (sum(exp_vars) + 1.0) for exp_v in exp_vars[:-1]] + [
        1.0 / (sum(exp_vars) + 1.0)
    ]

    space = build_space(dim, func_map, base_vars, diff_vars)

    base = np.array([0.8, 0.8, 0.0])
    point_a = gs.array([1.1, 0.39, 0.0])
    point_b = gs.array([1.3, 0.8, 0.0])

    start = time()
    print("Simplex metric matrix")
    print(space.metric.metric_matrix(base))
    print("Simplex distances")
    print(space.metric.dist_linear(point_a, point_b), np.linalg.norm(point_a - point_b))
    print("Simplex curvature")
    print(space.metric.scalar_curvature(base))
    print(space.metric.ricci_tensor(base))
    print("Simplex runtime (minutes)")
    print(round((time() - start) / 60.0, 4))


def simplex_fisher_information_example():
    """Compare against geomstats' built-in Fisher information metric on multinomials."""

    def array_map(arr):
        exps = gs.hstack([gs.exp(arr), gs.array([1.0])])
        return exps / exps.sum()

    dim = 4
    md = MultinomialDistributions(dim=dim, n_draws=20)

    point_a = gs.array([1.1, 0.39, 0.84, 2.4])
    point_b = gs.array([1.3, 0.79, 0.8, 2.0])
    point_a_map = array_map(point_a)
    point_b_map = array_map(point_b)

    print("Multinomial Fisher distance")
    print(md.metric.dist(point_a_map, point_b_map), np.linalg.norm(point_a - point_b))


def unit_cube_example():
    """Logistic map into an n-dimensional cube scaled by pi."""
    dim = 4
    base_vars = sp.symbols(f"x:{dim}", real=True)
    diff_vars = sp.symbols(f"dx:{dim}")
    func_map = [gs.pi / (1.0 + sp.exp(-var)) for var in base_vars]

    space = build_space(dim, func_map, base_vars, diff_vars)

    base = np.array([1.1, 0.19, 0.8, 0.42])
    point_a = np.array([1.1, 0.39, 0.84, 2.4])
    point_b = np.array([1.3, 0.79, 0.8, 2.0])

    print("Unit cube metric matrix")
    print(space.metric.metric_matrix(base))
    print("Unit cube distances")
    print(space.metric.dist(point_a, point_b), np.linalg.norm(point_a - point_b))
    print("Unit cube curvature")
    print(space.metric.scalar_curvature(base))
    print(space.metric.ricci_tensor(base))


if __name__ == "__main__":
    l2_metric_example()
    simplex_pullback_example()
    simplex_fisher_information_example()
    unit_cube_example()