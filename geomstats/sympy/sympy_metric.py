# -*- coding: utf-8 -*-
"""Symbolic pullback metric helpers built on top of geomstats."""

import os

os.environ["GEOMSTATS_BACKEND"] = "autograd"

import autograd.numpy as anp
import geomstats.backend as gs
import numpy as np
import sympy as sp
from geomstats.geometry.pullback_metric import PullbackMetric


class SymbolicMetricBase:
    def __init__(self, func_map, base_vars, diff_vars):
        self.dim = len(base_vars)
        self.func_map = func_map
        self.base_vars = base_vars
        self.diff_vars = diff_vars
        self._del_operator = None
        self._g = None
        self.g_matrix = None
        self.g_matrix_inv = None
        self._jacobian_g_matrix = None
        self._christoffels = None
        self._christoffels_sympy = None
        self._jacobian_christoffels = None
        self.linear_integrand = None
        self.start_vars = None
        self.end_vars = None
        self.t = None
        self.intgrl_step_size = 1.0 / 100.0

    def _validate_base_point(self, base_point):
        assert len(base_point) == len(self.base_vars)

    def _base_substitutions(self, base_point):
        self._validate_base_point(base_point)
        return [pair for pair in zip(self.base_vars, base_point)]

    def _ensure_metric_matrix(self):
        if self.g_matrix is None:
            self._create_induced_metric()


class SymbolicMetricMatrixMixin(SymbolicMetricBase):
    def _create_jacobian_g_matrix(self):
        self._ensure_metric_matrix()
        self._jacobian_g_matrix = {}
        for rw in range(self.dim):
            self._jacobian_g_matrix[rw] = self.g_matrix.row(rw).jacobian(self.base_vars)

    def _derive_metric(self):
        g = (
            sum(
                [
                    sp.diff(self.func_map[0], pair[0]) * pair[1]
                    for pair in zip(self.base_vars, self.diff_vars)
                ]
            )
        ) ** 2

        for entry in self.func_map[1:]:
            g += (
                sum(
                    [
                        sp.diff(entry, pair[0]) * pair[1]
                        for pair in zip(self.base_vars, self.diff_vars)
                    ]
                )
            ) ** 2
        self._g = sp.expand(g)

    def _create_induced_metric(self):
        square = lambda idx1: self._g.subs(
            [
                (diff_var, float(idx == idx1))
                for idx, diff_var in enumerate(self.diff_vars)
            ]
        )

        def off_diag(idx1, idx2):
            diff_subs = []
            for idx, diff_var in enumerate(self.diff_vars):
                if idx in (idx1, idx2):
                    diff_subs.append((diff_var**2, 0.0))
                    diff_subs.append((diff_var, 1.0))
                else:
                    diff_subs.append((diff_var, 0.0))
            return self._g.subs(diff_subs)

        if self._g is None:
            self._derive_metric()

        self.g_matrix = sp.Matrix(self.dim, self.dim, [0] * self.dim**2)
        for idx1 in range(self.dim):
            self.g_matrix[idx1, idx1] = square(idx1)
            for idx2 in range(idx1 + 1, self.dim):
                func = off_diag(idx1, idx2)
                self.g_matrix[idx1, idx2] = func
                self.g_matrix[idx2, idx1] = func

    def cometric_matrix(self, base_point=None):
        self._ensure_metric_matrix()
        if self.g_matrix_inv is None:
            self.g_matrix_inv = self.g_matrix.inv()

        calc_var = self._base_substitutions(base_point)
        return gs.array(self.g_matrix_inv.subs(calc_var))

    def metric_matrix(self, base_point=None):
        self._ensure_metric_matrix()
        calc_var = self._base_substitutions(base_point)
        return gs.array(self.g_matrix.subs(calc_var))

    def inner_product_derivative_matrix(self, base_point=None):
        if self._jacobian_g_matrix is None:
            self._create_jacobian_g_matrix()

        calc_var = self._base_substitutions(base_point)
        jac_vec = gs.zeros((self.dim, self.dim, self.dim))

        for rw, expr in self._jacobian_g_matrix.items():
            jac_vec[rw] = gs.array(expr.subs(calc_var))

        return jac_vec


class SymbolicMetricConnectionMixin(SymbolicMetricMatrixMixin):
    def _create_christoffels(self):
        self._ensure_metric_matrix()

        if self._jacobian_g_matrix is None:
            self._create_jacobian_g_matrix()

        if self.g_matrix_inv is None:
            self.g_matrix_inv = self.g_matrix.inv()

        gm = np.array(self.g_matrix_inv)
        jgm = np.empty((self.dim, self.dim, self.dim), dtype=object)
        for indx, expr in self._jacobian_g_matrix.items():
            jgm[indx] = np.array(expr)

        term_1 = gs.einsum("...lk,...jli->...kij", gm, jgm)
        term_2 = gs.einsum("...lk,...lij->...kij", gm, jgm)
        term_3 = -gs.einsum("...lk,...ijl->...kij", gm, jgm)
        christoffel_array = 0.5 * (term_1 + term_2 + term_3)

        self._christoffels_sympy = {}
        self._christoffels = {}
        for indx in range(self.dim):
            self._christoffels[indx] = sp.lambdify(
                np.array(self.base_vars),
                sp.Matrix(christoffel_array[indx]),
                modules={"Array": anp.array, "ImmutableDenseMatrix": anp.array},
            )
            self._christoffels_sympy[indx] = sp.Matrix(christoffel_array[indx])

    def christoffels(self, base_point=None):
        if self._christoffels is None:
            self._create_christoffels()
        if type(base_point) == anp.numpy_boxes.ArrayBox:
            base_point = gs.array(base_point._value)

        self._validate_base_point(base_point)
        christoffels_calc = gs.zeros((self.dim, self.dim, self.dim))
        for indx, expr in self._christoffels.items():
            christoffels_calc[indx] = gs.array(expr(*base_point))

        return christoffels_calc

    def _create_jacobian_christoffels(self):
        if self._christoffels is None:
            self._create_christoffels()

        self._jacobian_christoffels = {}
        for indx, expr in self._christoffels_sympy.items():
            for rw in range(self.dim):
                self._jacobian_christoffels[(indx, rw)] = expr.row(rw).jacobian(
                    self.base_vars
                )

    def jacobian_christoffels(self, base_point=None):
        if self._jacobian_christoffels is None:
            self._create_jacobian_christoffels()

        calc_var = self._base_substitutions(base_point)
        calc = gs.zeros((self.dim, self.dim, self.dim, self.dim))
        jacobian_christoffels = self._jacobian_christoffels
        for indx, expr in jacobian_christoffels.items():
            calc[indx[0]][indx[1]] = gs.array(expr.subs(calc_var))

        return calc

    def riemann_tensor(self, base_point=None):
        if len(self._space.shape) > 1:
            raise NotImplementedError(
                "Riemann tensor not implemented for manifolds with points of ndim > 1."
            )
        self._validate_base_point(base_point)

        if self._jacobian_g_matrix is None:
            self._create_jacobian_g_matrix()

        if self._jacobian_christoffels is None:
            self._create_jacobian_christoffels()

        _christoffels = self.christoffels(base_point)
        _jacobian_christoffels = self.jacobian_christoffels(base_point)

        prod_christoffels = gs.einsum(
            "...ijk,...klm->...ijlm", _christoffels, _christoffels
        )
        riemann_curvature = (
            gs.einsum("...ijlm->...lmji", _jacobian_christoffels)
            - gs.einsum("...ijlm->...ljmi", _jacobian_christoffels)
            + gs.einsum("...ijlm->...mjli", prod_christoffels)
            - gs.einsum("...ijlm->...lmji", prod_christoffels)
        )

        return riemann_curvature


class SymbolicMetricPathMixin(SymbolicMetricMatrixMixin):
    def arcLength(
        self,
        start_point=None,
        end_point=None,
        path=None,
        time_var=None,
        time_interval=None,
    ):
        if path is None:
            return self.dist_linear(start_point, end_point)

        self._ensure_metric_matrix()
        self.t = time_var
        subs_vars = [pair for pair in zip(self.base_vars, path)]
        ft = sp.Matrix(self.func_map).subs(subs_vars)
        dft = sp.diff(ft, self.t)
        gt = self.g_matrix.subs(subs_vars)
        integrand = (dft.T * gt * dft)[0]

        grid = np.arange(
            time_interval[0] + self.intgrl_step_size,
            time_interval[1] + self.intgrl_step_size,
            self.intgrl_step_size,
        )
        intgrl = 0.0
        for i in grid:
            pt = float(integrand.subs([(self.t, i)]))
            if i < 0.0:
                print(f"negative value at step {i}")
            else:
                intgrl += np.sqrt(pt) * self.intgrl_step_size

        return intgrl

    def dist_linear(self, start_point, end_point):
        if self.linear_integrand is None:
            self._create_dist_linear()

        integrand = self.linear_integrand.subs(
            [pair for pair in zip(self.start_vars, start_point)]
            + [pair for pair in zip(self.end_vars, end_point)]
        )

        grid = np.arange(
            self.intgrl_step_size,
            1.0 + self.intgrl_step_size,
            self.intgrl_step_size,
        )
        intgrl = 0.0
        for i in grid:
            pt = float(integrand.subs([(self.t, i)]))
            if i < 0.0:
                print(f"negative value at step {i}")
            else:
                intgrl += np.sqrt(pt) * self.intgrl_step_size

        return intgrl

    def _create_dist_linear(self):
        self._ensure_metric_matrix()
        self.start_vars = sp.symbols(f"s:{self.dim}", real=True)
        self.end_vars = sp.symbols(f"e:{self.dim}", real=True)
        self.t = sp.symbols("t", real=True)

        path = [
            (1.0 - self.t) * pair[0] + self.t * pair[1]
            for pair in zip(self.start_vars, self.end_vars)
        ]

        subs_vars = [pair for pair in zip(self.base_vars, path)]

        ft = sp.Matrix(self.func_map).subs(subs_vars)
        dft = sp.diff(ft, self.t)
        gt = self.g_matrix.subs(subs_vars)
        self.linear_integrand = (dft.T * gt * dft)[0]


class SymbolicMetric(SymbolicMetricConnectionMixin, SymbolicMetricPathMixin, PullbackMetric):
    def __init__(self, space, func_map, base_vars, diff_vars):
        PullbackMetric.__init__(self, space=space)
        SymbolicMetricBase.__init__(self, func_map, base_vars, diff_vars)

    def dist(self, point_a, point_b):
        sq_dist = float(self.squared_dist(point_a, point_b))
        return gs.sqrt(sq_dist)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        inner_prod_mat = self.metric_matrix(base_point)
        aux = gs.einsum("...j,...jk->...k", tangent_vec_a, inner_prod_mat)
        return gs.dot(aux, tangent_vec_b)

    def squared_norm(self, vector, base_point=None):
        return self.inner_product(vector, vector, base_point)

    def norm(self, vector, base_point=None):
        sq_norm = self.squared_norm(vector, base_point)
        return gs.sqrt(sq_norm)


__all__ = [
    "SymbolicMetricBase",
    "SymbolicMetricMatrixMixin",
    "SymbolicMetricConnectionMixin",
    "SymbolicMetricPathMixin",
    "SymbolicMetric",
]