# Geomstats_Sympy

## What the class does

`SymbolicMetric` is a custom metric class that combines `sympy` with `geomstats`.

It takes a symbolic map `func_map` from local coordinates to an ambient Euclidean space, differentiates that map symbolically, and builds the induced pullback metric matrix from those derivatives. After the symbolic expressions are built, the class evaluates them numerically through `geomstats` arrays so the metric can be used like any other `geomstats` metric.

## How it uses geomstats

`SymbolicMetric` inherits from `PullbackMetric` in `geomstats.geometry.pullback_metric`. That inheritance is the key integration point:

1. `geomstats` provides the metric interface and higher-level Riemannian operations such as `squared_dist`, `scalar_curvature`, and `ricci_tensor`.
2. `SymbolicMetric` fills in the metric-specific pieces by computing the metric matrix, its derivatives, and Christoffel symbols from symbolic expressions.
3. The class uses `geomstats.backend` as `gs` for array creation, tensor contractions with `gs.einsum`, and basic linear algebra operations in the format expected by `geomstats`.
4. A manifold or vector space is equipped with this metric through `space.equip_with_metric(...)`, which makes the resulting object available as `space.metric`.

In practice, the flow is:

1. Define symbolic coordinate variables such as `x0, x1, ...`.
2. Define symbolic differential variables such as `dx0, dx1, ...`.
3. Define a symbolic map `func_map` describing the embedding or transformation.
4. Create a `geomstats` space, usually `Euclidean(dim=..., equip=False)`.
5. Attach a metric class derived from `SymbolicMetric` with `space.equip_with_metric(...)`.
6. Call standard `geomstats` metric methods like `metric_matrix`, `dist`, `scalar_curvature`, or `ricci_tensor`.

## Main methods

## Class structure

The module now separates responsibilities into helper classes that can be inherited independently:

`SymbolicMetricBase`

Stores shared symbolic state, cached tensors, and helper validation methods.

`SymbolicMetricMatrixMixin`

Builds the symbolic metric, the metric matrix, the inverse metric, and metric derivatives.

`SymbolicMetricConnectionMixin`

Builds Christoffel symbols, their Jacobians, and the Riemann tensor.

`SymbolicMetricPathMixin`

Handles linear-path distance approximation and symbolic arc-length integration.

`SymbolicMetric`

Combines those inherited pieces and still keeps core metric-facing behavior such as `dist`, `inner_product`, `squared_norm`, and `norm`.

`metric_matrix(base_point)`

Builds and evaluates the induced metric tensor at a point.

`cometric_matrix(base_point)`

Returns the inverse metric matrix.

`christoffels(base_point)`

Computes the Christoffel symbols from the metric derivatives.

`riemann_tensor(base_point)`

Builds the Riemann curvature tensor from the Christoffel symbols and their derivatives.

`dist(point_a, point_b)`

Uses the inherited `geomstats` distance machinery, then returns the square root.

`dist_linear(point_a, point_b)`

Approximates distance along the straight-line path between two parameter points by numerically integrating the induced metric.

`arcLength(...)`

Computes the length of a supplied symbolic path, or falls back to `dist_linear` if no path is provided.

## Relationship to Example.py

The examples in [Example.py](Example.py) show the same integration pattern in four variations:

1. Identity map, which reproduces the Euclidean metric.
2. A symbolic map into the simplex.
3. A comparison against geomstats' built-in Fisher information metric for multinomial distributions.
4. A logistic map into an `n`-dimensional cube.

Those examples demonstrate that the class is not a replacement for `geomstats`; it is an extension point that lets you define a metric symbolically and still use the standard `geomstats` metric API.