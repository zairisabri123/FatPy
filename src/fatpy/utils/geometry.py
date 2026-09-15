r"""Geometric primitives for load path analysis.

Overview:
    This module provides the minimum enclosing ball (also called the smallest
    circumscribed ball) of a finite set of points in an arbitrary number of
    dimensions. It is the shared computational kernel behind the minimum
    circumscribed circle of a 2D shear stress path and the minimum
    circumscribed hyperball of a 5D deviatoric stress path.

Why a dedicated solver:
    Taking half of the longest distance between two points of the path - the
    longest chord - is **not** the minimum enclosing ball. Papadopoulos (1994,
    1997) showed that triads of points must be considered as well: for an
    equilateral triangle of side $s$ the longest chord gives $s/2$, whereas the
    true circumscribed radius is $s/\sqrt{3} \approx 0.577\,s$, an
    underestimation of 13.4 %.

    The algorithm implemented here handles pairs, triads and their higher
    dimensional equivalents in a single formulation: the support set of the
    optimal ball holds at most ``n_dim + 1`` points and is found by pivoting.

Input Shape Convention:
    - Points are given as an array of shape (n_points, n_dim). The solver
      itself is not batched; batching over load cases or material planes is
      handled by the calling modules.
"""

import itertools
import warnings

import numpy as np
from numpy.typing import NDArray

#: Relative tolerance used to decide whether a point lies inside the ball.
DEFAULT_TOLERANCE = 1e-12

#: Safety bound on the number of pivoting steps before falling back.
MAX_PIVOT_ITERATIONS = 200


def _check_points(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate the point set and return it as a float array.

    Args:
        points: Array of shape (n_points, n_dim).

    Returns:
        The validated array with dtype float64.

    Raises:
        ValueError: If the array is not two-dimensional, is empty, or holds
            non-finite values.
    """
    array = np.asarray(points, dtype=np.float64)

    if array.ndim != 2:
        raise ValueError(
            "Point set must be two-dimensional with shape (n_points, n_dim); "
            f"got shape {array.shape}."
        )
    if array.shape[0] < 1:
        raise ValueError("Point set must contain at least one point.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Point set must contain only finite values.")

    return array


def _ball_through_points(
    support: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    r"""Find the ball whose boundary passes through every support point.

    The centre is constrained to the affine hull of the support points, which
    reduces the problem to a small linear system. Writing the centre as

    $$ \mathbf{c} = \mathbf{p}_0 + \sum_{i \ge 1} u_i (\mathbf{p}_i -
    \mathbf{p}_0) $$

    and imposing equal distances to all support points gives

    $$ \mathbf{A}\mathbf{A}^{T}\mathbf{u} = \mathbf{b}, \qquad
    b_i = \tfrac{1}{2}\lVert \mathbf{p}_i - \mathbf{p}_0 \rVert^2 $$

    with $\mathbf{A} = [\mathbf{p}_i - \mathbf{p}_0]_{i \ge 1}$.

    Args:
        support: Array of shape (k, n_dim) with k >= 1 support points.

    Returns:
        A tuple ``(radius, center)``.
    """
    if support.shape[0] == 1:
        return 0.0, support[0].copy()

    offsets = support[1:] - support[0]
    gram = offsets @ offsets.T
    rhs = 0.5 * np.sum(offsets * offsets, axis=1)

    # lstsq tolerates the rank deficiency produced by affinely dependent
    # support points, which subset enumeration inevitably produces.
    weights = np.linalg.lstsq(gram, rhs, rcond=None)[0]

    center = support[0] + offsets.T @ weights
    radius = float(np.linalg.norm(support[0] - center))

    return radius, center


def _min_enclosing_ball_small(
    points: NDArray[np.float64], tol: float
) -> tuple[float, NDArray[np.float64], list[int]]:
    """Solve the minimum enclosing ball of a small point set exactly.

    Every non-empty subset is tried as a candidate support set, smallest first.
    This is exponential in the number of points and is only ever called with at
    most ``n_dim + 2`` of them, so the cost stays bounded and the result is
    exact.

    Args:
        points: Array of shape (k, n_dim) with a small k.
        tol: Relative tolerance used for the enclosure test.

    Returns:
        A tuple ``(radius, center, support_indices)``.
    """
    n_points = points.shape[0]
    best_radius = float(np.inf)
    best_center = points[0].copy()
    best_support: list[int] = [0]

    for size in range(1, n_points + 1):
        for subset in itertools.combinations(range(n_points), size):
            radius, center = _ball_through_points(points[list(subset)])

            if radius >= best_radius:
                continue

            distances = np.linalg.norm(points - center, axis=1)
            if np.all(distances <= radius * (1.0 + tol) + tol):
                best_radius = radius
                best_center = center
                best_support = list(subset)

        # Once a support set of this size covers everything, no larger subset
        # can give a smaller radius.
        if np.isfinite(best_radius):
            break

    return best_radius, best_center, best_support


def _badoiu_clarkson(
    points: NDArray[np.float64], n_iterations: int = 5000
) -> tuple[float, NDArray[np.float64]]:
    """Approximate the minimum enclosing ball by the Badoiu-Clarkson iteration.

    The centre is moved towards the currently farthest point with a decreasing
    step ``1 / (i + 1)``. Convergence is guaranteed but slow; this routine only
    serves as a fallback when the pivoting solver exhausts its iteration
    budget. The radius is measured afterwards, so the ball provably encloses
    every point.

    Args:
        points: Array of shape (n_points, n_dim).
        n_iterations: Number of iterations to perform.

    Returns:
        A tuple ``(radius, center)``.
    """
    center = points.mean(axis=0)

    for iteration in range(1, n_iterations + 1):
        farthest = points[np.argmax(np.sum((points - center) ** 2, axis=1))]
        center = center + (farthest - center) / (iteration + 1)

    radius = float(np.max(np.linalg.norm(points - center, axis=1)))

    return radius, center


def min_enclosing_ball(
    points: NDArray[np.float64],
    tol: float = DEFAULT_TOLERANCE,
) -> tuple[float, NDArray[np.float64]]:
    r"""Calculate the minimum ball enclosing a set of points.

    The ball of smallest radius containing every input point is unique. Its
    support set holds at most ``n_dim + 1`` points, so in two dimensions the
    solution is defined either by a pair of points (their diameter) or by a
    triad of points (their circumcircle). Both cases arise naturally from the
    algorithm, without the caller having to distinguish them.

    ??? abstract "Math Equations"
        The problem solved is

        $$ \min_{\mathbf{c},\,r} \; r
        \quad \text{subject to} \quad
        \lVert \mathbf{p}_i - \mathbf{c} \rVert \le r \;\; \forall i $$

        which is convex and has a unique optimum.

    The scheme is a pivoting loop: the exact minimum enclosing ball of the
    current support set is found by subset enumeration - cheap, because the
    support set is bounded by ``n_dim + 1`` - then the farthest outside point
    is added and the support set is recomputed. The radius increases strictly
    at every step, so the loop terminates.

    Args:
        points: Array of shape (n_points, n_dim) holding the path points. Any
            number of dimensions is accepted; 2 and 5 are the cases used for
            shear paths and deviatoric paths respectively.
        tol: Relative tolerance used to decide whether a point already lies
            inside the current ball.

    Returns:
        A tuple ``(radius, center)`` where ``radius`` is a float and ``center``
        is an array of shape (n_dim,).

    Raises:
        ValueError: If the point set is not two-dimensional, is empty, or holds
            non-finite values.

    Example:
        ```python
        import numpy as np
        from fatpy.utils.geometry import min_enclosing_ball

        # Equilateral triangle of side 2
        points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3.0)]])
        radius, center = min_enclosing_ball(points)
        print(round(radius, 4))  # 1.1547, i.e. 2 / sqrt(3)
        ```
    """
    array = _check_points(points)

    if array.shape[0] == 1:
        return 0.0, array[0].copy()

    # Duplicate points carry no information and slow the pivoting down.
    unique = np.unique(array, axis=0)
    if unique.shape[0] == 1:
        return 0.0, unique[0].copy()

    # Seed with the point farthest from the centroid: deterministic, and it
    # makes the first pivot meaningful.
    centroid = unique.mean(axis=0)
    support = [int(np.argmax(np.sum((unique - centroid) ** 2, axis=1)))]

    radius = 0.0
    center = unique[support[0]].copy()

    for _ in range(MAX_PIVOT_ITERATIONS):
        distances = np.linalg.norm(unique - center, axis=1)
        farthest = int(np.argmax(distances))

        if distances[farthest] <= radius * (1.0 + tol) + tol:
            return radius, center

        if farthest in support:
            # The farthest point already belongs to the support set: the small
            # solver has converged as far as the conditioning allows.
            break

        candidates = [*support, farthest]
        radius, center, local = _min_enclosing_ball_small(unique[candidates], tol)
        support = [candidates[index] for index in local]

    fallback_radius, fallback_center = _badoiu_clarkson(unique)

    distances = np.linalg.norm(unique - center, axis=1)
    encloses = bool(np.all(distances <= radius * (1.0 + 1e-9) + 1e-9))
    if encloses and radius <= fallback_radius:
        return radius, center

    warnings.warn(
        "min_enclosing_ball did not certify the pivoting solution and fell "
        "back to an iterative approximation. The returned ball encloses all "
        "points but may not be the smallest one.",
        RuntimeWarning,
        stacklevel=2,
    )

    return fallback_radius, fallback_center
