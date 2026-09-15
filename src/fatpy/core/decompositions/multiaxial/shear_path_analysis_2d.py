"""Shear Path Analysis 2D methods of multiaxial decompositions.

These methods are designed to evaluate the mean shear stress and shear stress amplitude
acting on a specified material plane, typically used in multiaxial fatigue analysis. By
projecting the stress tensor onto the plane of interest, they provide key parameters
for assessing crack initiation risks and fatigue life under complex loading paths.
"""
import numpy as np
from numpy.typing import NDArray

from fatpy.utils.geometry import DEFAULT_TOLERANCE, min_enclosing_ball

#: Number of in-plane shear components of a resolved stress path.
SHEAR_COMPONENTS_COUNT = 2


def min_circumscribed_circle(
    shear_path: NDArray[np.float64],
    tol: float = DEFAULT_TOLERANCE,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Calculate the minimum circle circumscribed to a 2D shear stress path.

    During a load cycle the shear traction resolved on a material plane traces
    a closed path in the plane. The radius of the smallest circle enclosing
    that path is the shear stress amplitude $C_a$ used by critical plane
    criteria, and the norm of its centre is the mean shear stress $C_m$.

    ??? abstract "Math Equations"
        $$ C_a = \min_{\mathbf{c}} \; \max_{i} \;
        \lVert \boldsymbol{\tau}_i - \mathbf{c} \rVert
        \qquad
        C_m = \lVert \mathbf{c}^{*} \rVert $$

        where $\boldsymbol{\tau}_i$ are the sampled shear vectors and
        $\mathbf{c}^{*}$ is the optimal centre.

    Warning:
        Half of the longest chord of the path is **not** this radius.
        Papadopoulos (1994, 1997) showed that triads of path points have to be
        examined as well; for an equilateral triangular path the longest chord
        underestimates the radius by 13.4 %.

    Note:
        For a single-harmonic load the path is an ellipse and the radius equals
        its **major semi-axis** $a_1$. This differs from the frequently used
        expression $\sqrt{a_1^2 + a_2^2}$, which is larger whenever the loading
        is non-proportional and corresponds to a circumscribed ellipse rather
        than a circumscribed circle.

    Args:
        shear_path: Array of shape (..., n_steps, 2). The last dimension holds
            the two in-plane shear components [MPa], the one before it the
            time steps of the cycle. Leading dimensions are preserved and may
            index load cases or candidate planes.
        tol: Relative tolerance used to decide whether a path point already
            lies inside the current circle.

    Returns:
        A tuple ``(radius, center)``. ``radius`` has shape (...) and holds the
        shear stress amplitude [MPa]; ``center`` has shape (..., 2) and holds
        the centre of the circle [MPa]. The mean shear stress is the Euclidean
        norm of ``center``.

    Raises:
        ValueError: If the last dimension is not of size 2, if the path holds
            fewer than one point, or if it holds non-finite values.

    Example:
        ```python
        import numpy as np
        from fatpy.core.decompositions.multiaxial.shear_path_analysis_2d import (
            min_circumscribed_circle,
        )

        angle = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        path = np.stack([100.0 * np.cos(angle), 100.0 * np.sin(angle)], axis=-1)
        radius, center = min_circumscribed_circle(path)
        print(round(float(radius), 3))  # 100.0
        ```
    """
    array = np.asarray(shear_path, dtype=np.float64)

    if array.ndim < 2 or array.shape[-1] != SHEAR_COMPONENTS_COUNT:
        raise ValueError(
            f"Shear path must have shape (..., n_steps, 2); got shape {array.shape}."
        )
    if array.shape[-2] < 1:
        raise ValueError("Shear path must contain at least one point.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Shear path must contain only finite values.")

    batch_shape = array.shape[:-2]
    flat = array.reshape((-1, array.shape[-2], SHEAR_COMPONENTS_COUNT))

    radii = np.empty(flat.shape[0], dtype=np.float64)
    centers = np.empty((flat.shape[0], SHEAR_COMPONENTS_COUNT), dtype=np.float64)

    for index, path in enumerate(flat):
        radii[index], centers[index] = min_enclosing_ball(path, tol=tol)

    return (
        radii.reshape(batch_shape),
        centers.reshape(batch_shape + (SHEAR_COMPONENTS_COUNT,)),
    )
