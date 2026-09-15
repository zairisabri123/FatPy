"""Load Path Analysis 5D methods of multiaxial decompositions.

These methods compute the amplitude (J_2,a) and mean (J_2,m) values of the second
invariant of the deviatoric stress tensor in five-dimensional deviatoric space. This
evaluation is particularly useful in advanced multiaxial fatigue models, where accurate
representation of cyclic loading paths and stress states in the deviatoric space is
critical for predicting material response.
"""
import numpy as np
from numpy.typing import NDArray
 
from fatpy.struct_mech.transformations import (
    ILYUSHIN_COMPONENTS_COUNT,
    tensor_to_ilyushin_space,
)
from fatpy.utils import voigt
from fatpy.utils.geometry import DEFAULT_TOLERANCE, min_enclosing_ball
 
 
def min_circumscribed_hyperball(
    stress_history_voigt: NDArray[np.float64],
    tol: float = DEFAULT_TOLERANCE,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Calculate the minimum hyperball circumscribed to a 5D stress path.
 
    The stress history is mapped into the five-dimensional deviatoric
    (Ilyushin) space, where the Euclidean norm equals $\sqrt{J_2}$. The radius
    of the smallest hyperball enclosing the resulting path is the amplitude
    $\sqrt{J_2}_{,a}$, and the norm of its centre is the mean value
    $\sqrt{J_2}_{,m}$. Both feed invariant-based criteria such as Crossland,
    Sines and Dang Van.
 
    ??? abstract "Math Equations"
        $$ \sqrt{J_2}_{,a} = \min_{\mathbf{c}} \; \max_{i} \;
        \lVert \mathbf{S}_i - \mathbf{c} \rVert
        \qquad
        \sqrt{J_2}_{,m} = \lVert \mathbf{c}^{*} \rVert $$
 
        with $\mathbf{S}_i$ the deviatoric stress states of the cycle expressed
        in Ilyushin space and $\mathbf{c}^{*}$ the optimal centre.
 
    Warning:
        Both returned quantities are amplitudes and means of $\sqrt{J_2}$, not
        of $J_2$ itself. This is the form that enters the Crossland criterion
        $\sqrt{J_2}_{,a} + \lambda\,\sigma_{H,max} \le \beta$. Squaring them
        is a common source of error.
 
    Note:
        The hyperball centre is discarded here because the component only
        exposes the two invariant scalars. Criteria that need the centre
        itself, such as the elastic shakedown state of Dang Van, can obtain it
        from `fatpy.utils.geometry.min_enclosing_ball` applied to the output of
        `fatpy.struct_mech.transformations.tensor_to_ilyushin_space`.
 
    Args:
        stress_history_voigt: Array of shape (..., n_steps, 6). The last
            dimension holds the Voigt stress components [MPa], the one before
            it the time steps of the cycle. Leading dimensions are preserved
            and may index nodes of a model.
        tol: Relative tolerance used to decide whether a path point already
            lies inside the current hyperball.
 
    Returns:
        A tuple ``(j2_amp, j2_mean)``, both of shape (...), holding the
        amplitude and the mean value of $\sqrt{J_2}$ [MPa].
 
    Raises:
        ValueError: If the last dimension is not of size 6, if the history
            holds fewer than one step, or if it holds non-finite values.
 
    Example:
        ```python
        import numpy as np
        from fatpy.core.decompositions.multiaxial.load_path_analysis_5d import (
            min_circumscribed_hyperball,
        )
 
        # Fully reversed torsion of 180 MPa
        time = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        history = np.zeros((360, 6))
        history[:, 5] = 180.0 * np.sin(time)
        j2_amp, j2_mean = min_circumscribed_hyperball(history)
        print(round(float(j2_amp), 2))  # 180.0
        ```
    """
    array = np.asarray(stress_history_voigt, dtype=np.float64)
    voigt.check_shape(array)
 
    if array.ndim < 2:
        raise ValueError(
            "Stress history must have shape (..., n_steps, 6); "
            f"got shape {array.shape}."
        )
    if array.shape[-2] < 1:
        raise ValueError("Stress history must contain at least one point.")
 
    deviatoric_path = tensor_to_ilyushin_space(array)
 
    batch_shape = deviatoric_path.shape[:-2]
    flat = deviatoric_path.reshape(
        (-1, deviatoric_path.shape[-2], ILYUSHIN_COMPONENTS_COUNT)
    )
 
    amplitudes = np.empty(flat.shape[0], dtype=np.float64)
    means = np.empty(flat.shape[0], dtype=np.float64)
 
    for index, path in enumerate(flat):
        radius, center = min_enclosing_ball(path, tol=tol)
        amplitudes[index] = radius
        means[index] = float(np.linalg.norm(center))
 
    return amplitudes.reshape(batch_shape), means.reshape(batch_shape)