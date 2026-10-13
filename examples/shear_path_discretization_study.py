"""Discretization study of the minimum circumscribed circle of a shear path.

Each reference path, whose exact circumscribed radius is known, is sampled
with N steps per cycle (N from 8 to 1000). For every N the starting phase is
drawn at random many times, stratified over one step interval (a regular
grid of phases would resonate with the step counts and hide the error), and
the worst case is kept. The relative
error of the radius, (R_N - R_exact) / R_exact, is reported with its sign,
the order of convergence is fitted on a log-log scale, and the minimum
number of steps per cycle for 1 % and 0.1 % accuracy is given, together with
the error at 72 steps per cycle.

Reference paths (exact radius):
    - proportional C(t) = [A sin wt, B sin wt]: sqrt(A^2 + B^2)
    - the same with a constant offset: unchanged
    - circle of radius R: R
    - ellipse C(t) = [f cos + g sin, p cos + q sin]: major semi-axis
      sqrt(S/2 + sqrt((S/2)^2 - (f q - g p)^2)), S = f^2 + g^2 + p^2 + q^2

Run it from the repository root:

    python examples/shear_path_discretization_study.py [output.png]
"""

import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from fatpy.core.decompositions.multiaxial.shear_path_analysis_2d import (  # noqa: E402
    min_circumscribed_circle,
)

Path2D = Callable[[NDArray[np.float64]], NDArray[np.float64]]

A, B = 3.0, 4.0
OFFSET = np.array([120.0, -45.0])
RADIUS = 50.0
F, G, P, Q = 3.0, 1.0, 0.5, 2.0
S = F**2 + G**2 + P**2 + Q**2
ELLIPSE_MAJOR = float(np.sqrt(S / 2 + np.sqrt((S / 2) ** 2 - (F * Q - G * P) ** 2)))

PATHS: dict[str, tuple[Path2D, float]] = {
    "proportional": (
        lambda t: np.column_stack((A * np.sin(t), B * np.sin(t))),
        float(np.hypot(A, B)),
    ),
    "proportional + offset": (
        lambda t: np.column_stack((A * np.sin(t), B * np.sin(t))) + OFFSET,
        float(np.hypot(A, B)),
    ),
    "circle": (
        lambda t: RADIUS * np.column_stack((np.cos(t), np.sin(t))),
        RADIUS,
    ),
    "ellipse": (
        lambda t: np.column_stack(
            (F * np.cos(t) + G * np.sin(t), P * np.cos(t) + Q * np.sin(t))
        ),
        ELLIPSE_MAJOR,
    ),
}
N_OFFSETS = 64  # random starting phases per step count
SEED = 2026


def worst_error(
    path: Path2D, exact: float, n_steps: int, rng: np.random.Generator
) -> float:
    """Signed relative error with the largest magnitude over random phases.

    Shifting the start by a whole step gives the same set of points, so the
    phases are drawn within one step interval, stratified: one random phase
    in each of `N_OFFSETS` equal sub-intervals. They stay random (no regular
    grid that could resonate with the step count) but cover every relative
    offset. The worst phase found is then refined on 33 phases within one
    sub-interval around it, so that the worst case is not underestimated.
    """
    step = 2.0 * np.pi / n_steps
    base = np.arange(n_steps) * step
    width = step / N_OFFSETS

    def errors_at(phases: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array(
            [
                (min_circumscribed_circle(path(base + phase))[1] - exact) / exact
                for phase in phases
            ]
        )

    phases = (np.arange(N_OFFSETS) + rng.uniform(size=N_OFFSETS)) * width
    errors = errors_at(phases)
    worst = phases[np.argmax(np.abs(errors))]
    local = errors_at(worst + np.linspace(-width, width, 33))
    errors = np.concatenate((errors, local))
    return float(errors[np.argmax(np.abs(errors))])


def minimum_steps(
    steps: NDArray[np.int64], errors: NDArray[np.float64], level: float
) -> int:
    """Smallest N from which every larger scanned N meets the accuracy."""
    bad = np.flatnonzero(np.abs(errors) > level)
    return int(steps[0]) if bad.size == 0 else int(steps[bad[-1] + 1])


def main() -> None:
    """Run the study, print the summary and save the convergence plot."""
    figure_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("shear_path_discretization_study.png")
    )
    rng = np.random.default_rng(SEED)
    plot_steps = np.unique(np.round(np.geomspace(8, 1000, 40)).astype(np.int64))
    scan_steps = np.union1d(np.arange(8, 151, dtype=np.int64), plot_steps)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    print(f"Worst relative error, {N_OFFSETS} random phases per N + local refinement")
    print(
        f"{'path':<22}{'sign':>10}{'order':>8}{'N(1%)':>8}{'N(0.1%)':>9}{'err(72)':>12}"
    )
    for name, (path, exact) in PATHS.items():
        scan = np.array([worst_error(path, exact, n, rng) for n in scan_steps])
        errors = scan[np.searchsorted(scan_steps, plot_steps)]
        if np.all(np.abs(scan) < 1e-12):
            sign, order = "exact", "-"
        else:
            sign = "negative" if np.all(scan <= 1e-15) else "MIXED"
            used = np.abs(errors) > 1e-13
            slope = np.polyfit(
                np.log(plot_steps[used]), np.log(np.abs(errors[used])), 1
            )[0]
            order = f"{-slope:.2f}"
            ax.loglog(plot_steps, np.abs(errors), marker="o", markersize=3, label=name)
        error_72 = scan[np.searchsorted(scan_steps, 72)]
        print(
            f"{name:<22}{sign:>10}{order:>8}"
            f"{minimum_steps(scan_steps, scan, 1e-2):>8}"
            f"{minimum_steps(scan_steps, scan, 1e-3):>9}"
            f"{error_72:>12.3e}"
        )

    theory = 1.0 - np.cos(np.pi / plot_steps)
    ax.loglog(plot_steps, theory, "k--", label="1 - cos(pi/N) (segment, worst phase)")
    ax.axhline(1e-2, color="grey", linestyle=":", linewidth=1)
    ax.axhline(1e-3, color="grey", linestyle=":", linewidth=1)
    ax.axvline(72, color="tab:red", linestyle=":", linewidth=1, label="N = 72")
    ax.set(
        xlabel="steps per cycle N",
        ylabel="|relative error of the radius| (worst phase)",
        title="Minimum circumscribed circle: discretization error\n"
        "(circle path is exact for N >= 3, not shown)",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=130)
    print(f"Figure saved: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
