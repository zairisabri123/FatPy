"""Signal processing utilities module.

This module provides various functions and classes for signal processing tasks.

Overview:
    A signal is a pure function of time, without any physical meaning (the
    quantity, component and unit are attached by `fatpy.data_parsing.loads`).
    Two kinds of signals are provided, following the loading definition of
    PragTic:

    - `ConstantAmplitudeSignal`: periodic signal defined by its amplitude,
      mean, waveform, period (or frequency) and phase. A ``CONSTANT``
      waveform gives a time-independent (static) value.
    - `VariableAmplitudeSignal`: load sequence given by its values at its own
      time instants. It is never interpolated.

    Periodic signals with different periods are evaluated together over their
    common period (least common multiple), see `common_time_axis`.

Conventions:
    - Time is in seconds, frequency in Hz and phase in degrees.
    - Periods are rounded to `TIME_PRECISION` (1e-9 s) to compute their least
      common multiple exactly with `fractions.Fraction`.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Time resolution [s]: periods are rounded to it and time instants are
#: matched with it.
TIME_PRECISION = 1e-9
_TIME_SCALE = round(1 / TIME_PRECISION)


class SignalError(ValueError):
    """Raised when a signal is inconsistently defined or evaluated."""


class Waveform(Enum):
    """Shape of one period of a constant-amplitude signal."""

    SINE = "sine"
    TRIANGLE = "triangle"
    SQUARE = "square"
    CONSTANT = "constant"


def waveform_shape(waveform: Waveform, theta: ArrayLike) -> NDArray[np.float64]:
    r"""Normalized waveform value, in [-1, 1], at the phase angle `theta`.

    ??? abstract "Math Equations"
        $$
        f(\theta) =
        \begin{cases}
        \sin\theta & \text{SINE} \\
        \frac{2}{\pi}\arcsin(\sin\theta) & \text{TRIANGLE} \\
        +1 \text{ if } \sin\theta \ge 0 \text{ else } -1 & \text{SQUARE} \\
        0 & \text{CONSTANT}
        \end{cases}
        $$

    Args:
        waveform: Shape of the signal.
        theta: Phase angle [rad], any shape.

    Returns:
        Array of the same shape as `theta`.
    """
    angle = np.asarray(theta, dtype=np.float64)
    match waveform:
        case Waveform.SINE:
            return np.sin(angle)
        case Waveform.TRIANGLE:
            return (2.0 / np.pi) * np.arcsin(np.clip(np.sin(angle), -1.0, 1.0))
        case Waveform.SQUARE:
            return np.where(np.sin(angle) >= 0.0, 1.0, -1.0)
        case Waveform.CONSTANT:
            return np.zeros_like(angle)


def period_lcm(periods: Sequence[float]) -> float:
    """Least common multiple of periods (the common period of the signals).

    Each period is rounded to `TIME_PRECISION` and converted to a
    `fractions.Fraction`, so the result is exact (e.g. 0.2 s and 0.3 s give
    0.6 s).

    Args:
        periods: Periods [s].

    Returns:
        The least common multiple [s].

    Raises:
        SignalError: If `periods` is empty or a period is not positive at
            `TIME_PRECISION`.
    """
    fractions = [Fraction(round(p * _TIME_SCALE), _TIME_SCALE) for p in periods]
    if not fractions:
        raise SignalError("At least one period is needed")
    if any(f <= 0 for f in fractions):
        raise SignalError(f"Periods must be positive, got {list(periods)}")
    numerator = math.lcm(*(f.numerator for f in fractions))
    denominator = math.gcd(*(f.denominator for f in fractions))
    return numerator / denominator


def common_time_axis(
    periods: Sequence[float],
    samples_per_period: int,
    max_cycles: float,
) -> NDArray[np.float64]:
    """Time instants covering one common period of several periodic signals.

    The time step resolves the shortest period with `samples_per_period`
    points, and the axis spans the least common multiple of the periods,
    both ends included.

    Args:
        periods: Periods of the signals [s].
        samples_per_period: Number of time steps in the shortest period.
        max_cycles: Maximum length of the common period, as a number of
            shortest periods.

    Returns:
        Array of shape (n,) with ``n = cycle / dt + 1``, starting at 0.

    Raises:
        SignalError: If `samples_per_period` is lower than 1, if a period is
            not positive, or if the common period exceeds `max_cycles`
            shortest periods (nearly incommensurate periods).
    """
    if samples_per_period < 1:
        raise SignalError(f"samples_per_period must be >= 1, got {samples_per_period}")
    cycle = period_lcm(periods)
    shortest = min(periods)
    if cycle > max_cycles * shortest:
        raise SignalError(
            f"The common period of {list(periods)} is {cycle:g} s, more than "
            f"{max_cycles:g} times the shortest period; the periods are nearly "
            "incommensurate"
        )
    dt = shortest / samples_per_period
    n = round(cycle / dt) + 1
    return np.arange(n, dtype=np.float64) * dt


class Signal(Protocol):
    """Interface shared by all signals."""

    def evaluate(self, time: ArrayLike) -> NDArray[np.float64]:
        """Signal values at the given time instants [s]."""
        ...

    @property
    def effective_period(self) -> float | None:
        """Period [s] of a periodic signal, ``None`` for a non-periodic one."""
        ...


@dataclass(frozen=True)
class ConstantAmplitudeSignal:
    r"""Periodic signal with constant amplitude and mean.

    ??? abstract "Math Equations"
        $$ x(t) = x_m + x_a \, f\!\left(\frac{2\pi t}{T} +
        \frac{\pi \varphi}{180}\right) $$

        with $f$ the normalized waveform (see `waveform_shape`).

    Attributes:
        amplitude: Amplitude $x_a$.
        mean: Mean value $x_m$.
        waveform: Shape of one period.
        period: Period $T$ [s]. Exactly one of `period` and `frequency` is
            given, none for a ``CONSTANT`` waveform.
        frequency: Frequency $1/T$ [Hz].
        phase: Phase shift $\varphi$ [deg].
    """

    amplitude: float = 0.0
    mean: float = 0.0
    waveform: Waveform = Waveform.SINE
    period: float | None = None
    frequency: float | None = None
    phase: float = 0.0

    def __post_init__(self) -> None:
        """Validate the period definition.

        Raises:
            SignalError: If a ``CONSTANT`` waveform has a period, a frequency
                or a non-zero amplitude, or if another waveform does not have
                exactly one positive period or frequency.
        """
        if self.waveform is Waveform.CONSTANT:
            if self.period is not None or self.frequency is not None:
                raise SignalError("A CONSTANT waveform has no period or frequency")
            if self.amplitude != 0.0:
                raise SignalError("A CONSTANT waveform requires amplitude = 0")
            return
        if (self.period is None) == (self.frequency is None):
            raise SignalError("Exactly one of period or frequency must be given")
        given = self.period if self.period is not None else self.frequency
        if given is not None and given <= 0.0:
            raise SignalError(f"Period and frequency must be positive, got {given}")

    @property
    def effective_period(self) -> float | None:
        """Period [s], ``None`` for a ``CONSTANT`` waveform."""
        if self.period is not None:
            return self.period
        if self.frequency is not None:
            return 1.0 / self.frequency
        return None

    def evaluate(self, time: ArrayLike) -> NDArray[np.float64]:
        """Signal values at the given time instants.

        Args:
            time: Time instants [s], any shape.

        Returns:
            Array of the same shape as `time`.
        """
        t = np.asarray(time, dtype=np.float64)
        period = self.effective_period
        if period is None:
            return np.full_like(t, self.mean)
        theta = 2.0 * np.pi * t / period + np.deg2rad(self.phase)
        return self.mean + self.amplitude * waveform_shape(self.waveform, theta)


@dataclass(frozen=True, eq=False)
class VariableAmplitudeSignal:
    """Load sequence defined by its values at its own time instants.

    The signal is not interpolated: it can only be evaluated at its own
    instants (see `instants`).

    Attributes:
        values: Signal values, shape (n,) with n >= 2.
        time: Time instants [s], strictly increasing, shape (n,).
        time_step: Constant time step [s], used if `time` is not given.
        instants: Resolved time instants: `time`, or ``arange(n) *
            time_step``, or the sequence indices ``0, 1, 2, ...`` if neither
            is given.
    """

    values: NDArray[np.float64]
    time: NDArray[np.float64] | None = None
    time_step: float | None = None
    instants: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Convert the inputs to float64 arrays and resolve the time instants.

        Raises:
            SignalError: If there are fewer than two values, if both `time`
                and `time_step` are given, if `time` does not match `values`
                or is not strictly increasing, or if `time_step` is not
                positive.
        """
        values = np.array(self.values, dtype=np.float64)
        if values.ndim != 1 or values.size < 2:
            raise SignalError("A variable-amplitude signal needs at least two values")
        object.__setattr__(self, "values", values)

        if self.time is not None and self.time_step is not None:
            raise SignalError("Give either time or time_step, not both")
        if self.time is not None:
            instants = np.array(self.time, dtype=np.float64)
            if instants.shape != values.shape:
                raise SignalError("time and values must have the same length")
            if np.any(np.diff(instants) <= 0.0):
                raise SignalError("time must be strictly increasing")
            object.__setattr__(self, "time", instants)
        elif self.time_step is not None:
            if self.time_step <= 0.0:
                raise SignalError(f"time_step must be positive, got {self.time_step}")
            instants = np.arange(values.size, dtype=np.float64) * self.time_step
        else:
            instants = np.arange(values.size, dtype=np.float64)
        object.__setattr__(self, "instants", instants)

    @property
    def effective_period(self) -> float | None:
        """Always ``None``: a load sequence is not periodic."""
        return None

    def evaluate(self, time: ArrayLike) -> NDArray[np.float64]:
        """Signal values at some of its own time instants.

        Args:
            time: Time instants [s], any shape; each must match one of
                `instants` within `TIME_PRECISION`.

        Returns:
            Array of the same shape as `time`.

        Raises:
            SignalError: If a requested instant is not one of `instants`
                (the signal is never interpolated).
        """
        t = np.asarray(time, dtype=np.float64)
        known = self.instants
        right = np.clip(np.searchsorted(known, t), 1, known.size - 1)
        index = np.where(t - known[right - 1] < known[right] - t, right - 1, right)
        if np.any(np.abs(known[index] - t) > TIME_PRECISION):
            raise SignalError(
                "A variable-amplitude signal is only defined at its own time "
                "instants (no interpolation)"
            )
        return self.values[index]
