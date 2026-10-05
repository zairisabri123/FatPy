"""Signal processing utilities module.

This module provides various functions and classes for signal processing tasks.

Overview:
    A signal is a pure function of time, without any physical meaning (the
    quantity, component and unit are attached by `fatpy.data_parsing.loads`).
    Two kinds of signals are provided:

    - `ConstantAmplitudeSignal`: periodic signal defined by its amplitude,
      mean, waveform, period (or frequency) and phase. A ``CONSTANT``
      waveform gives a time-independent (static) value.
    - `VariableAmplitudeSignal`: values given at their own time instants,
      never interpolated. Without a time scale (no ``time`` or
      ``time_step``) the values form a load sequence.

    Periodic signals with different periods are evaluated together over their
    common period (least common multiple), see `common_time_axis`.

Conventions:
    - Time is in seconds, frequency in Hz and phase in degrees.
    - To compute their least common multiple exactly, periods are replaced by
      the simplest `fractions.Fraction` within `TIME_PRECISION` (1e-9 s), so
      1/3 s or 0.2 s stay exact.

Future work:
    - PSD signal (random loading defined by a power spectral density), as a
      new class following the `Signal` protocol. It provides its own time
      instants through `Signal.instants`, so channels and load cases accept it
      unchanged.
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
_TIME_DIGITS = 9  # TIME_PRECISION = 10**-_TIME_DIGITS
#: Minimum number of time steps in the shortest period.
MIN_SAMPLES_PER_PERIOD = 4
# |sin(theta)| below this is a zero crossing of the square wave: theta = k*pi
# is not exactly representable, so sin(2*pi) is about -2.4e-16, not 0.
_SINE_ZERO = 1e-12


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

    For ``SQUARE``, ``|sin θ| < 1e-12`` counts as zero (value +1), so a cycle
    starts and ends on the same value.

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
            sine = np.sin(angle)
            return np.where((sine >= 0.0) | (np.abs(sine) < _SINE_ZERO), 1.0, -1.0)
        case Waveform.CONSTANT:
            return np.zeros_like(angle)


def _period_fraction(period: float) -> Fraction:
    """Simplest fraction within `TIME_PRECISION` of `period`.

    Denominators 1, 10, ..., 1e9 are tried in turn; the last one always
    matches, since it is a rounding to 1e-9.
    """
    exact = Fraction(period)
    for digits in range(_TIME_DIGITS + 1):
        candidate = exact.limit_denominator(10**digits)
        if abs(candidate - exact) <= TIME_PRECISION:
            break
    return candidate


def period_lcm(periods: Sequence[float]) -> float:
    """Least common multiple of periods (the common period of the signals).

    Each period is replaced by the simplest `fractions.Fraction` within
    `TIME_PRECISION`, so the result is exact for decimal and rational periods
    (e.g. 0.2 s and 0.3 s give 0.6 s, 1/3 s and 0.5 s give 1 s).

    Args:
        periods: Periods [s].

    Returns:
        The least common multiple [s].

    Raises:
        SignalError: If `periods` is empty or a period is not finite and
            positive at `TIME_PRECISION`.
    """
    if not periods:
        raise SignalError("At least one period is needed")
    fractions = []
    for period in periods:
        if not math.isfinite(period) or period <= 0.0:
            raise SignalError(f"Periods must be finite and positive, got {period!r}")
        fraction = _period_fraction(period)
        if fraction == 0:
            raise SignalError(
                f"Period {period!r} s is below TIME_PRECISION ({TIME_PRECISION} s)"
            )
        fractions.append(fraction)
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
        samples_per_period: Number of time steps in the shortest period, an
            integer, at least `MIN_SAMPLES_PER_PERIOD`.
        max_cycles: Maximum length of the common period, as a number of
            shortest periods.

    Returns:
        Array of shape (n,) with ``n = cycle / dt + 1``, starting at 0.

    Raises:
        SignalError: If `samples_per_period` is too low, if a period is not
            finite and positive, or if the common period exceeds `max_cycles`
            shortest periods (nearly incommensurate periods).
    """
    if (
        not isinstance(samples_per_period, (int, np.integer))
        or samples_per_period < MIN_SAMPLES_PER_PERIOD
    ):
        raise SignalError(
            f"samples_per_period must be an integer >= {MIN_SAMPLES_PER_PERIOD}, "
            f"got {samples_per_period!r}"
        )
    cycle = period_lcm(periods)
    shortest = min(periods)
    if cycle > max_cycles * shortest:
        raise SignalError(
            f"The common period of {list(periods)} s is {cycle:g} s, more than "
            f"{max_cycles:g} times the shortest period: the periods are nearly "
            "incommensurate. Round the frequency ratio to a ratio of small "
            "integers (e.g. 1:1.414 -> 5:7)."
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

    @property
    def instants(self) -> NDArray[np.float64] | None:
        """Own time instants [s], ``None`` if any instant can be evaluated."""
        ...


def _check_finite(**parameters: float | None) -> None:
    """Raise `SignalError` naming the first parameter that is NaN or inf."""
    for name, value in parameters.items():
        if value is not None and not math.isfinite(value):
            raise SignalError(f"{name} must be finite, got {value!r}")


def _finite_read_only(name: str, data: ArrayLike) -> NDArray[np.float64]:
    """Finite, read-only float64 copy of `data`.

    Raises:
        SignalError: If `data` contains NaN or inf.
    """
    array = np.array(data, dtype=np.float64)
    bad = np.flatnonzero(~np.isfinite(array))
    if bad.size:
        raise SignalError(
            f"{name} must be finite, got {array.flat[bad[0]]!r} at index {bad[0]}"
        )
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ConstantAmplitudeSignal:
    r"""Periodic signal with constant amplitude and mean.

    ??? abstract "Math Equations"
        $$ x(t) = x_m + x_a \, f\!\left(\frac{2\pi t}{T} +
        \frac{\pi \varphi}{180}\right) $$

        with $f$ the normalized waveform (see `waveform_shape`).

    Attributes:
        amplitude: Amplitude $x_a \ge 0$.
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
        """Validate the parameters.

        Raises:
            SignalError: If a parameter is not finite, if the amplitude is
                negative, if a ``CONSTANT`` waveform has a period, a frequency
                or a non-zero amplitude, or if another waveform does not have
                exactly one positive period or frequency.
        """
        _check_finite(
            amplitude=self.amplitude,
            mean=self.mean,
            phase=self.phase,
            period=self.period,
            frequency=self.frequency,
        )
        if self.amplitude < 0.0:
            raise SignalError(f"amplitude must be >= 0, got {self.amplitude!r}")
        if self.waveform is Waveform.CONSTANT:
            if self.period is not None or self.frequency is not None:
                raise SignalError(
                    "A CONSTANT waveform has no period or frequency, got "
                    f"period={self.period!r}, frequency={self.frequency!r}"
                )
            if self.amplitude != 0.0:
                raise SignalError(
                    "A CONSTANT waveform requires amplitude = 0, "
                    f"got {self.amplitude!r}"
                )
            return
        if (self.period is None) == (self.frequency is None):
            raise SignalError(
                "Exactly one of period or frequency must be given, got "
                f"period={self.period!r}, frequency={self.frequency!r}"
            )
        for name, value in (("period", self.period), ("frequency", self.frequency)):
            if value is not None and value <= 0.0:
                raise SignalError(f"{name} must be positive, got {value!r}")

    @property
    def effective_period(self) -> float | None:
        """Period [s], ``None`` for a ``CONSTANT`` waveform."""
        if self.period is not None:
            return self.period
        if self.frequency is not None:
            return 1.0 / self.frequency
        return None

    @property
    def instants(self) -> None:
        """Always ``None``: the signal can be evaluated at any instant."""
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
    """Signal defined by its values at its own time instants.

    The signal is not interpolated: it can only be evaluated at its own
    instants (see `instants`). Without `time` and `time_step` the values have
    no time scale and form a load sequence (see `is_sequence`).

    Attributes:
        values: Signal values, shape (n,) with n >= 2, finite, read-only.
        time: Time instants [s], finite, strictly increasing, shape (n,),
            read-only.
        time_step: Constant time step [s], used if `time` is not given.
        instants: Resolved time instants, read-only: `time`, or
            ``arange(n) * time_step``, or the indices ``0, 1, 2, ...`` of a
            load sequence.
    """

    values: NDArray[np.float64]
    time: NDArray[np.float64] | None = None
    time_step: float | None = None
    instants: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Store read-only float64 arrays and resolve the time instants.

        Raises:
            SignalError: If values are not 1D with at least two entries, if
                values or time contain NaN or inf, if both `time` and
                `time_step` are given, if `time` does not match `values` or
                is not strictly increasing, or if `time_step` is not finite
                and positive.
        """
        values = _finite_read_only("values", self.values)
        if values.ndim != 1 or values.size < 2:
            raise SignalError(
                f"values must be 1D with at least two entries, got shape {values.shape}"
            )
        object.__setattr__(self, "values", values)

        if self.time is not None and self.time_step is not None:
            raise SignalError(
                "Give either time or time_step, not both, got "
                f"time_step={self.time_step!r}"
            )
        if self.time is not None:
            instants = _finite_read_only("time", self.time)
            if instants.shape != values.shape:
                raise SignalError(
                    f"time and values must have the same shape, got "
                    f"{instants.shape} and {values.shape}"
                )
            steps = np.diff(instants)
            if np.any(steps <= 0.0):
                i = int(np.argmax(steps <= 0.0))
                raise SignalError(
                    "time must be strictly increasing, got "
                    f"{instants[i]!r} then {instants[i + 1]!r} at index {i + 1}"
                )
            object.__setattr__(self, "time", instants)
        else:
            if self.time_step is not None and (
                not math.isfinite(self.time_step) or self.time_step <= 0.0
            ):
                raise SignalError(
                    f"time_step must be finite and positive, got {self.time_step!r}"
                )
            step = 1.0 if self.time_step is None else self.time_step
            instants = np.arange(values.size, dtype=np.float64) * step
            instants.setflags(write=False)
        object.__setattr__(self, "instants", instants)

    @property
    def effective_period(self) -> float | None:
        """Always ``None``: the signal is not periodic."""
        return None

    @property
    def is_sequence(self) -> bool:
        """``True`` for a load sequence: neither `time` nor `time_step` given."""
        return self.time is None and self.time_step is None

    @property
    def duration(self) -> float:
        """Last instant minus first instant [s] (number of steps for a sequence)."""
        return float(self.instants[-1] - self.instants[0])

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
        off = np.abs(known[index] - t) > TIME_PRECISION
        if np.any(off):
            raise SignalError(
                f"Instant {float(t[off].flat[0])!r} s is not one of the signal "
                "instants (no interpolation)"
            )
        return self.values[index]
