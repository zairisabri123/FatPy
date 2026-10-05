"""Test functions for the signal module.

Covers the waveforms, the common period of periodic signals, the
constant- and variable-amplitude signals and their error conditions.
"""

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from fatpy.utils import signal
from fatpy.utils.signal import (
    ConstantAmplitudeSignal,
    SignalError,
    VariableAmplitudeSignal,
    Waveform,
    common_time_axis,
    period_lcm,
    waveform_shape,
)

RTOL = 1e-9
ATOL = 1e-9  # absolute tolerance for values that should be zero


# -- waveform_shape -----------------------------------------------------------


def test_waveform_shape_values() -> None:
    """Each waveform matches its definition at characteristic angles."""
    theta = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 2])
    np.testing.assert_allclose(
        waveform_shape(Waveform.SINE, theta), np.sin(theta), rtol=RTOL
    )
    np.testing.assert_allclose(
        waveform_shape(Waveform.TRIANGLE, theta),
        [0.0, 0.5, 1.0, -1.0],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_array_equal(
        waveform_shape(Waveform.SQUARE, theta), [1.0, 1.0, 1.0, -1.0]
    )
    np.testing.assert_array_equal(waveform_shape(Waveform.CONSTANT, theta), 0.0)


@pytest.mark.parametrize("waveform", list(Waveform))
def test_waveform_shape_is_bounded(waveform: Waveform) -> None:
    """Every waveform stays within [-1, 1]."""
    values = waveform_shape(waveform, np.linspace(-10.0, 10.0, 1001))
    assert np.all(np.abs(values) <= 1.0)


# -- period_lcm / common_time_axis -----------------------------------------


@pytest.mark.parametrize(
    ("periods", "expected"),
    [([1.0], 1.0), ([0.2, 0.3], 0.6), ([1.0, 0.5], 1.0), ([1.5, 2.0], 6.0)],
)
def test_period_lcm(periods: list[float], expected: float) -> None:
    """The least common multiple is exact for decimal periods."""
    assert period_lcm(periods) == pytest.approx(expected, rel=RTOL)


def test_common_time_axis_pragtic_example() -> None:
    """Periods 1.5 s and 2 s, N = 4: 17 samples with dt = 0.375 s."""
    time = common_time_axis([1.5, 2.0], samples_per_period=4, max_cycles=1000)
    assert time.size == 17
    np.testing.assert_allclose(np.diff(time), 0.375, rtol=RTOL)
    assert time[0] == 0.0
    assert time[-1] == pytest.approx(6.0, rel=RTOL)


# -- ConstantAmplitudeSignal -------------------------------------------------


def test_constant_amplitude_sine_definition() -> None:
    """x(t) = mean + amplitude * sin(2*pi*t/T + phase)."""
    sig = ConstantAmplitudeSignal(amplitude=200.0, mean=50.0, period=2.0, phase=30.0)
    t = np.linspace(0.0, 2.0, 33)
    expected = 50.0 + 200.0 * np.sin(np.pi * t + np.deg2rad(30.0))
    np.testing.assert_allclose(sig.evaluate(t), expected, rtol=RTOL)


def test_frequency_and_period_are_equivalent() -> None:
    """A frequency f gives the same signal as the period 1/f."""
    t = np.linspace(0.0, 1.0, 17)
    by_freq = ConstantAmplitudeSignal(amplitude=1.0, frequency=4.0)
    by_period = ConstantAmplitudeSignal(amplitude=1.0, period=0.25)
    assert by_freq.effective_period == pytest.approx(0.25, rel=RTOL)
    np.testing.assert_allclose(by_freq.evaluate(t), by_period.evaluate(t), rtol=RTOL)


def test_constant_waveform_is_static() -> None:
    """A CONSTANT waveform has no period and always returns the mean."""
    sig = ConstantAmplitudeSignal(mean=30.0, waveform=Waveform.CONSTANT)
    assert sig.effective_period is None
    np.testing.assert_array_equal(sig.evaluate([0.0, 1.0, 7.5]), 30.0)


# -- VariableAmplitudeSignal -------------------------------------------------


def test_variable_amplitude_default_instants_are_indices() -> None:
    """Without time or time_step the instants are 0, 1, 2, ..."""
    sig = VariableAmplitudeSignal(np.array([3.0, -1.0, 4.0]))
    np.testing.assert_array_equal(sig.instants, [0.0, 1.0, 2.0])
    assert sig.effective_period is None


def test_variable_amplitude_time_step_and_time() -> None:
    """Instants come from time_step or from the explicit time array."""
    values = np.array([0.0, 5.0, -2.0])
    by_step = VariableAmplitudeSignal(values, time_step=0.1)
    by_time = VariableAmplitudeSignal(values, time=np.array([0.0, 0.1, 0.2]))
    np.testing.assert_allclose(by_step.instants, [0.0, 0.1, 0.2], rtol=RTOL)
    np.testing.assert_allclose(by_time.instants, by_step.instants, rtol=RTOL)
    np.testing.assert_array_equal(by_step.evaluate(by_step.instants), values)


def test_variable_amplitude_evaluates_subset_of_instants() -> None:
    """Any of the own instants can be requested, in any order."""
    sig = VariableAmplitudeSignal(np.array([10.0, 20.0, 30.0]), time_step=0.5)
    np.testing.assert_array_equal(sig.evaluate([1.0, 0.0]), [30.0, 10.0])


# -- error conditions ---------------------------------------------------------


def test_period_lcm_empty_raises() -> None:
    """No period at all."""
    with pytest.raises(SignalError, match="At least one period"):
        period_lcm([])


def test_period_lcm_non_positive_raises() -> None:
    """A zero or negative period."""
    with pytest.raises(SignalError, match="positive"):
        period_lcm([1.0, 0.0])


def test_common_time_axis_samples_per_period_raises() -> None:
    """Fewer than one sample per period."""
    with pytest.raises(SignalError, match="samples_per_period"):
        common_time_axis([1.0], samples_per_period=0, max_cycles=1000)


def test_common_time_axis_too_long_cycle_raises() -> None:
    """Nearly incommensurate periods give a too long common period."""
    with pytest.raises(SignalError, match="incommensurate"):
        common_time_axis([1.0, 1.001], samples_per_period=64, max_cycles=1000)


def test_constant_waveform_with_period_raises() -> None:
    """A CONSTANT waveform cannot have a period or a frequency."""
    with pytest.raises(SignalError, match="no period"):
        ConstantAmplitudeSignal(waveform=Waveform.CONSTANT, period=1.0)
    with pytest.raises(SignalError, match="no period"):
        ConstantAmplitudeSignal(waveform=Waveform.CONSTANT, frequency=1.0)


def test_constant_waveform_with_amplitude_raises() -> None:
    """A CONSTANT waveform requires amplitude = 0."""
    with pytest.raises(SignalError, match="amplitude = 0"):
        ConstantAmplitudeSignal(amplitude=1.0, waveform=Waveform.CONSTANT)


def test_periodic_without_period_or_frequency_raises() -> None:
    """Neither period nor frequency for a periodic waveform."""
    with pytest.raises(SignalError, match="Exactly one"):
        ConstantAmplitudeSignal(amplitude=1.0)


def test_periodic_with_period_and_frequency_raises() -> None:
    """Both period and frequency."""
    with pytest.raises(SignalError, match="Exactly one"):
        ConstantAmplitudeSignal(amplitude=1.0, period=1.0, frequency=1.0)


def test_non_positive_period_raises() -> None:
    """A zero or negative period or frequency."""
    with pytest.raises(SignalError, match="positive"):
        ConstantAmplitudeSignal(amplitude=1.0, period=-1.0)
    with pytest.raises(SignalError, match="positive"):
        ConstantAmplitudeSignal(amplitude=1.0, frequency=0.0)


def test_variable_amplitude_single_value_raises() -> None:
    """Fewer than two values."""
    with pytest.raises(SignalError, match="at least two"):
        VariableAmplitudeSignal(np.array([1.0]))


def test_variable_amplitude_time_and_time_step_raises() -> None:
    """Both time and time_step."""
    with pytest.raises(SignalError, match="not both"):
        VariableAmplitudeSignal(
            np.array([1.0, 2.0]), time=np.array([0.0, 1.0]), time_step=1.0
        )


def test_variable_amplitude_time_length_mismatch_raises() -> None:
    """time and values of different lengths."""
    with pytest.raises(SignalError, match="same length"):
        VariableAmplitudeSignal(np.array([1.0, 2.0]), time=np.array([0.0, 1.0, 2.0]))


def test_variable_amplitude_non_increasing_time_raises() -> None:
    """time not strictly increasing."""
    with pytest.raises(SignalError, match="strictly increasing"):
        VariableAmplitudeSignal(np.array([1.0, 2.0]), time=np.array([1.0, 1.0]))


def test_variable_amplitude_non_positive_time_step_raises() -> None:
    """A zero or negative time step."""
    with pytest.raises(SignalError, match="time_step"):
        VariableAmplitudeSignal(np.array([1.0, 2.0]), time_step=0.0)


def test_variable_amplitude_no_interpolation() -> None:
    """Evaluating between the own instants is refused."""
    sig = VariableAmplitudeSignal(np.array([1.0, 2.0]), time_step=1.0)
    with pytest.raises(SignalError, match="no interpolation"):
        sig.evaluate([0.5])


def test_signal_error_is_value_error() -> None:
    """SignalError can be caught as a ValueError."""
    assert issubclass(SignalError, ValueError)


# -- immutability / dependencies ---------------------------------------------


@pytest.mark.parametrize(
    ("obj", "attribute"),
    [
        (ConstantAmplitudeSignal(amplitude=1.0, period=1.0), "amplitude"),
        (VariableAmplitudeSignal(np.array([1.0, 2.0])), "values"),
    ],
)
def test_signals_are_immutable(obj: object, attribute: str) -> None:
    """Signals are frozen dataclasses."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, attribute, 0.0)


def test_signal_module_does_not_import_data_parsing() -> None:
    """utils.signal stays independent of fatpy.data_parsing."""
    tree = ast.parse(Path(signal.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not [m for m in imported if "data_parsing" in m]
