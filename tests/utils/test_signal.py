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


def test_square_zero_crossings_count_as_positive() -> None:
    """Change 1: |sin| < 1e-12 at k*pi gives +1, so 0 and 2*pi agree."""
    theta = np.array([0.0, np.pi, 2 * np.pi, 4 * np.pi])
    assert np.sin(2 * np.pi) < 0.0  # the float rounding the rule handles
    np.testing.assert_array_equal(waveform_shape(Waveform.SQUARE, theta), 1.0)


@pytest.mark.parametrize("waveform", list(Waveform))
def test_closed_cycle_for_every_waveform(waveform: Waveform) -> None:
    """First and last sample of one period are equal, SQUARE included."""
    if waveform is Waveform.CONSTANT:
        sig = ConstantAmplitudeSignal(mean=50.0, waveform=waveform)
    else:
        sig = ConstantAmplitudeSignal(100.0, 50.0, waveform, period=1.0)
    values = sig.evaluate(common_time_axis([1.0], 64, 1000))
    assert values[-1] == pytest.approx(values[0], rel=RTOL)


# -- period_lcm / common_time_axis -----------------------------------------


@pytest.mark.parametrize(
    ("periods", "expected"),
    [
        ([1.0], 1.0),
        ([0.2, 0.3], 0.6),
        ([1.0, 0.5], 1.0),
        ([1.5, 2.0], 6.0),
        ([1 / 3, 0.5], 1.0),
        ([0.1 + 0.2], 0.3),
    ],
)
def test_period_lcm(periods: list[float], expected: float) -> None:
    """The least common multiple is exact for decimal and rational periods."""
    assert period_lcm(periods) == pytest.approx(expected, rel=RTOL)


def test_common_time_axis_pragtic_example() -> None:
    """Periods 1.5 s and 2 s, N = 4: 17 samples with dt = 0.375 s."""
    time = common_time_axis([1.5, 2.0], samples_per_period=4, max_cycles=1000)
    assert time.size == 17
    np.testing.assert_allclose(np.diff(time), 0.375, rtol=RTOL)
    assert time[0] == 0.0
    assert time[-1] == pytest.approx(6.0, rel=RTOL)


def test_common_time_axis_requires_four_samples() -> None:
    """Change 4: samples_per_period below 4 is refused, 4 is accepted."""
    with pytest.raises(SignalError, match=r"must be an integer >= 4, got 3"):
        common_time_axis([1.0], samples_per_period=3, max_cycles=1000)
    with pytest.raises(SignalError, match=r"must be an integer >= 4, got 4\.5"):
        common_time_axis([1.0], samples_per_period=4.5, max_cycles=1000)  # type: ignore[arg-type]
    assert common_time_axis([1.0], samples_per_period=4, max_cycles=1000).size == 5


def test_common_time_axis_incommensurate_advice() -> None:
    """Change 4: the error advises rounding the frequency ratio."""
    with pytest.raises(SignalError, match=r"incommensurate.*1:1\.414 -> 5:7"):
        common_time_axis([1.0, 1.4142], samples_per_period=64, max_cycles=1000)


# -- ConstantAmplitudeSignal -------------------------------------------------


def test_constant_amplitude_sine_definition() -> None:
    """x(t) = mean + amplitude * sin(2*pi*t/T + phase)."""
    sig = ConstantAmplitudeSignal(amplitude=200.0, mean=50.0, period=2.0, phase=30.0)
    t = np.linspace(0.0, 2.0, 33)
    expected = 50.0 + 200.0 * np.sin(np.pi * t + np.deg2rad(30.0))
    np.testing.assert_allclose(sig.evaluate(t), expected, rtol=RTOL)
    assert sig.instants is None


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


def test_negative_amplitude_raises() -> None:
    """Change 2: a negative amplitude is refused, with its value."""
    with pytest.raises(SignalError, match=r"amplitude must be >= 0, got -1\.0"):
        ConstantAmplitudeSignal(amplitude=-1.0, period=1.0)


@pytest.mark.parametrize(
    "parameter", ["amplitude", "mean", "phase", "period", "frequency"]
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_parameter_raises(parameter: str, bad: float) -> None:
    """Change 2: NaN or inf in any parameter is refused, naming it."""
    params = {"amplitude": 1.0, "period": 1.0} | {parameter: bad}
    if parameter == "frequency":
        del params["period"]
    with pytest.raises(SignalError, match=f"{parameter} must be finite"):
        ConstantAmplitudeSignal(**params)  # type: ignore[arg-type]


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


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_variable_amplitude_non_finite_raises(bad: float) -> None:
    """Change 3: NaN or inf in values or time is refused, with its index."""
    with pytest.raises(SignalError, match="values must be finite.*index 1"):
        VariableAmplitudeSignal(np.array([1.0, bad, 3.0]))
    with pytest.raises(SignalError, match="time must be finite.*index 2"):
        VariableAmplitudeSignal(np.ones(3), time=np.array([0.0, 1.0, bad]))


@pytest.mark.parametrize("attribute", ["values", "time", "instants"])
def test_variable_amplitude_arrays_are_read_only(attribute: str) -> None:
    """Change 3: values, time and instants cannot be modified in place."""
    sig = VariableAmplitudeSignal(np.array([1.0, 2.0]), time=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="read-only"):
        getattr(sig, attribute)[0] = 9.0


def test_variable_amplitude_does_not_alias_inputs() -> None:
    """Change 3: the caller's arrays stay writable and are not shared."""
    values = np.array([1.0, 2.0])
    sig = VariableAmplitudeSignal(values)
    values[0] = 9.0
    assert sig.values[0] == 1.0


def test_variable_amplitude_duration() -> None:
    """Change 3: duration is the last instant minus the first."""
    assert VariableAmplitudeSignal(np.zeros(5), time_step=0.1).duration == (
        pytest.approx(0.4, rel=RTOL)
    )
    timed = VariableAmplitudeSignal(np.zeros(3), time=np.array([2.0, 3.0, 7.5]))
    assert timed.duration == pytest.approx(5.5, rel=RTOL)


def test_variable_amplitude_is_sequence_without_time_scale() -> None:
    """Change 3: only values without time or time_step form a load sequence."""
    values = np.array([1.0, 2.0])
    assert VariableAmplitudeSignal(values).is_sequence is True
    assert VariableAmplitudeSignal(values, time_step=0.1).is_sequence is False
    timed = VariableAmplitudeSignal(values, time=np.array([0.0, 1.0]))
    assert timed.is_sequence is False


# -- error conditions ---------------------------------------------------------


def test_period_lcm_empty_raises() -> None:
    """No period at all."""
    with pytest.raises(SignalError, match="At least one period"):
        period_lcm([])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_period_lcm_invalid_period_raises(bad: float) -> None:
    """A zero, negative or non-finite period, named in the message."""
    with pytest.raises(SignalError, match="finite and positive"):
        period_lcm([1.0, bad])


def test_period_lcm_below_precision_raises() -> None:
    """A positive period shorter than TIME_PRECISION."""
    with pytest.raises(SignalError, match="below TIME_PRECISION"):
        period_lcm([1e-12])


def test_constant_waveform_with_period_raises() -> None:
    """A CONSTANT waveform cannot have a period or a frequency."""
    with pytest.raises(SignalError, match="no period"):
        ConstantAmplitudeSignal(waveform=Waveform.CONSTANT, period=1.0)
    with pytest.raises(SignalError, match="no period"):
        ConstantAmplitudeSignal(waveform=Waveform.CONSTANT, frequency=1.0)


def test_constant_waveform_with_amplitude_raises() -> None:
    """A CONSTANT waveform requires amplitude = 0."""
    with pytest.raises(SignalError, match="amplitude = 0, got 1.0"):
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
    """A zero or negative period or frequency, named with its value."""
    with pytest.raises(SignalError, match=r"period must be positive, got -1\.0"):
        ConstantAmplitudeSignal(amplitude=1.0, period=-1.0)
    with pytest.raises(SignalError, match=r"frequency must be positive, got 0\.0"):
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
    with pytest.raises(SignalError, match="same shape"):
        VariableAmplitudeSignal(np.array([1.0, 2.0]), time=np.array([0.0, 1.0, 2.0]))


def test_variable_amplitude_non_increasing_time_raises() -> None:
    """time not strictly increasing, with the offending values."""
    with pytest.raises(SignalError, match="strictly increasing.*index 2"):
        VariableAmplitudeSignal(np.ones(3), time=np.array([0.0, 1.0, 1.0]))


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
def test_variable_amplitude_invalid_time_step_raises(bad: float) -> None:
    """A zero, negative or non-finite time step."""
    with pytest.raises(SignalError, match="time_step must be finite and positive"):
        VariableAmplitudeSignal(np.array([1.0, 2.0]), time_step=bad)


def test_variable_amplitude_no_interpolation() -> None:
    """Evaluating between the own instants is refused, naming the instant."""
    sig = VariableAmplitudeSignal(np.array([1.0, 2.0]), time_step=1.0)
    with pytest.raises(SignalError, match=r"Instant 0\.5 s.*no interpolation"):
        sig.evaluate([0.5])


def test_signal_error_is_value_error() -> None:
    """SignalError can be caught as a ValueError."""
    assert issubclass(SignalError, ValueError)


# -- immutability / module ----------------------------------------------------


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


def test_module_docstring() -> None:
    """Change 5: typo fixed and future PSD signal documented."""
    doc = signal.__doc__ or ""
    assert "provided, :" not in doc
    assert "Future work:" in doc
    assert "PSD" in doc
    assert "Signal` protocol" in doc


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
