"""Test functions for the load definition module.

Covers the reference load cases (tension, torsion, in-phase and
out-of-phase combined loading, mean stress, different frequencies, force
sequence), the time axis rules, the Voigt export and the error conditions.
"""

import dataclasses

import numpy as np
import pytest

from fatpy.data_parsing.loads import (
    DEFAULT_SAMPLES_PER_PERIOD,
    Channel,
    ChannelHistory,
    LoadCase,
    LoadDefinitionError,
    LoadHistory,
    Quantity,
    voigt_index,
)
from fatpy.utils import voigt
from fatpy.utils.signal import (
    ConstantAmplitudeSignal,
    SignalError,
    VariableAmplitudeSignal,
    Waveform,
)

RTOL = 1e-9
ATOL = 1e-9  # absolute tolerance for values that should be zero


def tension(frequency: float = 1.0, mean: float = 0.0, phase: float = 0.0) -> Channel:
    """Channel s11: sine, 200 MPa, 1 Hz by default."""
    sig = ConstantAmplitudeSignal(200.0, mean, frequency=frequency, phase=phase)
    return Channel("tension", Quantity.STRESS, "s11", sig)


def torsion(frequency: float = 1.0, mean: float = 0.0, phase: float = 0.0) -> Channel:
    """Channel s12: sine, 115 MPa, 1 Hz by default."""
    sig = ConstantAmplitudeSignal(115.0, mean, frequency=frequency, phase=phase)
    return Channel("torsion", Quantity.STRESS, "s12", sig)


def static(name: str = "preload", mean: float = 30.0) -> Channel:
    """Static s22 channel (CONSTANT waveform)."""
    sig = ConstantAmplitudeSignal(mean=mean, waveform=Waveform.CONSTANT)
    return Channel(name, Quantity.STRESS, "s22", sig)


def force_sequence() -> Channel:
    """Fx sequence 0, 5000, -2000, 8000, 0 N with dt = 0.1 s."""
    values = np.array([0.0, 5000.0, -2000.0, 8000.0, 0.0])
    return Channel(
        "force", Quantity.FORCE, "Fx", VariableAmplitudeSignal(values, time_step=0.1)
    )


@pytest.fixture
def out_of_phase() -> LoadHistory:
    """Case 4: s11 (200 MPa) and s12 (115 MPa, phase 90 deg) at 1 Hz."""
    return LoadCase("out-of-phase", [tension(), torsion(phase=90.0)]).history()


# -- reference load cases -----------------------------------------------------


def test_case1_tension() -> None:
    """s11 = 200 sin(2 pi t) over one period of 1 s."""
    history = LoadCase("tension", [tension()]).history()
    t = history.time
    assert len(history) == DEFAULT_SAMPLES_PER_PERIOD + 1
    assert t[-1] == pytest.approx(1.0, rel=RTOL)
    np.testing.assert_allclose(
        history.get("tension").values,
        200.0 * np.sin(2 * np.pi * t),
        rtol=RTOL,
        atol=ATOL,
    )
    assert history.get("tension").unit == "MPa"
    assert history.is_sequence is False


def test_case2_torsion() -> None:
    """s12 = 115 sin(2 pi t)."""
    history = LoadCase("torsion", [torsion()]).history()
    np.testing.assert_allclose(
        history.get("torsion").values,
        115.0 * np.sin(2 * np.pi * history.time),
        rtol=RTOL,
        atol=ATOL,
    )


def test_case3_in_phase() -> None:
    """In phase: s11 / 200 and s12 / 115 are equal at every instant."""
    history = LoadCase("in-phase", [tension(), torsion()]).history()
    np.testing.assert_allclose(
        history.get("tension").values / 200.0,
        history.get("torsion").values / 115.0,
        rtol=RTOL,
        atol=ATOL,
    )


def test_case4_out_of_phase(out_of_phase: LoadHistory) -> None:
    """90 deg out of phase: (s11 / 200)^2 + (s12 / 115)^2 = 1."""
    s11 = out_of_phase.get("tension").values
    s12 = out_of_phase.get("torsion").values
    np.testing.assert_allclose((s11 / 200.0) ** 2 + (s12 / 115.0) ** 2, 1.0, rtol=RTOL)


def test_case5_mean_stress() -> None:
    """Mean 100 MPa, amplitude 200 MPa: max 300 MPa, min -100 MPa."""
    history = LoadCase("mean", [tension(mean=100.0)]).history()
    values = history.get("tension").values
    assert values.max() == pytest.approx(300.0, rel=RTOL)
    assert values.min() == pytest.approx(-100.0, rel=RTOL)
    assert values.mean() == pytest.approx(100.0, rel=RTOL)


def test_case6_different_frequencies() -> None:
    """s11 at 1 Hz and s12 at 2 Hz: cycle of 1 s sampled with 129 points."""
    history = LoadCase("frequencies", [tension(), torsion(frequency=2.0)]).history()
    assert len(history) == 129
    assert history.time[-1] == pytest.approx(1.0, rel=RTOL)
    np.testing.assert_allclose(np.diff(history.time), 0.5 / 64, rtol=RTOL)
    np.testing.assert_allclose(
        history.get("torsion").values,
        115.0 * np.sin(4 * np.pi * history.time),
        rtol=RTOL,
        atol=ATOL,
    )


def test_case7_force_sequence() -> None:
    """A force sequence is returned unchanged, at its own instants."""
    history = LoadCase("sequence", [force_sequence()]).history()
    force = history.get("force")
    np.testing.assert_array_equal(force.values, [0.0, 5000.0, -2000.0, 8000.0, 0.0])
    np.testing.assert_allclose(history.time, [0.0, 0.1, 0.2, 0.3, 0.4], rtol=RTOL)
    assert force.unit == "N"
    assert history.is_sequence is True


# -- time axis ----------------------------------------------------------------


def test_pragtic_example_time_axis() -> None:
    """Periods 1.5 s and 2 s with N = 4: 17 samples, dt = 0.375 s."""
    load_case = LoadCase(
        "pragtic",
        [
            tension(frequency=1 / 1.5),
            torsion(frequency=0.5),
        ],
        samples_per_period=4,
    )
    time = load_case.time_axis()
    assert time.size == 17
    np.testing.assert_allclose(np.diff(time), 0.375, rtol=RTOL)
    assert time[-1] == pytest.approx(6.0, rel=RTOL)


def test_last_periodic_sample_equals_first() -> None:
    """The time axis closes the cycle: last sample = first sample."""
    history = LoadCase(
        "closed", [tension(mean=100.0, phase=30.0), torsion(frequency=2.0, phase=45.0)]
    ).history()
    np.testing.assert_allclose(
        history.values_array()[-1], history.values_array()[0], rtol=RTOL
    )


def test_static_only_load_case_has_single_instant() -> None:
    """Only CONSTANT channels: the time axis is [0.0]."""
    history = LoadCase("static", [static()]).history()
    np.testing.assert_array_equal(history.time, [0.0])
    np.testing.assert_array_equal(history.get("preload").values, [30.0])


def test_static_channel_with_periodic_channels() -> None:
    """A static channel keeps its value over the periodic cycle."""
    history = LoadCase("preloaded", [tension(), static()]).history()
    np.testing.assert_array_equal(history.get("preload").values, 30.0)
    assert len(history) == DEFAULT_SAMPLES_PER_PERIOD + 1


# -- LoadHistory --------------------------------------------------------------


def test_values_array_and_table(out_of_phase: LoadHistory) -> None:
    """values_array stacks channels as columns; to_table adds time and units."""
    array = out_of_phase.values_array()
    assert array.shape == (len(out_of_phase), 2)
    np.testing.assert_array_equal(array[:, 1], out_of_phase.get("torsion").values)
    table = out_of_phase.to_table()
    assert list(table) == ["time [s]", "tension [MPa]", "torsion [MPa]"]
    np.testing.assert_array_equal(table["time [s]"], out_of_phase.time)


def test_to_voigt_stress_fills_only_columns_0_and_5(
    out_of_phase: LoadHistory,
) -> None:
    """Case 4: s11 goes to column 0, s12 to column 5, the rest stays zero."""
    stress = out_of_phase.to_voigt_stress()
    assert stress.shape == (len(out_of_phase), 6)
    np.testing.assert_array_equal(stress[:, 0], out_of_phase.get("tension").values)
    np.testing.assert_array_equal(stress[:, 5], out_of_phase.get("torsion").values)
    np.testing.assert_array_equal(stress[:, 1:5], 0.0)
    voigt.check_shape(stress)


def test_to_voigt_stress_ignores_non_stress_channels() -> None:
    """Force channels do not enter the Voigt stress array."""
    seq = VariableAmplitudeSignal(np.arange(5.0), time_step=0.1)
    stress_channel = Channel("s", Quantity.STRESS, "s33", seq)
    history = LoadCase("mixed", [force_sequence(), stress_channel]).history()
    stress = history.to_voigt_stress()
    np.testing.assert_array_equal(stress[:, 2], np.arange(5.0))
    np.testing.assert_array_equal(stress[:, [0, 1, 3, 4, 5]], 0.0)


# -- Quantity / voigt_index ---------------------------------------------------


@pytest.mark.parametrize(
    ("component", "index"),
    [
        ("s11", 0),
        ("s22", 1),
        ("s33", 2),
        ("s23", 3),
        ("s13", 4),
        ("s12", 5),
        ("e11", 0),
        ("e12", 5),
        ("Fx", None),
        ("Mz", None),
    ],
)
def test_voigt_index(component: str, index: int | None) -> None:
    """Voigt positions follow fatpy.utils.voigt; forces and moments have none."""
    assert voigt_index(component) == index


def test_quantity_units() -> None:
    """Each quantity has its fixed unit."""
    assert Quantity.STRESS.unit == "MPa"
    assert Quantity.STRAIN.unit == "mm/mm"
    assert Quantity.FORCE.unit == "N"
    assert Quantity.MOMENT.unit == "N*mm"


# -- error conditions ---------------------------------------------------------


def test_voigt_index_unknown_component_raises() -> None:
    """A component that belongs to no quantity."""
    with pytest.raises(LoadDefinitionError, match="Unknown"):
        voigt_index("s99")


def test_channel_incompatible_component_raises() -> None:
    """A component that does not belong to the channel quantity."""
    with pytest.raises(LoadDefinitionError, match="not a force component"):
        Channel("bad", Quantity.FORCE, "s11", tension().signal)


def test_load_case_without_channel_raises() -> None:
    """A load case needs at least one channel."""
    with pytest.raises(LoadDefinitionError, match="no channel"):
        LoadCase("empty", [])


def test_load_case_duplicate_names_raises() -> None:
    """Channel names must be unique."""
    with pytest.raises(LoadDefinitionError, match="duplicate channel names"):
        LoadCase("dup", [tension(), tension()])


def test_load_case_mixed_periodic_and_sequence_raises() -> None:
    """Periodic and variable-amplitude channels cannot be mixed."""
    with pytest.raises(LoadDefinitionError, match="mixes"):
        LoadCase("mixed", [tension(), force_sequence()]).time_axis()


def test_load_case_sequences_with_different_instants_raises() -> None:
    """Variable-amplitude channels must share their time instants."""
    other = VariableAmplitudeSignal(np.zeros(5), time_step=0.2)
    channel = Channel("other", Quantity.FORCE, "Fy", other)
    with pytest.raises(LoadDefinitionError, match="same time instants"):
        LoadCase("seq", [force_sequence(), channel]).time_axis()


def test_load_case_too_long_cycle_raises() -> None:
    """Nearly incommensurate periods are refused by the signal module."""
    load_case = LoadCase("long", [tension(), torsion(frequency=1.001)])
    with pytest.raises(SignalError, match="incommensurate"):
        load_case.history()


def test_to_voigt_stress_duplicate_component_raises() -> None:
    """Two channels cannot give the same stress component."""
    second = Channel("tension 2", Quantity.STRESS, "s11", tension().signal)
    history = LoadCase("dup", [tension(), second]).history()
    with pytest.raises(LoadDefinitionError, match="given by both"):
        history.to_voigt_stress()


def test_get_unknown_channel_raises(out_of_phase: LoadHistory) -> None:
    """get() of a channel that does not exist."""
    with pytest.raises(KeyError, match="bending"):
        out_of_phase.get("bending")


def test_load_definition_error_is_value_error() -> None:
    """LoadDefinitionError can be caught as a ValueError."""
    assert issubclass(LoadDefinitionError, ValueError)


# -- immutability ---------------------------------------------------------------


def test_dataclasses_are_immutable(out_of_phase: LoadHistory) -> None:
    """Channel, LoadCase, ChannelHistory and LoadHistory are frozen."""
    channel = tension()
    load_case = LoadCase("case", [channel])
    channel_history: ChannelHistory = out_of_phase.channels[0]
    for obj, attribute in (
        (channel, "name"),
        (load_case, "name"),
        (channel_history, "name"),
        (out_of_phase, "load_case_name"),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, attribute, "changed")
    assert isinstance(load_case.channels, tuple)
