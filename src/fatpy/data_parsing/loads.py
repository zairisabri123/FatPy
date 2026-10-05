"""Load data parsing module.

Definition of the loading conditions applied to a component, following the
load definition of PragTic (J. Papuga).

Overview:
    - A `Channel` gives one component of one physical `Quantity` (stress,
      strain, force or moment) a time `fatpy.utils.signal.Signal`. Channels
      are defined independently, so uniaxial (tension, torsion) and
      multiaxial (in-phase, out-of-phase) loadings are built from the same
      pieces.
    - A `LoadCase` groups the channels acting together. Its time axis is
      either:
        - one common period of the periodic (constant-amplitude) channels,
          see `fatpy.utils.signal.common_time_axis`, or
        - the time instants shared by the variable-amplitude channels.
      Static (``CONSTANT`` waveform) channels can be added to both.
    - `LoadCase.history` samples every channel on that time axis and returns
      a `LoadHistory` (value vs time of each channel), which
      `LoadHistory.to_voigt_stress` and `LoadHistory.to_voigt_strain` turn
      into the ``(n, 6)`` Voigt arrays used by FatPy methods.
    - A load case is a load sequence (`LoadCase.is_sequence`) when one of its
      variable-amplitude channels has no time scale (values only, see
      `fatpy.utils.signal.VariableAmplitudeSignal.is_sequence`).

PragTic mapping:
    - load regime -> `LoadCase`
    - load channel -> `Channel`
    - load defined by a mathematical formula -> `ConstantAmplitudeSignal`
    - load read from a file -> `VariableAmplitudeSignal`

Conventions:
    - Units are fixed by the quantity (see `Quantity.unit`) and time is in
      seconds (`TIME_UNIT`); values must be given in these units, nothing is
      converted.
    - Stress and strain components follow the Voigt order of
      `fatpy.utils.voigt`: (11, 22, 33, 23, 13, 12).
    - Shear strains are tensor components (ε_12 = γ_12 / 2), as in
      `fatpy.struct_mech.strain`.
    - Force and moment channels are external loads; they have no Voigt
      position.

Future work:
    - PSD loading (a new `Signal` class, no change to `Channel` or
      `LoadCase`).
    - Loading sequences with repetitions of load cases.
    - Conversion of force and moment channels to stresses.
    - Load spectra.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fatpy.utils import voigt
from fatpy.utils.signal import (
    MIN_SAMPLES_PER_PERIOD,
    TIME_PRECISION,
    Signal,
    SignalError,
    VariableAmplitudeSignal,
    common_time_axis,
)

#: Unit of time.
TIME_UNIT = "s"
#: Default number of time steps in the shortest period of a load case.
DEFAULT_SAMPLES_PER_PERIOD = 64
#: Maximum length of the common period, as a number of shortest periods.
MAX_CYCLE_IN_SHORTEST_PERIODS = 1000


class LoadDefinitionError(ValueError):
    """Raised when a load is inconsistently defined."""


class Quantity(Enum):
    """Physical quantity carried by a channel."""

    STRESS = "stress"
    STRAIN = "strain"
    FORCE = "force"
    MOMENT = "moment"

    @property
    def components(self) -> tuple[str, ...]:
        """Component names allowed for this quantity."""
        return _COMPONENTS[self]

    @property
    def unit(self) -> str:
        """Unit the values of this quantity are given in."""
        return _UNITS[self]


_VOIGT_SUFFIXES = ("11", "22", "33", "23", "13", "12")  # order of fatpy.utils.voigt
_COMPONENTS: dict[Quantity, tuple[str, ...]] = {
    Quantity.STRESS: tuple(f"s{s}" for s in _VOIGT_SUFFIXES),
    Quantity.STRAIN: tuple(f"e{s}" for s in _VOIGT_SUFFIXES),
    Quantity.FORCE: ("Fx", "Fy", "Fz"),
    Quantity.MOMENT: ("Mx", "My", "Mz"),
}
_UNITS: dict[Quantity, str] = {
    Quantity.STRESS: "MPa",
    Quantity.STRAIN: "mm/mm",
    Quantity.FORCE: "N",
    Quantity.MOMENT: "N*mm",
}
_VOIGT_INDEX: dict[str, int] = {
    c: i for q in (Quantity.STRESS, Quantity.STRAIN) for i, c in enumerate(q.components)
}


def voigt_index(component: str) -> int | None:
    """Position of a stress or strain component in a Voigt vector.

    Args:
        component: Component name (e.g. ``s12``, ``e23``, ``Fx``).

    Returns:
        0..5 for the 11, 22, 33, 23, 13, 12 components, ``None`` for force
        and moment components.

    Raises:
        LoadDefinitionError: If `component` belongs to no quantity.
    """
    if component in _VOIGT_INDEX:
        return _VOIGT_INDEX[component]
    if component in Quantity.FORCE.components + Quantity.MOMENT.components:
        return None
    raise LoadDefinitionError(f"Unknown load component {component!r}")


def _read_only(values: ArrayLike) -> NDArray[np.float64]:
    """Read-only float64 copy of `values`."""
    array = np.array(values, dtype=np.float64)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Channel:
    """One loaded component and its time signal (a PragTic load channel).

    Attributes:
        name: Channel name, not empty and unique within a load case.
        quantity: Physical quantity of the channel.
        component: Component of `quantity` (one of `Quantity.components`).
        signal: Time signal of the channel, in the unit of `quantity`.
    """

    name: str
    quantity: Quantity
    component: str
    signal: Signal

    def __post_init__(self) -> None:
        """Check the name and that the component belongs to the quantity.

        Raises:
            LoadDefinitionError: If `name` is empty or `component` is not a
                component of `quantity`.
        """
        if not self.name.strip():
            raise LoadDefinitionError(
                f"Channel name must not be empty, got {self.name!r}"
            )
        if self.component not in self.quantity.components:
            raise LoadDefinitionError(
                f"Channel {self.name!r}: {self.component!r} is not a "
                f"{self.quantity.value} component {self.quantity.components}"
            )

    @property
    def unit(self) -> str:
        """Unit of the channel values."""
        return self.quantity.unit

    def evaluate(self, time: ArrayLike) -> NDArray[np.float64]:
        """Channel values at the given time instants [s]."""
        return self.signal.evaluate(time)


@dataclass(frozen=True, eq=False)
class ChannelHistory:
    """Values of one channel sampled on the time axis of a `LoadHistory`.

    Attributes:
        name: Channel name.
        quantity: Physical quantity of the channel.
        component: Component of `quantity`.
        unit: Unit of `values`.
        values: Sampled values, shape (n,), read-only.
    """

    name: str
    quantity: Quantity
    component: str
    unit: str
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Store `values` as a read-only float64 array."""
        object.__setattr__(self, "values", _read_only(self.values))


@dataclass(frozen=True, eq=False)
class LoadHistory:
    """Sampled load case: the value vs time of every channel.

    Attributes:
        time: Time instants [s], shape (n,), read-only.
        channels: Sampled channels, in the order of the load case.
        load_case_name: Name of the load case the history comes from.
        is_sequence: ``True`` for a load sequence (values without a time
            scale, `time` holds the indices 0, 1, 2, ...).
    """

    time: NDArray[np.float64]
    channels: tuple[ChannelHistory, ...]
    load_case_name: str
    is_sequence: bool

    def __post_init__(self) -> None:
        """Store `time` as a read-only float64 array and check the lengths.

        Raises:
            LoadDefinitionError: If a channel does not have one value per
                time instant.
        """
        object.__setattr__(self, "time", _read_only(self.time))
        object.__setattr__(self, "channels", tuple(self.channels))
        for channel in self.channels:
            if channel.values.shape != self.time.shape:
                raise LoadDefinitionError(
                    f"Channel {channel.name!r} has {channel.values.shape} values "
                    f"for {self.time.shape} time instants"
                )

    def __len__(self) -> int:
        """Number of time instants."""
        return int(self.time.size)

    def get(self, name: str) -> ChannelHistory:
        """Sampled channel by name.

        Args:
            name: Channel name.

        Returns:
            The matching `ChannelHistory`.

        Raises:
            KeyError: If there is no channel called `name`.
        """
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise KeyError(f"No channel {name!r} in {[c.name for c in self.channels]}")

    def values_array(self) -> NDArray[np.float64]:
        """Channel values as columns, shape (n, n_channels)."""
        return np.column_stack([c.values for c in self.channels])

    def to_table(self) -> dict[str, NDArray[np.float64]]:
        """Columns keyed by ``"time [s]"`` and ``"<name> [<unit>]"``.

        Returns a plain dict (FatPy does not depend on pandas types); it can
        be passed directly to ``pandas.DataFrame``.
        """
        table = {f"time [{TIME_UNIT}]": self.time}
        table.update({f"{c.name} [{c.unit}]": c.values for c in self.channels})
        return table

    def to_voigt_stress(self) -> NDArray[np.float64]:
        """Stress history in Voigt notation, as used by FatPy methods.

        Each stress channel fills its Voigt column; missing components stay
        zero.

        Returns:
            Array of shape (n, 6) [MPa].

        Raises:
            LoadDefinitionError: If a channel is not a stress, or if two
                channels give the same component.
        """
        return self._to_voigt(Quantity.STRESS)

    def to_voigt_strain(self) -> NDArray[np.float64]:
        """Strain history in Voigt notation, as used by FatPy methods.

        Each strain channel fills its Voigt column (e11 -> 0, ..., e12 -> 5);
        missing components stay zero. Shear values are tensor strains
        (ε_12 = γ_12 / 2), the convention of `fatpy.struct_mech.strain`.

        Returns:
            Array of shape (n, 6) [mm/mm].

        Raises:
            LoadDefinitionError: If a channel is not a strain, or if two
                channels give the same component.
        """
        return self._to_voigt(Quantity.STRAIN)

    def _to_voigt(self, quantity: Quantity) -> NDArray[np.float64]:
        """Fill a (n, 6) Voigt array from channels that must all be `quantity`."""
        array = np.zeros((len(self), voigt.VOIGT_COMPONENTS_COUNT))
        source: dict[int, str] = {}
        for channel in self.channels:
            if channel.quantity is not quantity:
                raise LoadDefinitionError(
                    f"Channel {channel.name!r} is {channel.quantity.value} "
                    f"({channel.component!r}), it cannot enter a Voigt "
                    f"{quantity.value} array"
                )
            index = _VOIGT_INDEX[channel.component]
            if index in source:
                raise LoadDefinitionError(
                    f"Component {channel.component!r} is given by both channels "
                    f"{source[index]!r} and {channel.name!r}"
                )
            source[index] = channel.name
            array[:, index] = channel.values
        return array


@dataclass(frozen=True)
class LoadCase:
    """Channels acting together on the component (a PragTic load regime).

    Attributes:
        name: Load case name, not empty.
        channels: Channels of the load case (stored as a tuple).
        samples_per_period: Number of time steps in the shortest period of
            the periodic channels, an integer, at least
            `fatpy.utils.signal.MIN_SAMPLES_PER_PERIOD`.
    """

    name: str
    channels: Sequence[Channel]
    samples_per_period: int = DEFAULT_SAMPLES_PER_PERIOD

    def __post_init__(self) -> None:
        """Store the channels as a tuple and validate the load case.

        Raises:
            LoadDefinitionError: If the name is empty, if there is no
                channel, if channel names are not unique, or if
                `samples_per_period` is not an integer >= 4.
        """
        object.__setattr__(self, "channels", tuple(self.channels))
        if not self.name.strip():
            raise LoadDefinitionError(
                f"Load case name must not be empty, got {self.name!r}"
            )
        if not self.channels:
            raise LoadDefinitionError(f"Load case {self.name!r} has no channel")
        names = [c.name for c in self.channels]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise LoadDefinitionError(
                f"Load case {self.name!r}: duplicate channel names {duplicates}"
            )
        if (
            not isinstance(self.samples_per_period, (int, np.integer))
            or self.samples_per_period < MIN_SAMPLES_PER_PERIOD
        ):
            raise LoadDefinitionError(
                f"Load case {self.name!r}: samples_per_period must be an integer >= "
                f"{MIN_SAMPLES_PER_PERIOD}, got {self.samples_per_period!r}"
            )

    @property
    def is_sequence(self) -> bool:
        """``True`` if a variable-amplitude channel has no time scale."""
        return any(
            isinstance(c.signal, VariableAmplitudeSignal) and c.signal.is_sequence
            for c in self.channels
        )

    def time_axis(self) -> NDArray[np.float64]:
        """Common time instants of all channels.

        - Periodic channels: one common period, see
          `fatpy.utils.signal.common_time_axis`.
        - Channels with their own instants (variable amplitude): their shared
          instants. Static (``CONSTANT``) channels may be added; they keep
          their value at every instant.
        - Only static channels: the single instant ``[0.0]``.

        Returns:
            Array of shape (n,) [s].

        Raises:
            LoadDefinitionError: If periodic channels are mixed with channels
                that have their own instants, or if the latter do not share
                the same instants.
            SignalError: If the common period of the periodic channels is
                too long (see `MAX_CYCLE_IN_SHORTEST_PERIODS`).
        """
        sampled = [
            (c.name, instants)
            for c in self.channels
            if (instants := c.signal.instants) is not None
        ]
        periodic = [
            (c.name, period)
            for c in self.channels
            if (period := c.signal.effective_period) is not None
        ]
        if sampled and periodic:
            raise LoadDefinitionError(
                f"Load case {self.name!r} mixes periodic channels "
                f"{[n for n, _ in periodic]} with variable-amplitude channels "
                f"{[n for n, _ in sampled]}"
            )
        if sampled:
            first_name, time = sampled[0]
            for name, instants in sampled[1:]:
                if instants.shape != time.shape or not np.allclose(
                    instants, time, rtol=0.0, atol=TIME_PRECISION
                ):
                    raise LoadDefinitionError(
                        f"Load case {self.name!r}: channel {name!r} does not "
                        f"share the time instants of channel {first_name!r}"
                    )
            return time.copy()
        if periodic:
            return common_time_axis(
                [p for _, p in periodic],
                self.samples_per_period,
                MAX_CYCLE_IN_SHORTEST_PERIODS,
            )
        return np.zeros(1)

    def history(self) -> LoadHistory:
        """Sample every channel on the common time axis.

        Returns:
            The sampled `LoadHistory`.

        Raises:
            LoadDefinitionError: See `time_axis`, or if a channel signal
                cannot be evaluated (the error names the channel).
            SignalError: See `time_axis`.
        """
        time = self.time_axis()
        channels = []
        for c in self.channels:
            try:
                values = c.evaluate(time)
            except SignalError as err:
                raise LoadDefinitionError(
                    f"Load case {self.name!r}, channel {c.name!r}: {err}"
                ) from err
            channels.append(
                ChannelHistory(c.name, c.quantity, c.component, c.unit, values)
            )
        return LoadHistory(time, tuple(channels), self.name, self.is_sequence)
