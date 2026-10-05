"""Load data parsing module.

Definition of the loading conditions applied to a component, inspired by the
load definition of PragTic (J. Papuga).

Overview:
    - A `Channel` gives one component of one physical `Quantity` (stress,
      strain, force or moment) a time `fatpy.utils.signal.Signal`.
    - A `LoadCase` groups the channels acting together. It is either:
        - periodic: constant-amplitude channels evaluated over their common
          period (least common multiple of the channel periods), or
        - a load sequence: variable-amplitude channels sharing the same time
          instants.
      Static (``CONSTANT`` waveform) channels can be added to both.
    - `LoadCase.history` samples every channel on the common time axis and
      returns a `LoadHistory`, which `LoadHistory.to_voigt_stress` turns into
      the ``(n, 6)`` Voigt stress array used by FatPy methods.

Conventions:
    - Units are fixed by the quantity (see `Quantity.unit`); values must be
      given in these units, nothing is converted.
    - Stress and strain components follow the Voigt order of
      `fatpy.utils.voigt`: (11, 22, 33, 23, 13, 12).
    - Shear strains are tensor components (ε_12 = γ_12 / 2).
    - Force and moment channels are external loads; they have no Voigt
      position.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fatpy.utils import voigt
from fatpy.utils.signal import (
    TIME_PRECISION,
    Signal,
    VariableAmplitudeSignal,
    common_time_axis,
)

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


@dataclass(frozen=True)
class Channel:
    """One loaded component and its time signal.

    Attributes:
        name: Channel name, unique within a load case.
        quantity: Physical quantity of the channel.
        component: Component of `quantity` (one of `Quantity.components`).
        signal: Time signal of the channel, in the unit of `quantity`.
    """

    name: str
    quantity: Quantity
    component: str
    signal: Signal

    def __post_init__(self) -> None:
        """Check that the component belongs to the quantity.

        Raises:
            LoadDefinitionError: If `component` is not a component of
                `quantity`.
        """
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
        values: Sampled values, shape (n,).
    """

    name: str
    quantity: Quantity
    component: str
    unit: str
    values: NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class LoadHistory:
    """Sampled load case: every channel evaluated on a common time axis.

    Attributes:
        time: Time instants [s], shape (n,).
        channels: Sampled channels, in the order of the load case.
        load_case_name: Name of the load case the history comes from.
        is_sequence: ``True`` for a load sequence (variable-amplitude
            channels at their own instants), ``False`` for one common period
            of periodic channels (or a single static instant).
    """

    time: NDArray[np.float64]
    channels: tuple[ChannelHistory, ...]
    load_case_name: str
    is_sequence: bool

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

        The result can be passed directly to ``pandas.DataFrame``.
        """
        table = {"time [s]": self.time}
        table.update({f"{c.name} [{c.unit}]": c.values for c in self.channels})
        return table

    def to_voigt_stress(self) -> NDArray[np.float64]:
        """Stress history in Voigt notation, as used by FatPy methods.

        Stress channels fill their Voigt column; other columns stay zero and
        non-stress channels are ignored.

        Returns:
            Array of shape (n, 6) [MPa].

        Raises:
            LoadDefinitionError: If two channels give the same stress
                component.
        """
        stress = np.zeros((len(self), voigt.VOIGT_COMPONENTS_COUNT))
        source: dict[int, str] = {}
        for channel in self.channels:
            if channel.quantity is not Quantity.STRESS:
                continue
            index = _VOIGT_INDEX[channel.component]
            if index in source:
                raise LoadDefinitionError(
                    f"Stress component {channel.component!r} is given by both "
                    f"{source[index]!r} and {channel.name!r}"
                )
            source[index] = channel.name
            stress[:, index] = channel.values
        return stress


@dataclass(frozen=True)
class LoadCase:
    """Channels acting together on the component.

    Attributes:
        name: Load case name.
        channels: Channels of the load case (stored as a tuple).
        samples_per_period: Number of time steps in the shortest period of
            the periodic channels.
    """

    name: str
    channels: Sequence[Channel]
    samples_per_period: int = DEFAULT_SAMPLES_PER_PERIOD

    def __post_init__(self) -> None:
        """Store the channels as a tuple and validate them.

        Raises:
            LoadDefinitionError: If there is no channel or if channel names
                are not unique.
        """
        object.__setattr__(self, "channels", tuple(self.channels))
        if not self.channels:
            raise LoadDefinitionError(f"Load case {self.name!r} has no channel")
        names = [c.name for c in self.channels]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise LoadDefinitionError(
                f"Load case {self.name!r}: duplicate channel names {duplicates}"
            )

    @property
    def is_sequence(self) -> bool:
        """``True`` if the load case holds variable-amplitude channels."""
        return any(isinstance(c.signal, VariableAmplitudeSignal) for c in self.channels)

    def time_axis(self) -> NDArray[np.float64]:
        """Common time instants of all channels.

        - Periodic channels: one common period, see
          `fatpy.utils.signal.common_time_axis`.
        - Variable-amplitude channels: their shared time instants.
        - Only static (``CONSTANT``) channels: the single instant ``[0.0]``.

        Returns:
            Array of shape (n,) [s].

        Raises:
            LoadDefinitionError: If periodic and variable-amplitude channels
                are mixed, or if variable-amplitude channels do not share the
                same time instants.
            SignalError: If the common period of the periodic channels is
                too long (see `MAX_CYCLE_IN_SHORTEST_PERIODS`).
        """
        sequences = [
            c.signal
            for c in self.channels
            if isinstance(c.signal, VariableAmplitudeSignal)
        ]
        periods = [
            period
            for c in self.channels
            if (period := c.signal.effective_period) is not None
        ]
        if sequences and periods:
            raise LoadDefinitionError(
                f"Load case {self.name!r} mixes periodic and variable-amplitude "
                "channels"
            )
        if sequences:
            time = sequences[0].instants
            for signal in sequences[1:]:
                if signal.instants.shape != time.shape or not np.allclose(
                    signal.instants, time, rtol=0.0, atol=TIME_PRECISION
                ):
                    raise LoadDefinitionError(
                        f"Load case {self.name!r}: variable-amplitude channels "
                        "must share the same time instants"
                    )
            return time.copy()
        if periods:
            return common_time_axis(
                periods, self.samples_per_period, MAX_CYCLE_IN_SHORTEST_PERIODS
            )
        return np.zeros(1)

    def history(self) -> LoadHistory:
        """Sample every channel on the common time axis.

        Returns:
            The sampled `LoadHistory`.

        Raises:
            LoadDefinitionError: See `time_axis`.
            SignalError: See `time_axis`.
        """
        time = self.time_axis()
        channels = tuple(
            ChannelHistory(c.name, c.quantity, c.component, c.unit, c.evaluate(time))
            for c in self.channels
        )
        return LoadHistory(time, channels, self.name, self.is_sequence)
