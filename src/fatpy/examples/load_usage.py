"""Usage of the load definition module: the 7 reference load cases.

Prints, for each case, the time axis and a few samples of every channel, so
the behavior of `fatpy.data_parsing.loads` can be read from the console.
Run it with:

    python -m fatpy.examples.load_usage
"""

import io
import sys

import numpy as np

from fatpy.data_parsing.loads import Channel, LoadCase, LoadHistory, Quantity
from fatpy.struct_mech.stress import calc_von_mises_stress
from fatpy.utils.signal import ConstantAmplitudeSignal, VariableAmplitudeSignal

if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def sep(title: str) -> None:
    """Print a banner around a section title."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def show(history: LoadHistory, every: int = 8) -> None:
    """Print the time axis summary and every `every`-th sample as a table."""
    kind = "load sequence" if history.is_sequence else "one common period"
    dt = history.time[1] - history.time[0] if len(history) > 1 else 0.0
    print(
        f"{history.load_case_name}: {len(history)} samples ({kind}), "
        f"dt = {dt:g} s, t_end = {history.time[-1]:g} s"
    )
    table = history.to_table()
    print("".join(f"{key:>16}" for key in table))
    for i in range(0, len(history), every):
        print("".join(f"{column[i]:16.3f}" for column in table.values()))


def stress(name: str, component: str, signal: ConstantAmplitudeSignal) -> Channel:
    """Stress channel (MPa) carrying a constant-amplitude signal."""
    return Channel(name, Quantity.STRESS, component, signal)


# ---------------------------------------------------------------------------
sep("1. Tension: s11, sine, 200 MPa, 1 Hz")
tension = stress(
    "tension", "s11", ConstantAmplitudeSignal(amplitude=200.0, frequency=1.0)
)
show(LoadCase("tension", [tension]).history())

# ---------------------------------------------------------------------------
sep("2. Torsion: s12, sine, 115 MPa, 1 Hz")
torsion = stress(
    "torsion", "s12", ConstantAmplitudeSignal(amplitude=115.0, frequency=1.0)
)
show(LoadCase("torsion", [torsion]).history())

# ---------------------------------------------------------------------------
sep("3. In-phase tension + torsion")
in_phase = LoadCase("in-phase", [tension, torsion]).history()
show(in_phase)
ratio = in_phase.get("torsion").values[1:] / in_phase.get("tension").values[1:]
print(f"s12 / s11 is constant: {np.allclose(ratio, 115 / 200)} (= {115 / 200:g})")

# ---------------------------------------------------------------------------
sep("4. Out-of-phase: s12 shifted by 90 deg")
torsion_90 = stress(
    "torsion",
    "s12",
    ConstantAmplitudeSignal(amplitude=115.0, frequency=1.0, phase=90.0),
)
out_of_phase = LoadCase("out-of-phase", [tension, torsion_90]).history()
show(out_of_phase)
s11 = out_of_phase.get("tension").values
s12 = out_of_phase.get("torsion").values
ellipse = (s11 / 200.0) ** 2 + (s12 / 115.0) ** 2
print(
    f"(s11/200)^2 + (s12/115)^2: min = {ellipse.min():.12f}, max = {ellipse.max():.12f}"
)

voigt_stress = out_of_phase.to_voigt_stress()
print(f"\nto_voigt_stress(): shape {voigt_stress.shape}, columns 11,22,33,23,13,12")
print("first row :", voigt_stress[0])
print("max |col| :", np.abs(voigt_stress).max(axis=0))
von_mises = calc_von_mises_stress(voigt_stress)
print(f"von Mises over the cycle: {von_mises.min():.1f} .. {von_mises.max():.1f} MPa")

# ---------------------------------------------------------------------------
sep("5. Mean stress: mean 100 MPa, amplitude 200 MPa")
with_mean = stress(
    "tension",
    "s11",
    ConstantAmplitudeSignal(amplitude=200.0, mean=100.0, frequency=1.0),
)
history = LoadCase("mean stress", [with_mean]).history()
show(history)
values = history.get("tension").values
print(f"max = {values.max():g} MPa, min = {values.min():g} MPa")

# ---------------------------------------------------------------------------
sep("6. Different frequencies: s11 at 1 Hz, s12 at 2 Hz")
torsion_2hz = stress(
    "torsion", "s12", ConstantAmplitudeSignal(amplitude=115.0, frequency=2.0)
)
history = LoadCase("frequencies", [tension, torsion_2hz]).history()
show(history, every=16)
print(f"common cycle = {history.time[-1]:g} s, {len(history)} samples")

# ---------------------------------------------------------------------------
sep("7. Force sequence: Fx = 0, 5000, -2000, 8000, 0 N, dt = 0.1 s")
fx_values = np.array([0.0, 5000.0, -2000.0, 8000.0, 0.0])
force = Channel(
    "force", Quantity.FORCE, "Fx", VariableAmplitudeSignal(fx_values, time_step=0.1)
)
history = LoadCase("force sequence", [force]).history()
show(history, every=1)
print("output equals input:", np.array_equal(history.get("force").values, fx_values))
