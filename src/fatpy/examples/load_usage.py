"""Usage of the load definition module: the 7 reference load cases.

Builds the cases from independent channels, prints the out-of-phase case as a
table, shows the Voigt stress and strain exports and saves one figure per case
in ``output/load_cases/`` next to this file. Run it with:

    python -m fatpy.examples.load_usage
"""

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fatpy.data_parsing.loads import (  # noqa: E402
    TIME_UNIT,
    Channel,
    LoadCase,
    LoadHistory,
    Quantity,
)
from fatpy.utils.signal import (  # noqa: E402
    ConstantAmplitudeSignal,
    VariableAmplitudeSignal,
)

if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT = Path(__file__).parent / "output" / "load_cases"


def sine(
    amplitude: float, frequency: float = 1.0, mean: float = 0.0, phase: float = 0.0
) -> ConstantAmplitudeSignal:
    """Sine signal with the given amplitude, frequency [Hz], mean and phase [deg]."""
    return ConstantAmplitudeSignal(amplitude, mean, frequency=frequency, phase=phase)


# Independent channels: every case below is built from these pieces.
s11 = Channel("s11", Quantity.STRESS, "s11", sine(200.0))
s12 = Channel("s12", Quantity.STRESS, "s12", sine(115.0))
s12_90 = Channel("s12", Quantity.STRESS, "s12", sine(115.0, phase=90.0))
s11_mean = Channel("s11", Quantity.STRESS, "s11", sine(200.0, mean=100.0))
s12_2hz = Channel("s12", Quantity.STRESS, "s12", sine(115.0, frequency=2.0))
fx_values = np.array([0.0, 5000.0, -2000.0, 8000.0, 0.0])
fx = Channel(
    "Fx", Quantity.FORCE, "Fx", VariableAmplitudeSignal(fx_values, time_step=0.1)
)

CASES = [
    LoadCase("1 tension", [s11]),
    LoadCase("2 torsion", [s12]),
    LoadCase("3 in-phase", [s11, s12]),
    LoadCase("4 out-of-phase", [s11, s12_90]),
    LoadCase("5 mean stress", [s11_mean]),
    LoadCase("6 different frequencies", [s11, s12_2hz]),
    LoadCase("7 force history", [fx]),
]
PATH_PLOT = {"3 in-phase", "4 out-of-phase", "6 different frequencies"}


def print_table(history: LoadHistory) -> None:
    """Print every sample of a history, one column per channel."""
    table = history.to_table()
    print("".join(f"{key:>14}" for key in table))
    for row in zip(*table.values(), strict=True):
        print("".join(f"{value:14.3f}" for value in row))


def save_figure(history: LoadHistory) -> Path:
    """Save the value vs time of each channel (and the s11-s12 path)."""
    path_plot = history.load_case_name in PATH_PLOT
    fig, axes = plt.subplots(
        1, 2 if path_plot else 1, figsize=(11 if path_plot else 6, 4), squeeze=False
    )
    ax = axes[0, 0]
    for channel in history.channels:
        ax.plot(history.time, channel.values, marker=".", label=channel.name)
    units = sorted({c.unit for c in history.channels})
    ax.set(
        xlabel=f"time [{TIME_UNIT}]",
        ylabel=f"value [{', '.join(units)}]",
        title=f"Case {history.load_case_name}: channels vs time",
    )
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.legend()
    ax.grid(True)
    if path_plot:
        ax = axes[0, 1]
        ax.plot(history.get("s11").values, history.get("s12").values)
        ax.set(
            xlabel="s11 [MPa]",
            ylabel="s12 [MPa]",
            title=f"Case {history.load_case_name}: s11-s12 path",
            aspect="equal",
        )
        ax.grid(True)
    fig.tight_layout()
    path = OUTPUT / f"case_{history.load_case_name.replace(' ', '_')}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


OUTPUT.mkdir(parents=True, exist_ok=True)
for case in CASES:
    history = case.history()
    print(
        f"Case {case.name:<26} {len(history):4d} samples, "
        f"t_end = {history.time[-1]:g} s -> {save_figure(history).name}"
    )
print(f"Figures saved in {OUTPUT}")

print("\nCase 4 (out-of-phase), N = 8 samples per period:")
case4 = LoadCase("4 out-of-phase", [s11, s12_90], samples_per_period=8).history()
print_table(case4)

print("\nto_voigt_stress() of case 4, columns 11, 22, 33, 23, 13, 12 [MPa]:")
print(np.round(case4.to_voigt_stress(), 3))

strain_case = LoadCase(
    "out-of-phase strain",
    [
        Channel("e11", Quantity.STRAIN, "e11", sine(1e-3)),
        Channel("e12", Quantity.STRAIN, "e12", sine(0.65e-3, phase=90.0)),
    ],
    samples_per_period=8,
)
print("\nto_voigt_strain() of the same loading in strain [mm/mm]:")
with np.printoptions(precision=6, suppress=True):
    print(strain_case.history().to_voigt_strain())
