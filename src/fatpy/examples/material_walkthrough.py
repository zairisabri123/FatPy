"""Runnable walkthrough of every public function in `fatpy.data_parsing.material`.

Prints each step's result so the behavior can be read directly from the
console. Run it with:

    python -m fatpy.examples.material_walkthrough
"""

import io
import sys
import tempfile
from pathlib import Path

from fatpy.data_parsing.material import NOMENCLATURE, Material, SNCurve

if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # nomenclature symbols use non-ASCII


def sep(title: str) -> None:
    """Print a banner around a section title."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
sep("1. Creating a material directly (42CrMo4), values in nomenclature units")
steel = Material(
    material_name="42CrMo4",
    source="FatPy example",
    elastic_modulus=206_000,  # MPa
    poissons_ratio=0.3,
    yield_strength=980,
    ult_tensile_strength=1100,
    fat_lim_ten_m1=488,
    fat_lim_tor_m1=404,
    fat_strength_coef=1700,
    fat_strength_exp=-0.09,
    fat_ductility_coef=0.6,
    fat_ductility_exp=-0.6,
    cyc_hard_coef=1400,
    cyc_hard_exp=0.12,
)
print(steel.summary())
print("\nrepr(steel) =", repr(steel))

# ---------------------------------------------------------------------------
sep("2. Reading parameters: plain attributes, None when unknown")
print("steel.fat_lim_ten_m1 =", steel.fat_lim_ten_m1)
print("steel.walker_exp     =", steel.walker_exp, "(not defined)")
print("Checking required parameters is left to each fatigue method.")

# ---------------------------------------------------------------------------
sep("3. describe(): parameter metadata (WG6 spelling accepted)")
spec = Material.describe("fat_lim_tor_-1")
print(spec)
print(f"{spec.name} -> symbol {spec.symbol}, unit {spec.unit}")

# ---------------------------------------------------------------------------
sep("4. S-N curve defined directly from its parameters")
sn_ten = SNCurve(
    description="push-pull, R=-1, median",
    fatigue_limit=488,
    exponent=10,
    n_cycles_at_limit=2e6,
    load_mode="ten",
    stress_ratio=-1,
)
steel.sn_curves.append(sn_ten)
print(f"C = {sn_ten.coefficient:.4e}")
for n in (1e4, 1e5, 1e6, 2e6, 1e8):
    print(f"  sigma_a({n:.0e}) = {sn_ten.stress_amp_at(n):7.1f} MPa")
for s in (800, 600, 500, 488, 400):
    print(f"  N({s} MPa) = {sn_ten.cycles_at(s):.3e}")

# ---------------------------------------------------------------------------
sep("5. S-N curve fitted from test points (torsion)")
tau = [620, 580, 540, 500, 470, 440, 420, 410, 400]
n_f = [2.1e4, 5.0e4, 1.2e5, 3.1e5, 7.5e5, 1.6e6, 3.0e6, 1e7, 1e7]
runout = [False] * 7 + [True, True]  # two runouts
sn_tor = SNCurve.from_test_points(
    "torsion, R=-1, batch A", tau, n_f, runout, n_cycles_at_limit=2e6, load_mode="tor"
)
steel.sn_curves.append(sn_tor)
print(f"Fitted FL     = {sn_tor.fatigue_limit:.1f} MPa")
print(f"Exponent w    = {sn_tor.exponent:.2f}  (slope = {-1 / sn_tor.exponent:.4f})")
print(f"Log stdev     = {sn_tor.stdev_log_stress:.4f}")
print(f"N(550 MPa)    = {sn_tor.cycles_at(550):.3e}")

print("\nsn_curve('tor') ->", steel.sn_curve("tor"))
print("sn_curve('ben') ->", steel.sn_curve("ben"))

# ---------------------------------------------------------------------------
sep("6. at_survival_prob(): design curve from the median curve's scatter")
for ps in (0.5, 0.975, 0.10):
    shifted = sn_tor.at_survival_prob(ps)
    print(f"Ps = {ps:<5}: FL = {shifted.fatigue_limit:.1f} MPa")

# ---------------------------------------------------------------------------
sep("7. Error cases (only where a wrong result would otherwise be silent)")
try:
    SNCurve.from_test_points("1 failure", [500, 400], [1e5, 1e7], [False, True])
except ValueError as e:
    print("Not enough test points :", e)
try:
    SNCurve.from_test_points("wrong order", [400, 500], [1e5, 1e7])
except ValueError as e:
    print("Stress not decreasing  :", e)

# ---------------------------------------------------------------------------
sep("8. from_dict(): one row of a WG6 Excel table")
row = {
    "material_name": "C45",
    "yield_strength": 430,
    "ult_tensile_strength": 700,
    "fat_lim_ten_-1": 300,  # WG6 key with -1
    "fat_lim_tor_B-1": 210,  # Excel file's own spelling variant
    "cyc_hard_coeff": 1100,  # WG6 alias
    "RBEND": 320,  # proposed parameter
    "hardness": None,  # ignored
    "test_temperature": 20,  # unknown -> extra
}
c45 = Material.from_dict(row)
print(c45.summary())
print("extra :", c45.extra)

# ---------------------------------------------------------------------------
sep("9. to_dict() / from_dict() round trip (with an S-N curve)")
c45.sn_curves.append(sn_tor)
d = c45.to_dict(include_sn_curves=True)
print("keys :", sorted(d))
print("Round trip identical :", Material.from_dict(d) == c45)

# ---------------------------------------------------------------------------
sep("10. to_json() / from_json(): persistence to a file")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "c45.json"
    c45.to_json(path)
    print(path.read_text(encoding="utf-8")[:200], "...")
    print("Loaded back identical :", Material.from_json(path) == c45)

# ---------------------------------------------------------------------------
sep("11. Nomenclature")
groups: dict[str, list[str]] = {}
for entry in NOMENCLATURE.values():
    groups.setdefault(entry.group, []).append(entry.key)
for group_name, keys in groups.items():
    print(f"{group_name:<15} {len(keys):2d} parameters")
print("Total :", len(NOMENCLATURE))
print(
    "Proposed (not yet in the Excel file) :",
    [k for k, s in NOMENCLATURE.items() if s.proposed],
)
