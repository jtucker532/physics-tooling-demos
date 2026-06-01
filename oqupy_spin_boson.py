"""
oqupy_spin_boson.py
===================

Non-Markovian open-system dynamics of a spin-boson model using OQuPy's
TEMPO (Time-Evolving Matrix Product Operator) algorithm.

Physics
-------
A single two-level system (the "spin") is driven coherently and coupled to a
bosonic bath with an Ohmic spectral density. Unlike a Lindblad master equation,
TEMPO captures the *non-Markovian* memory of the bath exactly (to a controllable
numerical tolerance), so it resolves bath-induced effects that a memoryless
treatment misses.

    System:   H = (Omega / 2) * sigma_x          (coherent driving)
    Coupling: sigma_z  to an Ohmic bath, J(w) ~ alpha * w * exp(-w / cutoff)

We start in the spin-up state and track <sigma_z>(t), watching coherent Rabi
oscillation get damped by the bath.

Run:
    python oqupy_spin_boson.py
"""

import numpy as np
import matplotlib.pyplot as plt
import oqupy

# ----------------------------------------------------------------------
# 1. System Hamiltonian: coherent drive about the x-axis
# ----------------------------------------------------------------------
Omega = 1.0  # Rabi (driving) frequency
system = oqupy.System(0.5 * Omega * oqupy.operators.sigma("x"))

# ----------------------------------------------------------------------
# 2. Bath: Ohmic spectral density, coupled through sigma_z
# ----------------------------------------------------------------------
alpha = 0.3          # dimensionless system-bath coupling strength
cutoff = 5.0         # bath cutoff frequency
temperature = 0.0    # zero-temperature bath

correlations = oqupy.PowerLawSD(
    alpha=alpha,
    zeta=1.0,                    # zeta = 1 -> Ohmic
    cutoff=cutoff,
    cutoff_type="exponential",
    temperature=temperature,
)
bath = oqupy.Bath(0.5 * oqupy.operators.sigma("z"), correlations)

# ----------------------------------------------------------------------
# 3. TEMPO numerical parameters
#    dt     : time step
#    tcut   : bath memory cutoff time (how far back correlations are kept)
#    epsrel : relative SVD truncation tolerance (controls MPS bond dimension)
# ----------------------------------------------------------------------
tempo_parameters = oqupy.TempoParameters(
    dt=0.05,
    tcut=2.0,
    epsrel=1e-5,
)

# ----------------------------------------------------------------------
# 4. Evolve the reduced system state from the spin-up state
# ----------------------------------------------------------------------
initial_state = oqupy.operators.spin_dm("z+")  # |up><up|

dynamics = oqupy.tempo_compute(
    system=system,
    bath=bath,
    initial_state=initial_state,
    start_time=0.0,
    end_time=10.0,
    parameters=tempo_parameters,
)

# Expectation value of sigma_z as a function of time
times, s_z = dynamics.expectations(oqupy.operators.sigma("z"), real=True)

# ----------------------------------------------------------------------
# 5. Plot
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(times, s_z, lw=2)
plt.axhline(0.0, color="0.7", lw=0.8, zorder=0)
plt.xlabel("Time  $t$")
plt.ylabel(r"$\langle \sigma_z \rangle$")
plt.title(f"Spin-boson dynamics (TEMPO), Ohmic bath $\\alpha={alpha}$")
plt.tight_layout()
plt.savefig("oqupy_spin_boson.png", dpi=150)
print("Saved figure to oqupy_spin_boson.png")
print(f"Final <sigma_z> at t={times[-1]:.2f}: {s_z[-1]:.4f}")
