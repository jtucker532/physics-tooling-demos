"""
qutip_driven_qubit.py
=====================

Open-system dynamics of a driven qubit using QuTiP's Lindblad master-equation
solver (`mesolve`).

Physics
-------
A two-level system is driven on resonance while simultaneously undergoing
energy relaxation (T1) and pure dephasing (T2). This is the workhorse model of
superconducting-qubit control: coherent Rabi flopping competing with
decoherence.

    H = (Omega / 2) * sigma_x                       (resonant drive)
    Collapse operators:
        sqrt(gamma_1)   * sigma_minus               (T1 relaxation)
        sqrt(gamma_phi) * sigma_z / sqrt(2)         (pure dephasing)

We start in the ground state |0> (the state relaxation drives toward) and
track <sigma_z>(t) and the excited-state population, watching the drive pump
population upward while Rabi oscillations decay toward the steady state.

Run:
    python qutip_driven_qubit.py
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# ----------------------------------------------------------------------
# 1. Hamiltonian: resonant drive about the x-axis
# ----------------------------------------------------------------------
Omega = 2.0 * np.pi * 1.0        # Rabi frequency (rad/time)
H = 0.5 * Omega * qt.sigmax()

# ----------------------------------------------------------------------
# 2. Collapse operators (Lindblad dissipators)
# ----------------------------------------------------------------------
gamma_1 = 0.5                    # relaxation rate (1/T1)
gamma_phi = 0.2                  # pure dephasing rate
c_ops = [
    np.sqrt(gamma_1) * qt.sigmam(),
    np.sqrt(gamma_phi / 2.0) * qt.sigmaz(),
]

# ----------------------------------------------------------------------
# 3. Initial state and time grid
#    In QuTiP, sigmam() = |0><1| relaxes |1> -> |0>, so basis(2, 0) = |0> is
#    the ground state that decoherence drives toward. We start there.
# ----------------------------------------------------------------------
psi0 = qt.basis(2, 0)
tlist = np.linspace(0.0, 6.0, 600)

# ----------------------------------------------------------------------
# 4. Solve the master equation
# ----------------------------------------------------------------------
result = qt.mesolve(
    H, psi0, tlist, c_ops,
    e_ops=[qt.sigmaz(), qt.num(2)],  # <sigma_z> and excited-state population
)
exp_sz = result.expect[0]
pop_excited = result.expect[1]

# ----------------------------------------------------------------------
# 5. Plot
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(tlist, exp_sz, lw=2, label=r"$\langle \sigma_z \rangle$")
ax.plot(tlist, pop_excited, lw=2, ls="--", label="excited population")
ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
ax.set_xlabel("Time  $t$")
ax.set_ylabel("Expectation value")
ax.set_title(r"Driven qubit with $T_1$ and $T_2$ decay (QuTiP mesolve)")
ax.legend()
fig.tight_layout()
fig.savefig("qutip_driven_qubit.png", dpi=150)
print("Saved figure to qutip_driven_qubit.png")
print(f"Steady-state <sigma_z> ~ {exp_sz[-1]:.4f}")
