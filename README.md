# physics-tooling-demos

Small, self-contained, reproducible examples demonstrating working proficiency
with three open-source physics/scientific-computing packages:

- **QuTiP** &mdash; open quantum-system dynamics via the Lindblad master equation
- **OQuPy** &mdash; non-Markovian open-system dynamics via the TEMPO algorithm
- **emcee** &mdash; Bayesian parameter estimation with affine-invariant MCMC

Each script is standalone, runs in a few seconds to a couple of minutes, prints
a short numerical result, and saves a figure.

## Contents

| Script | Tool | What it does |
| --- | --- | --- |
| `qutip_driven_qubit.py` | QuTiP | Drives a qubit on resonance with `mesolve` under T1 relaxation and pure dephasing; plots `<sigma_z>` and excited-state population vs. time. |
| `oqupy_spin_boson.py` | OQuPy | Evolves a spin-boson model (Ohmic bath, coupling via `sigma_z`) with TEMPO, capturing non-Markovian memory; plots `<sigma_z>` vs. time. |
| `emcee_line_fit.py` | emcee | Fits a straight line to noisy data, sampling the posterior over slope and intercept and reporting median values with credible intervals. |

## Setup

These examples were developed and run with Python 3.11 on macOS (Apple Silicon).
The pinned versions are recorded in `requirements.txt`.

```bash
conda create -n physics-eval python=3.11 -y
conda activate physics-eval
pip install -r requirements.txt
```

Note: OQuPy depends on `tensornetwork`, which pins `numpy < 2`, while some other
quantum packages require `numpy >= 2`. This repository deliberately uses
`numpy 1.26.4` so that QuTiP, OQuPy, and emcee coexist in a single environment.

## Running

```bash
python qutip_driven_qubit.py
python oqupy_spin_boson.py
python emcee_line_fit.py
```

Each run writes a `.png` figure alongside the script.

## A note on physics

All three examples sit in the same conceptual area: building a Hamiltonian or a
statistical model, evolving or sampling it, and computing observables with honest
treatment of dissipation (QuTiP, OQuPy) or uncertainty (emcee). The QuTiP and
OQuPy scripts model the same physical system &mdash; a two-level system coupled to
its environment &mdash; at two levels of approximation: memoryless (Lindblad) and
memory-resolved (TEMPO).
