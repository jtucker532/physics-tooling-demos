"""
emcee_line_fit.py
=================

Bayesian parameter estimation with `emcee`, the affine-invariant ensemble MCMC
sampler. We fit a straight line to noisy data and recover the full posterior
distribution over the slope and intercept (with credible intervals), rather
than just a single best-fit point.

Model
-----
    y = m * x + b + noise,     noise ~ Normal(0, sigma)

We place flat (uniform) priors on m and b, use a Gaussian likelihood, and let
emcee sample the posterior with an ensemble of walkers. This is the canonical
emcee demonstration and maps directly onto real physics data-fitting tasks
(e.g. extracting a decay rate or a calibration slope with honest error bars).

Run:
    python emcee_line_fit.py
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1. Generate synthetic data from known "true" parameters
# ----------------------------------------------------------------------
m_true, b_true = 2.3, -1.1
sigma = 0.6                       # measurement noise level

x = np.sort(rng.uniform(0, 10, size=40))
y_true = m_true * x + b_true
y = y_true + rng.normal(0.0, sigma, size=x.size)

# ----------------------------------------------------------------------
# 2. Define log-prior, log-likelihood, log-posterior
# ----------------------------------------------------------------------
def log_prior(theta):
    m, b = theta
    if -10.0 < m < 10.0 and -10.0 < b < 10.0:
        return 0.0          # flat prior within bounds
    return -np.inf

def log_likelihood(theta, x, y, sigma):
    m, b = theta
    model = m * x + b
    return -0.5 * np.sum((y - model) ** 2 / sigma ** 2)

def log_posterior(theta, x, y, sigma):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, sigma)

# ----------------------------------------------------------------------
# 3. Set up and run the sampler
# ----------------------------------------------------------------------
n_walkers, n_dim = 32, 2
initial = np.array([1.0, 0.0]) + 1e-2 * rng.normal(size=(n_walkers, n_dim))

sampler = emcee.EnsembleSampler(
    n_walkers, n_dim, log_posterior, args=(x, y, sigma)
)
sampler.run_mcmc(initial, 3000, progress=False)

# Discard burn-in and thin the chain
flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)

# ----------------------------------------------------------------------
# 4. Report posterior medians with 16th/84th percentile credible intervals
# ----------------------------------------------------------------------
labels = ["m (slope)", "b (intercept)"]
truths = [m_true, b_true]
print("Posterior estimates (median with +/- 1 sigma credible interval):")
for i, label in enumerate(labels):
    lo, med, hi = np.percentile(flat_samples[:, i], [16, 50, 84])
    print(f"  {label:14s} = {med:+.3f}  (+{hi-med:.3f} / -{med-lo:.3f})   "
          f"[true = {truths[i]:+.3f}]")

# ----------------------------------------------------------------------
# 5. Plot data + posterior-sampled fit lines
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(x, y, yerr=sigma, fmt=".k", capsize=2, label="data")
x_grid = np.linspace(0, 10, 100)
for m, b in flat_samples[rng.integers(len(flat_samples), size=80)]:
    ax.plot(x_grid, m * x_grid + b, color="C0", alpha=0.05)
ax.plot(x_grid, m_true * x_grid + b_true, color="C3", lw=2, label="truth")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Bayesian line fit with emcee (posterior draws)")
ax.legend()
fig.tight_layout()
fig.savefig("emcee_line_fit.png", dpi=150)
print("Saved figure to emcee_line_fit.png")
