import numpy as np
from scipy.stats import qmc

# Global parameter defaults
d_min = 0.3       # Minimum allowed distance to avoid points being too close to obstacles
d_opt = 0.4       # Optimal distance (highest sampling probability)
sigma = 0.5       # Controls the width of the probability distribution
floor_prob = 0.2  # Minimum sampling probability in open areas


def density_probability(d, d_min, d_opt, sigma, floor):
    """Vectorized Gaussian-shaped probability of accepting a sample with wall distance `d`.
    Returns 0 wherever d < d_min."""
    d_arr = np.asarray(d, dtype=float)
    prob = floor + (1.0 - floor) * np.exp(-((d_arr - d_opt) ** 2) / (2.0 * sigma ** 2))
    prob = np.where(d_arr < d_min, 0.0, prob)
    if np.isscalar(d) or np.ndim(d) == 0:
        return float(prob)
    return prob


def halton_sampling(
    n_samples,
    min_wall_distance,
    bounds,
    d_min,
    d_opt,
    sigma,
    floor,
    halton_sampler=None,
    rng=None,
    oversample=1,
):
    """Draw `n_samples` Halton points scaled to `bounds`, then apply distance-based
    rejection sampling so points near obstacles/inflation are unlikely to be kept.

    The Halton sampler is persisted across calls — pass the returned sampler back
    in to continue the deterministic sequence (otherwise repeated calls would
    return the same points).
    """
    bounds = np.asarray(bounds, dtype=float)
    dim = bounds.shape[0]
    if halton_sampler is None:
        halton_sampler = qmc.Halton(d=dim, scramble=False)
    if rng is None:
        # Use the legacy global RNG so the rejection sampling honors np.random.seed()
        # (set_global_seed) — otherwise an independent default_rng() makes the roadmap
        # (and everything downstream, incl. the GNN prune) differ run-to-run.
        rng = np.random

    n_draw = int(max(1, n_samples * max(1, oversample)))
    raw = halton_sampler.random(n_draw)
    samples = qmc.scale(raw, bounds[:, 0], bounds[:, 1])

    if d_min <= 0:
        return samples, halton_sampler

    dists = min_wall_distance(samples)
    dists = np.asarray(dists, dtype=float)
    probs = density_probability(dists, d_min, d_opt, sigma, floor)
    accept = rng.random(len(samples)) < probs
    return samples[accept], halton_sampler
