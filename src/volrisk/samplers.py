import numpy as np


def log_posterior(theta, x, y):
    """Log-posterior for CAPM regression y = alpha + beta*x + eps, theta = (alpha, beta, log_sigma)."""
    a, b, ls = theta
    s = np.exp(ls)
    resid = y - a - b * x
    loglik = -len(y) * ls - 0.5 * np.sum(resid ** 2) / s ** 2
    logprior = -0.5 * a ** 2 - 0.5 * (b - 1.0) ** 2 - 0.5 * (ls / 2.0) ** 2
    return float(loglik + logprior)


def metropolis(logp, x0, n_iter, step, seed=0):
    """Random-walk Metropolis-Hastings sampler in log-space."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, float)
    d = len(x)
    step = np.broadcast_to(np.asarray(step, float), (d,))
    chain = np.empty((n_iter, d))
    lp = logp(x)
    accepted = 0
    for i in range(n_iter):
        prop = x + step * rng.standard_normal(d)
        lp_prop = logp(prop)
        if np.log(rng.uniform()) < lp_prop - lp:
            x, lp = prop, lp_prop
            accepted += 1
        chain[i] = x
    return chain, accepted / n_iter


def tune_step(logp, x0, step=0.05, rounds=5, n_iter=400, seed=0):
    """Tune proposal step size in warm-up to hit ~20-40% acceptance rate."""
    x = np.asarray(x0, float)
    for r in range(rounds):
        ch, acc = metropolis(logp, x, n_iter, step, seed=seed + r)
        x = ch[-1]
        if acc > 0.40:
            step *= 1.3
        elif acc < 0.20:
            step *= 0.7
    return step, x