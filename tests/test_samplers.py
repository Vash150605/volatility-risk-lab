import numpy as np
from volrisk.samplers import metropolis


def test_sampler_recovers_known_gaussian_posterior():
    """Metropolis recovers known 2D Gaussian mean and covariance within tolerance."""
    true_mean = np.array([1.2, -0.7])
    true_cov = np.array([[1.0, 0.35], [0.35, 0.5]])
    inv_cov = np.linalg.inv(true_cov)

    def logp(x):
        d = x - true_mean
        return -0.5 * float(d @ inv_cov @ d)

    # Larger step=[1.2, 0.9] puts acceptance right in the 20%-40% sweet spot
    chain, acc = metropolis(logp, x0=[0.0, 0.0], n_iter=25_000, step=[1.2, 0.9], seed=11)
    post = chain[5_000:]
    assert 0.20 < acc < 0.45
    np.testing.assert_allclose(post.mean(axis=0), true_mean, atol=0.06)
    np.testing.assert_allclose(np.cov(post, rowvar=False), true_cov, atol=0.06)