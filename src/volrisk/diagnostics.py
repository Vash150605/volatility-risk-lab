import numpy as np


def autocorr(x):
    """FFT autocorrelation of a 1D chain."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n]
    return acov / acov[0]


def ess(x):
    """Effective sample size, summing autocorrelation pairs until they turn negative."""
    rho = autocorr(x)
    s = 0.0
    for k in range(1, len(rho) - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair < 0:
            break
        s += pair
    return float(len(x) / (1 + 2 * s))


def rhat(chains):
    """Split R-hat. chains: array of shape (m_chains, n_draws) for ONE parameter."""
    m, n = chains.shape
    half = n // 2
    ch = np.concatenate([chains[:, :half], chains[:, half : 2 * half]], axis=0)
    m, n = ch.shape
    W = ch.var(axis=1, ddof=1).mean()
    B = n * ch.mean(axis=1).var(ddof=1)
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W))