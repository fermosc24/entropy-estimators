"""
entropy_estimators.py

Entropy estimation library implementing multiple estimation techniques:
- Maximum Likelihood Estimation (MLE)
- James-Stein Shrinkage Estimator (JSE)
- Chao-Shen Estimator (CAE)
- Chao-Wang-Jost Estimator (CWJ)
- Nemenman-Bialek-de Ruyter Estimator (NBRS)
- Nemenman-Shafee-Bialek Estimator (NSB)
"""

import numpy as np
from scipy.special import polygamma
from mpmath import psi, rf, power, quadgl, mp, memoize

Euler = -polygamma(0, 1.)
DPS = 40
psi = memoize(psi)
rf = memoize(rf)
power = memoize(power)

##########################
# General Utility Functions
##########################

def sample_frequencies(M, n):
    """
    Sample elements without replacement from a multiset defined by frequencies.

    Parameters:
        M (np.ndarray): Array of non-negative integers representing frequencies.
        n (int): Number of samples to draw (must be <= M.sum()).

    Returns:
        np.ndarray: Array with sampled frequencies of same shape as M.
    """
    if not np.issubdtype(M.dtype, np.integer):
        raise ValueError("M must contain integer frequencies.")
    total = M.sum()
    if n > total:
        raise ValueError("Cannot sample more elements than available instances.")
    population = np.repeat(np.arange(M.size), M.flatten())
    sampled = np.random.choice(population, size=n, replace=False)
    counts = np.bincount(sampled, minlength=M.size)
    return counts.reshape(M.shape)


def dict_to_ndarray(d):
    """
    Convert dictionary with tuple keys to ndarray and index maps.

    Parameters:
        d (dict): Dictionary with tuple keys and integer values.

    Returns:
        tuple: (ndarray, list of index maps for each dimension)
    """
    if not d:
        raise ValueError("Input dictionary is empty.")
    t = len(next(iter(d)))
    label_sets = [set() for _ in range(t)]
    for key in d:
        if len(key) != t:
            raise ValueError("All keys must be the same length.")
        for i, label in enumerate(key):
            label_sets[i].add(label)
    index_maps = [{label: idx for idx, label in enumerate(sorted(s))} for s in label_sets]
    shape = tuple(len(m) for m in index_maps)
    arr = np.zeros(shape, dtype=int)
    for key, value in d.items():
        idx = tuple(index_maps[i][label] for i, label in enumerate(key))
        arr[idx] = value
    return arr, index_maps

##########################
# Entropy Estimation
##########################

def Entropy(counts, method="MLE", K=None, base=2):
    """
    Estimate entropy from frequency counts using selected method.

    Parameters:
        counts (list): List of non-negative integer counts.
        method (str): Entropy estimator ('MLE', 'JSE', 'CAE', 'CWJ', 'NBRS', 'NSB').
        K (int): Total number of possible outcomes.
        base (int): Base of logarithm for entropy output.

    Returns:
        float: Estimated entropy value.
    """
    if not counts:
        return 0.0
    if method == "CAE":
        return ChaoShen(counts)/np.log(base)
    elif method == "CWJ":
        return ChaoWangJost(counts)/np.log(base)
    elif method == "NBRS":
        return NBRS(counts)/np.log(base)
    elif method == "JSE":
        c2 = list(counts)
        if K and K > len(c2):
            c2 += [0] * (K - len(c2))
        return JamesSteinShrink(c2)/np.log(base)
    elif method == "NSB":
        c2 = list(counts)
        if K and K > len(c2):
            c2 += [0] * (K - len(c2))
        else:
            K = len(c2)
        freqs = np.array(c2)
        NK = make_nxkx(freqs, K)
        return S(NK, np.sum(freqs), K)/np.log(base)
    else:
        return EntropyML(counts)/np.log(base)


def EntropyML(counts):
    """
    Maximum Likelihood Entropy Estimate.

    Parameters:
        counts (list): List of non-negative integer counts.

    Returns:
        float: MLE estimate of entropy in nats.
    """
    freqs = np.array(counts)
    p = freqs / np.sum(freqs)
    return -np.sum(p[p > 0] * np.log(p[p > 0]))


def FreqShrink(counts):
    """
    James-Stein shrinkage frequency smoothing.

    Parameters:
        counts (list): List of non-negative integer counts.

    Returns:
        np.ndarray: Smoothed probability distribution.
    """
    t = 1. / float(len(counts))
    N = float(np.sum(counts))
    p = np.array(counts) / N
    if N <= 1.:
        lambdaF = 1.
    else:
        lambdaF = (1. - np.sum(p**2.)) / ((N-1.) * np.sum((t - p)**2.))
    lambdaF = min(max(lambdaF, 0.), 1.)
    return lambdaF * t + (1. - lambdaF) * p


def JamesSteinShrink(counts):
    """
    James-Stein Shrinkage Entropy Estimate.

    Parameters:
        counts (list): List of non-negative integer counts.

    Returns:
        float: Shrinkage entropy in nats.
    """
    p = FreqShrink(counts)
    return -np.sum(p * np.log(p))


def ChaoShen(counts):
    """
    Chao-Shen entropy estimator with coverage adjustment.

    Parameters:
        counts (list): List of non-negative integer counts.

    Returns:
        float: Chao-Shen entropy estimate in nats.
    """
    freqs = np.array(counts)
    freqs = freqs[freqs > 0]
    f1 = float(np.sum(freqs == 1))
    n = float(np.sum(freqs))
    C = 1. - f1 / n
    p = freqs / n
    if C > 0.:
        p = C * p
    return -np.sum(p * np.log(p) / (1. - (1. - p)**n))


def ChaoWangJost(counts):
    """
    Chao-Wang-Jost entropy estimator.

    Parameters:
        counts (list): List of non-negative integer counts.

    Returns:
        float: CWJ entropy estimate in nats.
    """
    freqs = np.array(counts)
    freqs = freqs[freqs > 0]
    n = float(np.sum(freqs))
    f1 = float(np.sum(freqs == 1))
    f2 = float(np.sum(freqs == 2))
    if f2 > 0:
        A = 2. * f2 / ((n - 1.) * f1 + 2. * f2)
    elif f1 > 0:
        A = 2. / ((n - 1.) * (f1 - 1.) + 2.)
    else:
        A = 1.
    freqs = freqs[freqs < n]
    R = np.sum([_CWJ_aux(x, n) for x in freqs])
    r = np.arange(1, int(n))
    if A != 1.0:
        R -= f1 * (np.log(A) + np.sum(((1. - A)**r) / r)) * ((1. - A)**(1. - n)) / n
    return R

_CWJ_Chart = {}

def _CWJ_aux(Xi, n):
    if (Xi, n) not in _CWJ_Chart:
        val = Xi * np.sum(1. / np.arange(Xi, int(n))) / n
        _CWJ_Chart[Xi, n] = val
    return _CWJ_Chart[Xi, n]


def NBRS(counts):
    """
    NBRS entropy estimator (Nemenman-Bialek-de Ruyter van Steveninck).

    Parameters:
        counts (list): List of non-negative integer counts.

    Returns:
        float: NBRS entropy estimate in nats.
    """
    N = float(np.sum(counts))
    freqs = np.array(counts)
    freqs = freqs[freqs > 0]
    f1 = np.sum(freqs == 1)
    Delt = N - f1
    if Delt > 0.:
        S = Euler / np.log(2) - 1 + 2. * np.log(N) / np.log(2.) - polygamma(0, Delt)
        return S * np.log(2)
    return EntropyML(counts)

##########################
# NSB Entropy Estimation
##########################

"""
nsb_entropy.py

June, 2011 written by Sungho Hong, Computational Neuroscience Unit, Okinawa Institute of
Science and Technology
May 2019 updated to python3 by Charlie Strauss, Los Alamos National Lab

This script is a python version of Mathematica functions by Christian Mendl
implementing the Nemenman-Shafee-Bialek (NSB) estimator of entropy. For the
details of the method, check out the references below.

It depends on mpmath and numpy package.

References:
http://christian.mendl.net/pages/software.html
http://nsb-entropy.sourceforge.net
Ilya Nemenman, Fariel Shafee, and William Bialek. Entropy and Inference,
Revisited. arXiv:physics/0108025
Ilya Nemenman, William Bialek, and Rob de Ruyter van Steveninck. Entropy and
information in neural spike trains: Progress on the sampling problem. Physical
Review E 69, 056111 (2004)

"""

def make_nxkx(n, K):
    """
    Construct histogram-of-histogram for NSB estimator.

    Parameters:
        n (np.ndarray): Original histogram array.
        K (int): Total number of possible outcomes.

    Returns:
        dict: Dictionary mapping count values to number of bins with that count.
    """
    nxkx = {}
    nn = n[n > 0]
    for x in np.unique(nn):
        nxkx[x] = (nn == x).sum()
    if K > nn.size:
        nxkx[0] = K - nn.size
    return nxkx


def S(nxkx, N, K):
    """
    NSB entropy estimate via numerical integration.

    Parameters:
        nxkx (dict): Histogram-of-histogram.
        N (int): Total number of samples.
        K (int): Total number of possible outcomes.

    Returns:
        float: NSB entropy estimate in nats.
    """
    mp.dps = DPS
    mp.pretty = True
    f = lambda w: _Si(w, nxkx, N, K)
    g = lambda w: _measure(w, nxkx, N, K)
    return quadgl(f, [0, 1], maxdegree=20) / quadgl(g, [0, 1], maxdegree=20)


def _Si(w, nxkx, N, K):
    sbeta = w / (1 - w)
    beta = sbeta * sbeta
    return _rho(beta, nxkx, N, K) * _S1(beta, nxkx, N, K) * _dxi(beta, K) * 2 * sbeta / (1 - w)**2


def _measure(w, nxkx, N, K):
    sbeta = w / (1 - w)
    beta = sbeta * sbeta
    return _rho(beta, nxkx, N, K) * _dxi(beta, K) * 2 * sbeta / (1 - w)**2


def _S1(beta, nxkx, N, K):
    kappa = beta * K
    return -sum(nxkx[x] * (x + beta) / (N + kappa) * (psi(0, x + beta + 1) - psi(0, N + kappa + 1)) for x in nxkx)


def _rho(beta, nxkx, N, K):
    kappa = beta * K
    num = np.prod([power(rf(beta, float(x)), nxkx[x]) for x in nxkx])
    denom = rf(kappa, float(N))
    return num / denom


def _dxi(beta, K):
    return K * psi(1, K * beta + 1) - psi(1, beta + 1)


##########################
# Optional: NSB Std Estimation
##########################

def dS(nxkx, N, K):
    """
    NSB entropy estimator standard deviation.

    Parameters:
        nxkx (dict): Histogram-of-histogram.
        N (int): Total number of samples.
        K (int): Total number of possible outcomes.

    Returns:
        float: Variance of estimated entropy (can be used to derive std).
    """
    mp.dps = DPS
    mp.pretty = True
    f = lambda w: _dSi(w, nxkx, N, K)
    g = lambda w: _measure(w, nxkx, N, K)
    return quadgl(f, [0, 1], maxdegree=20) / quadgl(g, [0, 1], maxdegree=20)


def _dSi(w, nxkx, N, K):
    sbeta = w / (1 - w)
    beta = sbeta * sbeta
    return _rho(beta, nxkx, N, K) * _S2(beta, nxkx, N, K) * _dxi(beta, K) * 2 * sbeta / (1 - w)**2


def _S2(beta, nxkx, N, K):
    kappa = beta * K
    nx = list(nxkx.keys())
    dsum = 0.0
    ndsum = 0.0
    for x in nx:
        xbeta = x + beta
        Nk2 = N + kappa + 2
        psNK2 = psi(0, Nk2)
        ps1NK2 = psi(1, Nk2)
        s1 = (psi(0, xbeta + 2) - psNK2)**2 + psi(1, xbeta + 2) - ps1NK2
        s1 *= nxkx[x] * xbeta * (xbeta + 1)
        s2 = (psi(0, xbeta + 1) - psNK2)**2 - ps1NK2
        s2 *= nxkx[x] * (nxkx[x] - 1) * xbeta**2
        dsum += s1 + s2
    for i in range(len(nx) - 1):
        for j in range(i + 1, len(nx)):
            x1, x2 = nx[i], nx[j]
            x1b, x2b = x1 + beta, x2 + beta
            psNK2 = psi(0, N + kappa + 2)
            ps1NK2 = psi(1, N + kappa + 2)
            s = (psi(0, x1b + 1) - psNK2) * (psi(0, x2b + 1) - psNK2) - ps1NK2
            s *= nxkx[x1] * nxkx[x2] * x1b * x2b
            ndsum += s
    return (dsum + 2 * ndsum) / (N + kappa) / (N + kappa + 1)
