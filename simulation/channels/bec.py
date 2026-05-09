import numpy as np

INF = 1e6   # LLR magnitude for a known (non-erased) bit


def bec_llr(bits: np.ndarray, epsilon: float) -> np.ndarray:
    """Binary Erasure Channel.

    Returns LLR array: +/-INF for received bits, 0 for erasures.
    Positive LLR → bit 0, negative → bit 1 (same sign convention as AWGN path).
    """
    bits     = bits.reshape(-1).astype(bool)
    llr      = np.where(bits, -INF, INF)
    erased   = np.random.rand(len(bits)) < epsilon
    llr[erased] = 0.0
    return llr


def bec_fer_bound(epsilon: float, k: int, n: int) -> float:
    """Union bound on FER for a (n, k) code over BEC(epsilon).

    Exact for any linear code: FER = P(rank of erased columns < n-k).
    This returns the trivial erasure-pattern bound P(≥ n-k+1 erasures).
    """
    from math import comb
    p = 0.0
    for e in range(n - k + 1, n + 1):
        p += comb(n, e) * (epsilon ** e) * ((1 - epsilon) ** (n - e))
    return p
