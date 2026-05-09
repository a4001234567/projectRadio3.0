import numpy as np


def ebno_to_sigma(ebno_db: float, rate: float, bits_per_symbol: int) -> float:
    return 10 ** (-ebno_db / 20) / np.sqrt(2 * rate * bits_per_symbol)


# ── Binary (BPSK) ─────────────────────────────────────────────────────────────

def bpsk_mod(bits: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * bits.reshape(-1).astype(float)

def bpsk_llr(y: np.ndarray, sigma: float) -> np.ndarray:
    return (2.0 / sigma ** 2) * y.reshape(-1)


# ── Non-binary (Gray-coded QAM) ───────────────────────────────────────────────

def qam_constellation(M: int) -> np.ndarray:
    """1-D PAM-M constellation, normalized to unit average power."""
    levels = np.arange(-(M - 1), M, 2, dtype=float)   # ±1, ±3, …
    return levels / np.sqrt(np.mean(levels ** 2))

def qam_mod(bits: np.ndarray, M: int) -> np.ndarray:
    """Gray-coded M-QAM modulation. bits length must be divisible by log2(M)."""
    m = int(np.log2(M))
    bits = bits.reshape(-1)
    syms = np.zeros(len(bits) // m)
    for k in range(m):
        syms += bits[k::m] * (2 ** (m - 1 - k))
    # Gray decode index → PAM level
    gray = syms.astype(int) ^ (syms.astype(int) >> 1)
    const = qam_constellation(M)
    return const[gray]

def qam_llr(y: np.ndarray, sigma: float, M: int) -> np.ndarray:
    """Max-log LLR for each coded bit from received PAM symbol y."""
    m   = int(np.log2(M))
    c   = qam_constellation(M)
    y   = y.reshape(-1, 1)          # (n_sym, 1)
    d2  = (y - c) ** 2             # squared distances to all constellation points

    llrs = np.empty((len(y), m))
    for k in range(m):
        # bit k of Gray index for each constellation point
        idx    = np.arange(M)
        gray_k = ((idx ^ (idx >> 1)) >> (m - 1 - k)) & 1
        d_0    = np.min(d2[:, gray_k == 0], axis=1)
        d_1    = np.min(d2[:, gray_k == 1], axis=1)
        llrs[:, k] = (d_1 - d_0) / sigma ** 2   # max-log approximation

    return llrs.reshape(-1)
