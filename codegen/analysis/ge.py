import numpy as np
from dataclasses import dataclass


@dataclass
class CodeInfo:
    K_design: int   # design rows
    N: int          # code length (columns)
    rank: int       # GF(2) rank of H
    k_design: int   # design dimension = N - K_design
    k_actual: int   # actual dimension = N - rank
    rate_design: float
    rate_actual: float
    redundant_rows: int  # rank-deficient rows = K_design - rank

    def __str__(self):
        return (
            f"Shape      : {self.K_design} x {self.N}\n"
            f"GF(2) rank : {self.rank}  ({self.redundant_rows} redundant rows)\n"
            f"k (design) : {self.k_design}   rate = {self.rate_design:.6f}\n"
            f"k (actual) : {self.k_actual}   rate = {self.rate_actual:.6f}"
        )


def analyse(H: np.ndarray) -> CodeInfo:
    """Compute design and actual rate of H over GF(2)."""
    K, N = H.shape
    r = gf2_rank(H)
    k_design = N - K
    k_actual = N - r
    return CodeInfo(
        K_design=K, N=N, rank=r,
        k_design=k_design, k_actual=k_actual,
        rate_design=k_design / N,
        rate_actual=k_actual / N,
        redundant_rows=K - r,
    )


def gf2_rank(H: np.ndarray) -> int:
    """Rank of H over GF(2) via row reduction."""
    work = H.astype(np.uint8).copy()
    K, N = work.shape
    r = 0
    for col in range(N):
        pivot = next((i for i in range(r, K) if work[i, col]), None)
        if pivot is None:
            continue
        work[[r, pivot]] = work[[pivot, r]]
        rows = np.where(work[:, col])[0]
        work[rows[rows != r]] ^= work[r]
        r += 1
        if r == K:
            break
    return r


def info_positions(H: np.ndarray) -> np.ndarray:
    """Return indices of information (non-pivot) columns via GF(2) row reduction."""
    work = H.astype(np.uint8).copy()
    K, N = work.shape
    pivot_cols = []
    r = 0
    for col in range(N):
        pivot = next((i for i in range(r, K) if work[i, col]), None)
        if pivot is None:
            continue
        work[[r, pivot]] = work[[pivot, r]]
        rows = np.where(work[:, col])[0]
        work[rows[rows != r]] ^= work[r]
        pivot_cols.append(col)
        r += 1
        if r == K:
            break
    return np.array(sorted(set(range(N)) - set(pivot_cols)), dtype=np.int32)


def systematic_form(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (H_sys, col_perm) where H_sys = H[:, col_perm] is in systematic form [P | I]."""
    work = H.astype(np.uint8).copy()
    K, N = work.shape
    col_perm = list(range(N))
    r = 0
    for col in range(N):
        pivot = next((i for i in range(r, K) if work[i, col]), None)
        if pivot is None:
            continue
        work[[r, pivot]] = work[[pivot, r]]
        col_perm[r], col_perm[col] = col_perm[col], col_perm[r]
        work[:, [r, col]] = work[:, [col, r]]
        rows = np.where(work[:, r])[0]
        work[rows[rows != r]] ^= work[r]
        r += 1
        if r == K:
            break
    return work, np.array(col_perm, dtype=np.int32)
