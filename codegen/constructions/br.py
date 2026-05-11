"""
BR parity-check matrix construction.

S_{P-1} is the (P-1)×(P-1) lower companion matrix:
  - subdiagonal = 1  : S[i, i-1] = 1  for i = 1, ..., P-2
  - last column = 1  : S[i, P-2] = 1  for i = 0, ..., P-2

S_{P-1} is equivalent to the P×P lower circulant P with truncation τ=1,
so the BR Vandermonde block (i, j) = S^{i * cols[j] mod P} produces the
same code as the QC Vandermonde with block size P and truncate=1,
without needing truncation.
"""
from typing import List
import numpy as np


def make_br_basis(P: int) -> np.ndarray:
    """
    Build the (P-1)×(P-1) BR companion matrix S over F2.

    Parameters
    ----------
    P : prime block size of the QC equivalent code

    Returns
    -------
    S : (P-1, P-1) uint8 array
    """
    N = P - 1
    S = np.zeros((N, N), dtype=np.uint8)
    for i in range(1, N):
        S[i, i - 1] = 1      # subdiagonal
    S[:, N - 1] = 1          # last column
    return S


def build_br_matrix(J: int, cols: List[int], P: int) -> np.ndarray:
    """
    Build the BR parity-check matrix.

    Block (i, j) = S^{i * cols[j] mod P}, where S = make_br_basis(P).
    No truncation needed — the (P-1)×(P-1) blocks already match the
    punctured QC circulant size.

    Parameters
    ----------
    J    : number of row-blocks
    cols : L column generators from Z_P  (length L)
    P    : prime — sets block size to P-1

    Returns
    -------
    H : (J*(P-1), L*(P-1)) uint8 array
    """
    L  = len(cols)
    N  = P - 1
    S  = make_br_basis(P)

    # Precompute S^k for k = 0, ..., P-1 over F2
    S_powers = [None] * P
    S_powers[0] = np.eye(N, dtype=np.uint8)
    for k in range(1, P):
        S_powers[k] = (S_powers[k - 1] @ S) % 2

    H = np.zeros((J * N, L * N), dtype=np.uint8)
    for i in range(J):
        for j, c in enumerate(cols):
            exp = (i * c) % P
            H[i*N:(i+1)*N, j*N:(j+1)*N] = S_powers[exp]

    return H
