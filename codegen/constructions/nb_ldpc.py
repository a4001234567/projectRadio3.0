"""
Non-binary QC-LDPC matrix construction over GF(2^m).

Binary support: block (i, j) is a circulant shifted by i*j mod block_size.
NB scaling:    block (i, j) is multiplied by scalar[j]^i in GF(2^m).

The resulting H is stored as integer-valued numpy array and written
using the .nonbinary format in io/matrix_io.py.
"""
import random
from typing import List, Optional

import numpy as np
import sympy

from codegen.analysis.cr import check_CR
from fileio.matrix_io import writer


# ── GF(2^m) field tables ──────────────────────────────────────────────────────

def build_gf_tables(primitive_poly: sympy.Poly):
    """Return (N, Q, GF_VEC, GF_POW, GF_MUL) for GF(2^m) defined by primitive_poly."""
    x = sympy.symbols('x')
    N = primitive_poly.degree()
    Q = 1 << N

    def _poly_to_int(poly):
        coeffs = [c % 2 for c in poly.all_coeffs()][::-1]
        return sum(b << i for i, b in enumerate(coeffs))

    GF_VEC = []
    cur = sympy.poly(1, x, domain='GF(2)')
    for _ in range(1, Q):
        GF_VEC.append(_poly_to_int(cur))
        cur = (cur * sympy.poly(x, domain='GF(2)')) % primitive_poly

    GF_POW = [0] * Q
    for i in range(1, Q):
        GF_POW[GF_VEC[i - 1]] = i - 1

    GF_MUL = np.zeros((Q, Q), dtype='uint8')
    for i in range(1, Q):
        for j in range(1, Q):
            GF_MUL[i, j] = GF_VEC[(GF_POW[i] + GF_POW[j]) % (Q - 1)]

    return N, Q, GF_VEC, GF_POW, GF_MUL


def gf_pow(n: int, k: int, Q: int, GF_VEC: list, GF_POW: list) -> int:
    """Compute n^k in GF(2^m).  Returns 0 if n==0, 1 if k==0."""
    if n == 0:
        return 0
    if k == 0:
        return 1
    return GF_VEC[(GF_POW[n] * k) % (Q - 1)]


# ── Binary support ────────────────────────────────────────────────────────────

def _circulant_block(shift: int, size: int) -> np.ndarray:
    block = np.zeros((size, size), dtype=np.int64)
    for row in range(size):
        block[row, (row + shift) % size] = 1
    return block


def build_support(j: int, l: int, block_size: int) -> np.ndarray:
    """Binary support: block (i,j) = circulant with shift i*j mod block_size."""
    H = np.zeros((j * block_size, l * block_size), dtype=np.int64)
    for i in range(j):
        for jj in range(l):
            r0, c0 = i * block_size, jj * block_size
            H[r0:r0+block_size, c0:c0+block_size] = _circulant_block((i * jj) % block_size, block_size)
    return H


# ── NB construction ───────────────────────────────────────────────────────────

def build_nb_qc_ldpc(
    j: int,
    l: int,
    block_size: int,
    primitive_poly: sympy.Poly,
    scalars: Optional[List[int]] = None,
    rand_seed: int = 42,
    filename: str = '',
) -> np.ndarray:
    """
    Build a non-binary QC-LDPC parity-check matrix over GF(2^m).

    Block (i, j) = P^{i·j mod block_size} · scalar[j]^i

    Parameters
    ----------
    j, l           : row- and column-block counts
    block_size     : circulant period
    primitive_poly : sympy.Poly defining GF(2^m)
    scalars        : l non-zero GF elements; randomly chosen if None
    rand_seed      : seed for random scalar selection
    filename       : if non-empty, write matrix to this file

    Returns
    -------
    H : (j*block_size) × (l*block_size) integer array over GF(2^m)
    """
    N, Q, GF_VEC, GF_POW, GF_MUL = build_gf_tables(primitive_poly)

    if scalars is None:
        rng = random.Random(rand_seed)
        scalars = rng.sample(range(1, Q), l)
    assert len(scalars) == l and all(0 < s < Q for s in scalars), \
        "scalars must be l non-zero GF elements"

    H = build_support(j, l, block_size)

    for i in range(j):
        for jj in range(l):
            scale = gf_pow(scalars[jj], i, Q, GF_VEC, GF_POW)
            r0, c0 = i * block_size, jj * block_size
            block = H[r0:r0+block_size, c0:c0+block_size]
            H[r0:r0+block_size, c0:c0+block_size] = np.where(block, scale, 0)

    assert check_CR(H.astype(bool)), "CR condition violated"

    if filename:
        writer(filename, H, fieldSize=Q, mode='normal', compress=False)

    return H


if __name__ == '__main__':
    x = sympy.symbols('x')
    poly = sympy.poly(x**5 + x**2 + 1, domain='GF(2)')
    H = build_nb_qc_ldpc(j=3, l=19, block_size=19, primitive_poly=poly, filename='Example5.txt')
    print(f'H shape: {H.shape}  field size: {1 << poly.degree()}  max element: {H.max()}')
