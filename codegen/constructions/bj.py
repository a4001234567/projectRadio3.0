"""
BJ LDPC code constructions over GF(2^m).

Construction 1 (binary matrix)
    Codewords C_i = {alpha^i*g(x) + lambda*x*g(x) : lambda in GFq}.
    Each GFq symbol expanded to m bits. Result shape: (q^2) x (m*(q+1)).

Construction 2 (marker / QC matrix)
    Each GFq symbol replaced by a circulant permutation matrix.
    Result is a (q-1)(q+1) x (q-1)(q+1) block matrix (before truncation).
    Parameterised by gamma (row blocks), rho (column blocks), u (extra cols).
"""

import sys
import os
import numpy as np
import galois

from codegen.qc.grid import generate_rulers, write_board, writeNDarray
from fileio.matrix_io import writer


# ── GF(2^m) field setup ───────────────────────────────────────────────────────

def setup(m: int) -> dict:
    """
    Set up the BJ code over F_q = GF(2^m), q = 2^m.

    Returns a dict with:
        m, q, n     : field and code parameters (n = q+1)
        GFq         : GF(2^m) field object
        alpha_q     : primitive element of GFq
        g_asc       : ascending coefficients [g_0, ..., g_{q-1}] of generator poly
        log_table   : {int(elem): discrete_log} for nonzero GFq elements
    """
    q = 2 ** m
    GFq        = galois.GF(q)
    GFq_square = galois.GF(q * q)

    alpha_q        = GFq.primitive_element
    alpha_qsquared = GFq_square.primitive_element
    beta  = pow(alpha_qsquared, q - 1)
    rootA = beta ** (q // 2)
    rootB = beta * rootA

    def mapin(x):
        assert GFq_square.log(x) % (q + 1) == 0
        return alpha_q ** (GFq_square.log(x) // (q + 1))

    f_mid = galois.Poly([1, -mapin(rootA + rootB), mapin(rootA * rootB)], field=GFq)
    xn1   = galois.Poly([1] + [0] * q + [1], field=GFq)

    g_poly, rem = divmod(xn1, f_mid)
    assert rem == 0

    g_asc     = list(g_poly.coeffs[::-1])
    log_table = {int(alpha_q ** t): t for t in range(q - 1)}

    return {
        'm': m, 'q': q, 'n': q + 1,
        'GFq': GFq, 'alpha_q': alpha_q,
        'g_asc': g_asc, 'log_table': log_table,
    }


def g_marker(k: int, g_asc: list, log_table: dict, n: int):
    """Discrete-log marker for the k-th coefficient of g (mod n)."""
    k = k % n
    if k >= len(g_asc):
        return ()
    v = int(g_asc[k])
    return (log_table[v],) if v != 0 else ()


# ── Construction 2: marker matrix ─────────────────────────────────────────────

def build_marker_matrix(m: int) -> list:
    """
    Build the full (q+1) x (q+1) block-level marker matrix.

    Entry (i, j): int t -> circulant P^t,  () -> zero block.
    """
    ctx = setup(m)
    q, n = ctx['q'], ctx['n']
    g_asc, log_table = ctx['g_asc'], ctx['log_table']
    return [
        [g_marker(j - i, g_asc, log_table, n) for j in range(n)]
        for i in range(1, n + 1)
    ]


def build_bj_matrix(
    m: int,
    gamma: int,
    rho: int,
    u: int = 0,
    filename: str = '',
) -> np.ndarray:
    """
    Build the Type-2 BJ parity-check matrix with gamma/rho/u truncation.

    Parameters
    ----------
    m     : GF(2^m) field parameter
    gamma : row blocks    (1 <= gamma <= q+1)
    rho   : column blocks (1 <= rho   <= q+1)
    u     : extra columns from the (rho+1)-th block (0 <= u <= q-2)
    filename : if non-empty, save to this path (sparse, compressed)

    Returns
    -------
    H : uint8 numpy array of shape (gamma*(q-1), rho*(q-1)+u)
    """
    q     = 2 ** m
    block = q - 1
    N     = rho * block + u

    mm = build_marker_matrix(m)

    sub = [row[:rho] for row in mm[:gamma]]
    rulers = generate_rulers(sub, block)

    if u == 0:
        H = writeNDarray(rulers, block)
    else:
        sub_full  = [row[:rho + 1] for row in mm[:gamma]]
        rulers_full = generate_rulers(sub_full, block)
        H = writeNDarray(rulers_full, block)[:, :N]

    if filename:
        writer(filename, H.astype(np.uint8), mode='sparse', compress=True,
               comments=[f'BJ Type2  m={m} gamma={gamma} rho={rho} u={u}  N={N}'])

    return H.astype(np.uint8)


# ── Construction 2b: non-binary matrix ───────────────────────────────────────

def build_nb_bj_matrix(
    m: int,
    gamma: int,
    rho: int,
    filename: str = '',
) -> np.ndarray:
    """
    Build the non-binary BJ parity-check matrix over GF(2^m).

    H[i, j] = alpha^t  if g_{j-i mod n} = alpha^t  (non-zero GF element)
             = 0        if g_{j-i mod n} = 0

    Shape: gamma x rho (integer-valued, elements in 0..q-1).
    Compatible with the NB-EMS decoder in simulation/nb_decoder.py.
    Use fileio.matrix_io.writer(..., fieldSize=q) to save.

    Parameters
    ----------
    m     : GF(2^m) field parameter  (q = 2^m)
    gamma : row count    (1 <= gamma <= q+1)
    rho   : column count (1 <= rho   <= q+1)
    filename : if non-empty, save in .nonbinary format
    """
    ctx = setup(m)
    q, n = ctx['q'], ctx['n']
    g_asc, log_table = ctx['g_asc'], ctx['log_table']

    H = np.zeros((gamma, rho), dtype=np.int64)
    for i in range(gamma):
        for j in range(rho):
            marker = g_marker(j - i - 1, g_asc, log_table, n)
            if marker:                      # (t,) -> alpha^t, represented as GF_VEC[t]
                t = marker[0]
                # store as power index t+1 so 0 means "zero element"
                # convention: H[i,j] = t+1  (1-indexed power), or use raw GF_VEC value
                # We store the actual integer representation of alpha^t in GF(q)
                alpha_q = ctx['alpha_q']
                H[i, j] = int(alpha_q ** t)

    if filename:
        from fileio.matrix_io import writer
        writer(filename, H, fieldSize=q, mode='sparse', compress=False)

    return H


# ── Construction 1: binary matrix ─────────────────────────────────────────────

def build_binary_matrix(m: int) -> np.ndarray:
    """
    Build the full binary matrix for Construction 1.
    Shape: (q^2) x (m*(q+1)).
    """
    ctx = setup(m)
    q, n   = ctx['q'], ctx['n']
    GFq    = ctx['GFq']
    alpha_q, g_asc, log_table = ctx['alpha_q'], ctx['g_asc'], ctx['log_table']

    def qtuple(x):
        x = int(x)
        logs = 0 if x == 0 else 1 + log_table[x]
        return tuple(1 if idx == logs else 0 for idx in range(q))

    g_full  = g_asc + [GFq(0)]
    xg_full = [GFq(0)] + g_asc
    anchors = [GFq(0)] + [alpha_q ** i for i in range(q - 1)]
    lambdas = [GFq(0)] + [alpha_q ** k for k in range(q - 1)]

    rows = []
    for anc in anchors:
        for lam in lambdas:
            cw  = [anc * g_full[k] + lam * xg_full[k] for k in range(n)]
            row = []
            for sym in cw:
                row.extend(qtuple(int(sym)))
            rows.append(row)

    return np.array(rows, dtype=np.uint8)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from codegen.analysis.ge import analyse

    m     = 5
    gamma = 12
    rho   = 33
    u     = 0

    q     = 2 ** m
    block = q - 1

    print(f"Building BJ matrix: m={m}  gamma={gamma}  rho={rho}  u={u}")
    H = build_bj_matrix(m, gamma, rho, u)

    info = analyse(H)
    print(info)

    out = os.path.join(os.path.dirname(__file__), 'output',
                       f'bj_m{m}_g{gamma}_r{rho}_u{u}.zip')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    writer(out, H, mode='sparse', compress=True)
    print(f"Saved: {out}")
