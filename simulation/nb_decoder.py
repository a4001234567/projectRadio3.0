#
#  NB-LDPC Decoder — GF(2^m) EMS decoder module.
#
#  References:
#  [1] BeiDou Navigation Satellite System Signal In Space Interface Control
#      Document Open Service Signal B1C (Version 1.0), December, 2017
#  [2] E.Li et al., Trellis-based Extended Min-Sum algorithm for non-binary LDPC
#      codes and its hardware structure, IEEE Trans. on Communications, 2013
#
import numpy as np
import sympy
import numba
from typing import List, Mapping

# GF(2^m) parameters
x = sympy.symbols('x')
primitive_poly = sympy.poly(x**7 + x + 1, domain='GF(2)')
N_GF = primitive_poly.degree()
Q_GF = 1 << N_GF

NM_EMS   = 32   # EMS candidate list size
MAX_ITER = 15

# GF tables (populated by init_table())
GF_VEC = []
GF_POW = []
GF_MUL = np.zeros((Q_GF, Q_GF), dtype='uint8')


def init_table():
    global GF_VEC, GF_POW, GF_MUL
    if len(GF_VEC):
        return
    _x = sympy.symbols('x')
    _ppoly = sympy.poly(_x**7 + _x + 1, domain='GF(2)')

    def _poly_to_int(poly):
        coeffs = [c % 2 for c in poly.all_coeffs()][::-1]
        return sum(b << i for i, b in enumerate(coeffs))

    cur = sympy.poly(1, _x, domain='GF(2)')
    for _ in range(1, Q_GF):
        GF_VEC.append(_poly_to_int(cur))
        cur = (cur * sympy.poly(_x, domain='GF(2)')) % _ppoly

    GF_POW = [0] * Q_GF
    for i in range(1, Q_GF):
        GF_POW[GF_VEC[i - 1]] = i - 1

    GF_MUL = np.zeros((Q_GF, Q_GF), dtype='uint8')
    for i in range(1, Q_GF):
        for j in range(1, Q_GF):
            GF_MUL[i, j] = GF_VEC[(GF_POW[i] + GF_POW[j]) % (Q_GF - 1)]


def bin2gf(syms):
    n = len(syms) // N_GF
    code = np.zeros(n, dtype='uint8')
    for i in range(n):
        for j in range(N_GF):
            code[i] = (code[i] << 1) + syms[i * N_GF + j]
    return code


def gf2bin(code):
    n = len(code)
    syms = np.zeros(n * N_GF, dtype='uint8')
    for i in range(n):
        for j in range(N_GF):
            syms[i * N_GF + j] = (code[i] >> (N_GF - 1 - j)) & 1
    return syms


def permute_V2C(h, V2C):
    V2C_p = np.zeros(Q_GF, dtype='float32')
    np.put(V2C_p, GF_MUL[h], V2C)
    return V2C_p


def permute_C2V(h, C2V):
    return C2V[GF_MUL[h]].copy()


def ext_min_sum(L1: np.ndarray, L2: np.ndarray) -> np.ndarray:
    idx1 = np.argsort(L1)[:NM_EMS]
    idx2 = np.argsort(L2)[:NM_EMS]
    maxL = L1[idx1[-1]] + L2[idx2[-1]]
    Ls = np.full(Q_GF, maxL, dtype='float64')
    for i in idx1:
        for j in idx2:
            if L1[i] + L2[j] < Ls[i ^ j]:
                Ls[i ^ j] = L1[i] + L2[j]
    return Ls.copy()


def allButOneSum(sumFunc, vals):
    left = [vals[0]]
    n = len(vals)
    for i in range(1, n):
        left.append(sumFunc(left[i - 1], vals[i]))
    right = [vals[-1]]
    for i in range(n - 2, -1, -1):
        right.append(sumFunc(vals[i], right[-1]))
    right.reverse()
    return ([right[1]] +
            [sumFunc(left[i - 1], right[i + 1]) for i in range(1, n - 1)] +
            [left[-2]])


def init_LLR(code, err_prob):
    L = np.zeros((len(code), Q_GF), dtype='float32')
    for i in range(len(code)):
        for j in range(Q_GF):
            nerr = bin(code[i] ^ j).count('1')
            L[i][j] = -np.log(err_prob) * nerr
    return L


def check_parity(ie, je, he, m, code):
    s = np.zeros(m, dtype='uint8')
    for i in range(len(ie)):
        s[ie[i]] ^= GF_MUL[he[i]][code[je[i]]]
    return np.all(s == 0)


def decode_NB_LDPC(H_idx, H_ele, m, n, syms):
    """Decode NB-LDPC codeword.

    Parameters
    ----------
    H_idx  : (ie, je) from np.where(H)
    H_ele  : non-zero values H[H_idx]
    m, n   : matrix rows, cols
    syms   : received binary symbols (n*N_GF,)

    Returns
    -------
    (decoded_syms, nerr) on success; (decoded_syms, -1) on decode failure.
    """
    init_table()
    code = bin2gf(syms)
    ie, je = H_idx
    he = H_ele
    ne = len(he)

    V2C = np.zeros((ne, Q_GF), dtype='float32')
    C2V = np.zeros((ne, Q_GF), dtype='float32')

    L = init_LLR(code, 0.9)
    for i in range(ne):
        V2C[i] = permute_V2C(he[i], L[je[i]])

    jeValInv: Mapping[int, List[int]] = {}
    for idx, val in enumerate(je):
        jeValInv.setdefault(int(val), []).append(idx)

    ieValInv: Mapping[int, List[int]] = {}
    for idx, val in enumerate(ie):
        ieValInv.setdefault(int(val), []).append(idx)

    for _ in range(MAX_ITER):
        if check_parity(ie, je, he, m, code):
            syms_dec = gf2bin(code)
            nerr = int(np.count_nonzero(syms_dec ^ syms))
            return syms_dec[:m * N_GF], nerr

        for iIndex in ieValInv.values():
            C2V[iIndex] = allButOneSum(ext_min_sum, V2C[iIndex])
        for i in range(ne):
            C2V[i] -= np.min(C2V[i])
            C2V[i]  = permute_C2V(he[i], C2V[i])

        for i in range(ne):
            Ls = L[je[i]].copy()
            for j in jeValInv[je[i]]:
                if i != j:
                    Ls += C2V[j]
            Ls -= np.min(Ls)
            V2C[i] = permute_V2C(he[i], Ls)

        for i in range(n):
            for j in jeValInv[i]:
                L[i] += C2V[j]
            L[i] -= np.min(L[i])
            code[i] = np.argmin(L[i])

    return gf2bin(code)[:m * N_GF], -1
