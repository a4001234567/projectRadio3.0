"""
NB-LDPC simulation runner (argument-driven).

Usage
-----
  python3 simulation/runner_nb.py  matrix.zip  5.0,5.5,6.0
                                   [--max-err 100]   [--max-runs 10000]
                                   [--max-iter 10]   [--nm-ems 32]
                                   [--ratio 0.5]     [--resolution 100]
                                   [--round-per-sim 100]

Output appended to <matrix>-NB-AWGN.txt.
Primitive polynomial fixed to x^7+x+1 (GF(128), matching B1C standard).
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sympy
import numba
from typing import List, Mapping
from fileio.matrix_io import get_reader
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, freeze_support, Manager
from time import time

# ── parse args at module level so forked workers inherit all globals ───────────

_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument('filename',        help='NB-LDPC matrix file (zip or txt)')
_p.add_argument('ebno',            help='Comma-separated Eb/N0 list, e.g. 5.0,5.5,6.0')
_p.add_argument('--max-err',       type=int,   default=100,
                help='Block errors to collect per SNR point (default 100)')
_p.add_argument('--max-runs',      type=int,   default=10000,
                help='Max total rounds (default 10000)')
_p.add_argument('--max-iter',      type=int,   default=10,
                help='BP iterations (default 10)')
_p.add_argument('--nm-ems',        type=int,   default=32,
                help='EMS candidate list size (default 32)')
_p.add_argument('--ratio',         type=float, default=0.5,
                help='Worker/CPU ratio (default 0.5)')
_p.add_argument('--resolution',    type=int,   default=100,
                help='Print-progress interval in rounds (default 100)')
_p.add_argument('--round-per-sim', type=int,   default=100,
                help='Rounds per pool task (default 100)')
_args = _p.parse_args()

filename      = _args.filename
ebno_vec      = [float(x) for x in _args.ebno.split(',')]
max_err       = _args.max_err
MAX_RUNS      = _args.max_runs
MAX_ITER      = _args.max_iter
NM_EMS        = _args.nm_ems
ratio         = _args.ratio
resolution    = _args.resolution
round_per_sim = _args.round_per_sim

# ── GF(2^7) tables ────────────────────────────────────────────────────────────

_x = sympy.symbols('x')
_primitive_poly = sympy.poly(_x**7 + _x + 1, domain='GF(2)')
N_GF = _primitive_poly.degree()
Q_GF = 1 << N_GF

def _poly_to_int(poly):
    coeffs = [c % 2 for c in poly.all_coeffs()][::-1]
    return sum(b << i for i, b in enumerate(coeffs))

GF_VEC = []
_cur = sympy.poly(1, _x, domain='GF(2)')
for _i in range(1, Q_GF):
    GF_VEC.append(_poly_to_int(_cur))
    _cur = (_cur * sympy.poly(_x, domain='GF(2)')) % _primitive_poly

GF_POW = [0] * Q_GF
for _i in range(1, Q_GF):
    GF_POW[GF_VEC[_i - 1]] = _i - 1

GF_MUL = np.zeros((Q_GF, Q_GF), dtype='uint8')
for _i in range(1, Q_GF):
    for _j in range(1, Q_GF):
        GF_MUL[_i, _j] = GF_VEC[(GF_POW[_i] + GF_POW[_j]) % (Q_GF - 1)]

# ── load matrix ───────────────────────────────────────────────────────────────

HcheckMat = get_reader(output_form='ARRAY')(filename)
H_idx     = np.where(HcheckMat)
H_ele     = HcheckMat[H_idx]
H, W      = HcheckMat.shape
R         = 1.0 - H / W
ie, je    = H_idx
he        = H_ele
ne        = len(he)

jeValInv: Mapping[int, List[int]] = {}
for _idx, _val in enumerate(je):
    jeValInv.setdefault(int(_val), []).append(_idx)

ieValInv: Mapping[int, List[int]] = {}
for _idx, _val in enumerate(ie):
    ieValInv.setdefault(int(_val), []).append(_idx)

print(f'{os.path.basename(filename)}  shape={H}×{W}  R={R:.6f}  GF({Q_GF})')
print(f'NM_EMS={NM_EMS}  MAX_ITER={MAX_ITER}  max_err={max_err}')

# ── JIT-compiled EMS functions ────────────────────────────────────────────────

@numba.jit(nopython=True)
def permute_V2C(h, V2C):
    V2C_p = np.zeros(Q_GF, dtype='float64')
    for i in range(Q_GF):
        V2C_p[GF_MUL[h][i]] = V2C[i]
    return V2C_p

@numba.jit(nopython=True)
def permute_C2V(h, C2V):
    C2V_p = np.zeros(Q_GF, dtype='float64')
    for i in range(Q_GF):
        C2V_p[i] = C2V[GF_MUL[h][i]]
    return C2V_p

@numba.jit(nopython=True)
def ext_min_sum(L1, L2):
    if len(L1) == 0:
        return L2
    idx1 = np.argsort(L1)
    idx2 = np.argsort(L2)
    maxL = L1[idx1[NM_EMS - 1]] + L2[idx2[NM_EMS - 1]]
    Ls = np.full(Q_GF, maxL, dtype='float64')
    for i in idx1[:NM_EMS]:
        for j in idx2[:NM_EMS]:
            if L1[i] + L2[j] < Ls[i ^ j]:
                Ls[i ^ j] = L1[i] + L2[j]
    return Ls

@numba.jit(nopython=True)
def initLLR(naiveLLR, codeLength):
    L = np.zeros((codeLength, Q_GF), dtype='float64')
    for i in range(codeLength):
        for j in range(Q_GF):
            for bit in range(N_GF):
                if j & (1 << bit):
                    L[i, j] += naiveLLR[i * N_GF + bit, 0]
        L[i] -= np.min(L[i])
    return L

def allButOneSum(sumFunc, vals):
    left = [vals[0]]
    n = len(vals)
    for i in range(1, n):
        left.append(sumFunc(left[i - 1], vals[i]))
    right = [vals[-1]]
    for i in range(n - 2, -1, -1):
        right.append(sumFunc(vals[i], right[-1]))
    right.reverse()
    result = ([right[1]] +
              [sumFunc(left[i - 1], right[i + 1]) for i in range(1, n - 1)] +
              [left[-2]])
    return result

def _add(a, b): return a + b

def doIterDecode(C2V, V2C, L, code):
    """Run MAX_ITER BP iterations.  Returns (hasBlockError, numSymbolErrors)."""
    for _ in range(MAX_ITER):
        for iIndex in ieValInv.values():
            C2V[iIndex] = allButOneSum(ext_min_sum, V2C[iIndex])
        for i in range(ne):
            C2V[i] -= np.min(C2V[i])
            C2V[i]  = permute_C2V(he[i], C2V[i])
        for jIndex in jeValInv.values():
            V2C[jIndex] = allButOneSum(_add, C2V[jIndex])
        for i in range(ne):
            V2C[i] -= np.min(V2C[i])
            V2C[i]  = permute_V2C(he[i], V2C[i])
        for i in range(W):
            for j in jeValInv[i]:
                L[i] += C2V[j]
            L[i] -= np.min(L[i])
            code[i] = np.argmin(L[i])
        if not np.any(code):
            return False, 0
    num_sym_err = int(np.sum(code != 0))
    return bool(np.any(code)), num_sym_err

# ── multi-round worker ────────────────────────────────────────────────────────

def simulate_multi_round(Err_collected):
    had_runs   = []
    bit_errs   = []
    block_errs = []
    for _ in range(round_per_sim):
        had_run   = [None] * len(ebno_vec)
        block_err = [0]    * len(ebno_vec)
        bit_err   = [0]    * len(ebno_vec)
        codeX   = np.zeros((W * N_GF, 1))
        symbolX = 1.0 - 2.0 * codeX
        noise   = np.random.randn(W * N_GF, 1)
        for i_ebno, ebno in enumerate(ebno_vec):
            if Err_collected[i_ebno] >= max_err:
                continue
            sigma    = np.power(10.0, -ebno / 20.0) / np.sqrt(2.0 * R)
            y        = symbolX + sigma * noise
            naiveLLR = 2.0 * y / (sigma ** 2)
            L        = initLLR(naiveLLR, W)
            V2C      = np.zeros((ne, Q_GF), dtype='float64')
            C2V      = np.zeros((ne, Q_GF), dtype='float64')
            for i in range(ne):
                V2C[i] = permute_V2C(he[i], L[je[i]])
            code = np.zeros(W, dtype='uint8')
            hasErr, numSymErr = doIterDecode(C2V, V2C, L, code)
            if hasErr:
                had_run[i_ebno]   = False
                block_err[i_ebno] = 1
                bit_err[i_ebno]   = numSymErr
                Err_collected[i_ebno] += 1
            else:
                had_run[i_ebno] = True
        had_runs.append(had_run)
        bit_errs.append(bit_err)
        block_errs.append(block_err)
    return had_runs, bit_errs, block_errs

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    freeze_support()
    num_block_err = np.zeros(len(ebno_vec))
    num_sym_err   = np.zeros(len(ebno_vec))
    num_runs      = np.zeros(len(ebno_vec))
    num_proc      = max(1, int(cpu_count() * ratio))
    out_file      = f'{filename}-NB-AWGN.txt'

    print(f'Workers: {num_proc}  MAX_RUNS: {MAX_RUNS}  round_per_sim: {round_per_sim}')
    start = time()
    try:
        with Manager() as manager:
            Err_collected = manager.list([0] * len(ebno_vec))
            cnt = 0
            with Pool(num_proc) as pool:
                for had_runs, bit_errs, block_errs in pool.imap_unordered(
                        simulate_multi_round,
                        (Err_collected for _ in range(MAX_RUNS // round_per_sim))):
                    for had_run, bit_err, block_err in zip(had_runs, bit_errs, block_errs):
                        for idx, (hr, be, bl) in enumerate(zip(had_run, bit_err, block_err)):
                            if hr is False:
                                if num_block_err[idx] >= max_err:
                                    continue
                                num_block_err[idx] += 1
                                num_sym_err[idx]   += be
                                num_runs[idx]      += 1
                            elif hr is True:
                                num_runs[idx] += 1
                    if all(num_block_err >= max_err):
                        pool.terminate()
                        break
                    cnt += 1
                    if (cnt * round_per_sim) % resolution == 0:
                        elapsed = time() - start
                        done = cnt * round_per_sim
                        print(f'  {done:6d} rounds  elapsed={elapsed:.0f}s  '
                              f'block_errs={list(map(int, num_block_err))}')
    except KeyboardInterrupt:
        print('Interrupted')

    elapsed = time() - start
    bler = np.where(num_runs > 0, num_block_err / num_runs, 0.0)
    ber  = np.where(num_runs > 0, num_sym_err   / num_runs / W, 0.0)

    print(f'\nDone in {elapsed:.1f}s')
    hdr = f'  ebno      : ' + '  '.join(f'{e:.2f}' for e in ebno_vec)
    blerln = '  ave_bler  : ' + '  '.join(f'{v:.4e}' for v in bler)
    berln  = '  ave_ber   : ' + '  '.join(f'{v:.4e}' for v in ber)
    runsln = '  num_runs  : ' + '  '.join(f'{int(r)}'  for r in num_runs)
    for line in (hdr, blerln, berln, runsln):
        print(line)

    with open(out_file, 'a') as f:
        f.write(hdr   + '\n')
        f.write(blerln + '\n')
        f.write(berln  + '\n')
        f.write(runsln + '\n')
    print(f'Appended to {out_file}')
