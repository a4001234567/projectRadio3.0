"""
Example 3 — Equivalence verification: BR form ↔ Punctured Vandermonde

Three checks:

  1. Q · S^k = J − P^k_punct  (over F2, for all k = 1 .. P-1)
     where Q = J - I (all-ones with zero diagonal),
           S = BR companion matrix (P-1)×(P-1),
           P^k_punct = top-left (P-1)×(P-1) of lower circulant shift^k.

  2. T · H_BR = H_Vand  (block-by-block over F2)
     where T is the (J·N)×(J·N) block matrix:
       T[0,0] = I,  T[i,0] = J,  T[i,i] = -Q  (i > 0),  all other blocks 0.

  3. T is non-singular  (equivalently: Q is non-singular over F2).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from codegen.constructions.br import make_br_basis
from fileio.matrix_io import get_reader

P   = 37
N   = P - 1          # block size = 36
J_blk = 5            # number of block rows
L_blk = P            # number of block cols

out = os.path.dirname(__file__)

# ── Building blocks ───────────────────────────────────────────────────────────
S = make_br_basis(P).astype(np.int64)
I_blk = np.eye(N, dtype=np.int64)
J_mat = np.ones((N, N), dtype=np.int64)
Q     = J_mat - I_blk   # off-diagonal all-ones

def make_P_punct(k):
    """Top-left N×N of the lower circulant shift^k (over Z)."""
    block = np.zeros((N, N), dtype=np.int64)
    for r in range(N):
        c = (r - k) % P
        if c < N:
            block[r, c] = 1
    return block

reader = get_reader(output_form='ARRAY')
H_br   = reader(os.path.join(out, 'br.zip')).astype(np.int64)
H_vand = reader(os.path.join(out, 'vandermonde_punctured.zip')).astype(np.int64)

lines  = []
passed = True

def log(s=''):
    lines.append(s)
    print(s)

log('=' * 60)
log('Example 3 — Equivalence verification')
log(f'P={P}  N=P-1={N}  J={J_blk}  L={L_blk}')
log('=' * 60)

# ── Check 1: Q · S^k = J − P^k_punct  over F2 ────────────────────────────────
log('\n[1] Q · S^k = J − P^k_punct  (over F2, k = 1 .. P-1)')
Sk  = I_blk.copy()
ok1 = True
for k in range(1, P):
    Sk  = (Sk @ S) % 2
    lhs = (Q @ Sk) % 2
    rhs = (J_mat - make_P_punct(k)) % 2
    if not np.array_equal(lhs, rhs):
        log(f'  FAIL at k={k}')
        ok1 = False
        break
if ok1:
    log(f'  PASS  (verified for k = 1 .. {P-1})')
passed = passed and ok1

# ── Check 2: T · H_BR = H_Vand  block-by-block ───────────────────────────────
log('\n[2] T · H_BR = H_Vand  (block-by-block over F2)')
log('    T: diagonal = [I, -Q, -Q, ...], first-col off-diagonal = J')
ok2 = True
for i in range(J_blk):
    for j in range(L_blk):
        br_ij   = H_br  [i*N:(i+1)*N, j*N:(j+1)*N]
        vand_ij = H_vand[i*N:(i+1)*N, j*N:(j+1)*N]
        br_0j   = H_br  [0:N,          j*N:(j+1)*N]   # block-row 0 = I
        if i == 0:
            result = br_ij % 2                          # T[0,0] = I
        else:
            result = (J_mat @ br_0j - Q @ br_ij) % 2   # J·I + (−Q)·S^{i·j}
        if not np.array_equal(result, vand_ij):
            log(f'  FAIL at block ({i},{j})')
            ok2 = False
            break
    if not ok2:
        break
if ok2:
    log(f'  PASS  (all {J_blk}×{L_blk} blocks)')
passed = passed and ok2

# ── Check 3: T non-singular  (rank(Q) = N over F2) ───────────────────────────
log('\n[3] T non-singular  ↔  rank(Q) = N over F2')
M    = (Q % 2).copy()
rank = 0
for col in range(N):
    pivot = next((r for r in range(rank, N) if M[r, col]), None)
    if pivot is None:
        continue
    M[[rank, pivot]] = M[[pivot, rank]]
    for r in range(N):
        if r != rank and M[r, col]:
            M[r] = (M[r] + M[rank]) % 2
    rank += 1
singular = (rank < N)
log(f'  rank(Q) = {rank}  (N={N})  →  {"non-singular ✓" if not singular else "SINGULAR ✗"}')
passed = passed and not singular

# ── Summary ───────────────────────────────────────────────────────────────────
log('\n' + '=' * 60)
log('Result: ' + ('ALL CHECKS PASSED ✓' if passed else 'SOME CHECKS FAILED ✗'))
log('=' * 60)

verify_path = os.path.join(out, 'verify.txt')
with open(verify_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'\nSaved → {verify_path}')
