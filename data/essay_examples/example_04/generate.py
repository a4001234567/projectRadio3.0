"""
Generate Example 4 — Moore QC-LDPC variants.
J=3, L=36, N=227, truncate=1  =>  shape 678×8136, rate 11/12.

Variants
--------
moore_wt1.zip          : wt1, identity_col=True  (block MDS ✓)
moore_wt1_noshift.zip  : wt1, identity_col=False (block MDS ✗)
moore_wt2.zip          : wt2, independent Golomb ruler pairs (girth ≥ 6, MDS ✓)
moore_wt2_diagonal.zip : wt2, diagonal A=(1,48), rulers[i][j]=A<<(i+j) (girth ≥ 6, MDS ✓)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from codegen.constructions.moore import build_moore_matrix
from codegen.qc.ruler import CircularRuler
from codegen.qc.grid import writeNDarray
from fileio.matrix_io import writer as save_matrix

OUT  = os.path.dirname(__file__)
J, L, N, T = 3, 36, 227, 1

def path(name):
    return os.path.join(OUT, name)

# ── wt1 with identity_col shift ───────────────────────────────────────────────
print(f"Building wt1 (shift)  N={N} J={J} L={L} …")
H = build_moore_matrix(J, L, N, weight=1, truncate=T, identity_col=True,
                       filename=path('moore_wt1.zip'), return_array=True)
print(f"  shape={H.shape}  col-deg={sorted(set(H.sum(0).tolist()))}")

# ── wt1 without shift ─────────────────────────────────────────────────────────
print(f"Building wt1 (no-shift)  N={N} J={J} L={L} …")
H = build_moore_matrix(J, L, N, weight=1, truncate=T, identity_col=False,
                       filename=path('moore_wt1_noshift.zip'), return_array=True)
print(f"  shape={H.shape}  col-deg={sorted(set(H.sum(0).tolist()))}")

# ── wt2 independent rulers ────────────────────────────────────────────────────
print(f"Building wt2 (independent)  N={N} J={J} L={L} …")
H = build_moore_matrix(J, L, N, weight=2, truncate=T,
                       filename=path('moore_wt2.zip'), return_array=True)
print(f"  shape={H.shape}  col-deg={sorted(set(H.sum(0).tolist()))}")

# ── wt2 diagonal  rulers[i][j] = A << (i+j) ──────────────────────────────────
print(f"Building wt2 (diagonal)  N={N} J={J} L={L}  A=(1,48) …")
A = CircularRuler(N, (1, 48))
rulers = [[A << (i + j) for j in range(L)] for i in range(J)]
H = writeNDarray(rulers, N, truncate=T).astype(np.uint8)
save_matrix(path('moore_wt2_diagonal.zip'), H, mode='sparse', compress=True,
            comments=[f'Diagonal Moore wt2  J={J} L={L} N={N}  A=(1,48)  rulers[i][j]=A<<(i+j)'])
print(f"  shape={H.shape}  col-deg={sorted(set(H.sum(0).tolist()))}")

print("\nDone — example_04")
