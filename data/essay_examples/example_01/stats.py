"""
Compute and save stats for Example 1 matrices.
  - Rates and GF(2) rank
  - 6-cycles: exact via matrix-power formula  tr(B0^3)/6 - sum_v C(dv,3)
  - 8/10-cycles: optimised DFS with SIGALRM timeout
"""
import os, sys, time, signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from fileio.matrix_io import get_reader
from codegen.analysis.cycle_count_fast import count_n_cycles

TIMEOUT_DFS = 55   # seconds for 8- and 10-cycle DFS

class _Timeout(Exception):
    pass

def _handler(signum, frame):
    raise _Timeout()

out = os.path.dirname(__file__)

# ── helpers ───────────────────────────────────────────────────────────────────

def gf2_rank(H):
    M = H.astype(np.uint8).copy()
    m, n = M.shape
    rank = 0
    for col in range(n):
        pivot = None
        for row in range(rank, m):
            if M[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for row in range(m):
            if row != rank and M[row, col]:
                M[row] = (M[row] + M[rank]) % 2
        rank += 1
    return rank

def count_6_cycles_fast(H):
    """
    Exact 6-cycle count via check-check overlap matrix.
    N6 = tr(B0^3)/6  -  sum_v  C(dv, 3)
    Valid for girth >= 6 (no 4-cycles), where B0[i,j] in {0,1}.
    """
    B  = H.astype(np.int64) @ H.astype(np.int64).T   # (m x m) overlap
    np.fill_diagonal(B, 0)                             # B0
    tr_B3 = int(np.trace(B @ B @ B))
    col_degs = H.sum(axis=0).astype(int)
    correction = int(np.sum(col_degs * (col_degs - 1) * (col_degs - 2) // 6))
    return tr_B3 // 6 - correction

def timed_dfs(H, n):
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(TIMEOUT_DFS)
    t0 = time.time()
    try:
        result = count_n_cycles(H, n)
        signal.alarm(0)
        return result, time.time() - t0
    except _Timeout:
        return None, time.time() - t0

# ── load matrices ─────────────────────────────────────────────────────────────

reader = get_reader(output_form='ARRAY')
matrices = {
    'unpunctured': reader(os.path.join(out, 'vandermonde_unpunctured.zip')),
    'punctured':   reader(os.path.join(out, 'vandermonde_punctured.zip')),
    'peg':         reader(os.path.join(out, 'vandermonde_peg.zip')),
}

J, L = 13, 67

lines = []
lines.append('=' * 64)
lines.append('Example 1 — Vandermonde J=13, L=67, block_size=67')
lines.append('=' * 64)

for name, H in matrices.items():
    m, n = H.shape
    lines.append(f'\n[{name}]  shape={H.shape}')

    col_degs = H.sum(axis=0)
    row_degs = H.sum(axis=1)
    lines.append(f'  design rate   = {1 - J/L:.6f}  (1 - {J}/{L})')
    lines.append(f'  col-deg range : {col_degs.min()} – {col_degs.max()}')
    lines.append(f'  row-deg range : {row_degs.min()} – {row_degs.max()}')

    # GF(2) rank
    print(f'  [{name}] computing GF(2) rank …', flush=True)
    t0 = time.time()
    rk = gf2_rank(H)
    k  = n - rk
    lines.append(f'  GF(2) rank    = {rk}   (took {time.time()-t0:.1f}s)')
    lines.append(f'  k             = {k}')
    lines.append(f'  actual rate   = {k/n:.6f}')

    # 6-cycles (fast matrix-power)
    print(f'  [{name}] counting 6-cycles (matrix-power) …', flush=True)
    t0 = time.time()
    n6 = count_6_cycles_fast(H)
    lines.append(f'  6-cycles      : {n6:,}   (took {time.time()-t0:.1f}s)')

    # 8- and 10-cycles (optimised DFS with timeout)
    for cyc in [8, 10]:
        print(f'  [{name}] counting {cyc}-cycles (DFS, timeout={TIMEOUT_DFS}s) …', flush=True)
        cnt, elapsed = timed_dfs(H, cyc)
        if cnt is None:
            lines.append(f'  {cyc}-cycles     : TIMEOUT after {elapsed:.0f}s')
        else:
            lines.append(f'  {cyc}-cycles     : {cnt:,}   (took {elapsed:.1f}s)')

lines.append('\n' + '=' * 64)

stats_path = os.path.join(out, 'stats.txt')
with open(stats_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('\n'.join(lines))
print(f'\nSaved → {stats_path}')
