"""
Compute and save stats for Example 2 matrices.
  - Rates and GF(2) rank
  - 6-cycles: exact via matrix-power formula  tr(B0^3)/6 - sum_v C(dv,3)
  - 8/10-cycles: optimised DFS with SIGALRM timeout
"""
import os, sys, time, signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from fileio.matrix_io import get_reader
from codegen.analysis.cycle_count_fast import count_n_cycles

TIMEOUT_DFS = 55

class _Timeout(Exception):
    pass

def _handler(signum, frame):
    raise _Timeout()

out = os.path.dirname(__file__)

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
    B  = H.astype(np.int64) @ H.astype(np.int64).T
    np.fill_diagonal(B, 0)
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

reader = get_reader(output_form='ARRAY')
matrices = {
    'punctured': reader(os.path.join(out, 'vandermonde_punctured.zip')),
    '6-free':    reader(os.path.join(out, 'vandermonde_6free.zip')),
    'peg':       reader(os.path.join(out, 'vandermonde_6free_peg.zip')),
}

J, L = 3, 13

lines = []
lines.append('=' * 64)
lines.append('Example 2 — Vandermonde J=3, L=13, block_size=67')
lines.append('=' * 64)

for name, H in matrices.items():
    m, n = H.shape
    lines.append(f'\n[{name}]  shape={H.shape}')

    col_degs = H.sum(axis=0)
    row_degs = H.sum(axis=1)
    lines.append(f'  design rate   = {1 - J/L:.6f}  (1 - {J}/{L})')
    lines.append(f'  col-deg range : {col_degs.min()} – {col_degs.max()}')
    lines.append(f'  row-deg range : {row_degs.min()} – {row_degs.max()}')

    print(f'  [{name}] computing GF(2) rank …', flush=True)
    t0 = time.time()
    rk = gf2_rank(H)
    k  = n - rk
    lines.append(f'  GF(2) rank    = {rk}   (took {time.time()-t0:.1f}s)')
    lines.append(f'  k             = {k}')
    lines.append(f'  actual rate   = {k/n:.6f}')

    print(f'  [{name}] counting 6-cycles …', flush=True)
    t0 = time.time()
    n6 = count_6_cycles_fast(H)
    lines.append(f'  6-cycles      : {n6:,}   (took {time.time()-t0:.1f}s)')

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
