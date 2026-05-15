"""
CLI-driven AWGN simulation runner.  Identical logic to runner.py but
all parameters are passed as command-line arguments instead of being
hard-coded at the top of the file.

Usage
-----
  python3 simulation/runner_args.py  <matrix.zip>  <ebno1,ebno2,...>
                                     [--max-iter  50]
                                     [--max-runs  50000000]
                                     [--resolution 100000]
                                     [--max-err   200]
                                     [--constellation BPSK]

Examples
--------
  python3 simulation/runner_args.py \\
      data/essay_examples/example_02/vandermonde_punctured.zip \\
      2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.45,4.5,4.6,4.7,4.8

Output is appended to  <matrix.zip>-AWGN.txt  (same format as runner.py).
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation.modulation.modulation import modulation
from simulation.modulation.demodulation import demodulation
import numpy as np
from time import time, time_ns
from multiprocessing import Pool, cpu_count, freeze_support, Manager
from simulation.modulation.constellation import get_constellation
from fileio.matrix_io import get_reader
from simulation.decoders.bp import make_Flooding_BP_Decoder


def find_info_positions(H):
    K, N = H.shape
    work = H.astype(np.uint8).copy()
    pivot_cols = []
    r = 0
    for col in range(N):
        pivot = next((i for i in range(r, K) if work[i, col]), None)
        if pivot is None:
            continue
        work[[r, pivot]] = work[[pivot, r]]
        rows_to_elim = np.where(work[:, col])[0]
        rows_to_elim = rows_to_elim[rows_to_elim != r]
        work[rows_to_elim] ^= work[r]
        pivot_cols.append(col)
        r += 1
        if r == K:
            break
    return np.array(sorted(set(range(N)) - set(pivot_cols)), dtype=np.int32)


# ── Parse args and set up globals at module level ─────────────────────────────
# (same pattern as runner.py — workers inherit these via fork)

_p = argparse.ArgumentParser(add_help=True)
_p.add_argument('filename')
_p.add_argument('ebno', help='Comma-separated Eb/N0 values in dB')
_p.add_argument('--max-iter',      type=int, default=50)
_p.add_argument('--max-runs',      type=int, default=int(5e7))
_p.add_argument('--resolution',    type=int, default=int(1e5))
_p.add_argument('--max-err',       type=int, default=200)
_p.add_argument('--constellation', type=str, default='BPSK')
_args = _p.parse_args()

filename    = _args.filename
ebno_vec    = [float(x) for x in _args.ebno.split(',')]
max_iter    = _args.max_iter
max_runs    = _args.max_runs
resolution  = _args.resolution
max_err     = _args.max_err
conste_name = _args.constellation

H = get_reader()(filename)
K, N = H.shape

print('Finding information bit positions via GF(2) row reduction ...')
info_pos = find_info_positions(H)
k = len(info_pos)
R = k / N
print(f'H shape: {H.shape}  |  GF(2) rank: {N-k}  |  k: {k}  |  rate: {R:.6f}')
print(f'Eb/N0 points: {ebno_vec}')

num_block_err = np.zeros((len(ebno_vec), 1))
num_bit_err   = np.zeros((len(ebno_vec), 1))
num_iter      = np.zeros((len(ebno_vec), 1))
num_runs      = np.zeros((len(ebno_vec), 1))
constellation_suite = get_constellation(conste_name, N)
m    = constellation_suite[-2]
is2D = constellation_suite[-3]
indx, indy = np.nonzero(H)

round_per_sim = min(1000, resolution)


# ── Worker functions ──────────────────────────────────────────────────────────

def init_worker():
    seed = ((1 << 32) - 1) & (os.getpid() ^ time_ns())
    np.random.seed(seed)


def simulate_multi_round(Err_collected):
    fbp = make_Flooding_BP_Decoder(max_iter=max_iter,
                                   indx=indx, indy=indy, H=K, W=N)
    cur_had_runs, cur_bit_errs, cur_block_errs, cur_iters = [], [], [], []
    for _ in range(round_per_sim):
        cur_had_run   = [None] * len(ebno_vec)
        cur_block_err = [0]    * len(ebno_vec)
        cur_bit_err   = [0]    * len(ebno_vec)
        cur_iter_val  = [0]    * len(ebno_vec)

        x      = np.zeros((N, 1))
        symbol = modulation(x, N, *constellation_suite)
        n      = np.random.randn(N // m, 1) + (1j * np.random.randn(N // m, 1) if is2D else 0)

        for i_ebno, ebno in enumerate(ebno_vec):
            if Err_collected[i_ebno] >= max_err:
                continue
            sigma = np.pow(10, -ebno / 20) / np.sqrt(2 * R * m)
            y   = symbol + sigma * n
            llr = demodulation(y, sigma, N, *constellation_suite).reshape(-1)
            x_hat, iter_cnt = fbp(llr=llr)

            cur_iter_val[i_ebno] = iter_cnt
            if np.any(x_hat):
                cur_had_run[i_ebno]   = False
                cur_block_err[i_ebno] += 1
                cur_bit_err[i_ebno]   += np.sum(x_hat[info_pos])
                Err_collected[i_ebno] += 1
            else:
                cur_had_run[i_ebno] = True

        cur_had_runs.append(cur_had_run)
        cur_bit_errs.append(cur_bit_err)
        cur_block_errs.append(cur_block_err)
        cur_iters.append(cur_iter_val)
    return cur_had_runs, cur_bit_errs, cur_block_errs, cur_iters


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    freeze_support()
    start    = time()
    num_proc = int(cpu_count() * 0.8)
    print(f'{num_proc} processes')
    try:
        with Manager() as manager:
            Err_collected = manager.list([0] * len(ebno_vec))
            cnt = 0
            with Pool(num_proc, initializer=init_worker) as pool:
                for res in pool.imap_unordered(simulate_multi_round,
                                               (Err_collected for _ in range(max_runs // round_per_sim))):
                    for had_run, bit_err, block_err, iters in zip(*res):
                        for idx, (hr, be, bke, it) in enumerate(
                                zip(had_run, bit_err, block_err, iters)):
                            if hr is None:
                                continue
                            num_runs[idx]      += 1
                            num_iter[idx]      += it
                            if not hr:
                                num_block_err[idx] += 1
                                num_bit_err[idx]   += be
                    if all(num_block_err >= max_err):
                        pool.terminate()
                        break
                    cnt += 1
                    if (cnt * round_per_sim) % resolution == 0:
                        rounds_already = cnt * round_per_sim
                        remain_rounds  = 0
                        for i in num_block_err.reshape(-1):
                            if i == 0:
                                remain_rounds = max_runs
                                continue
                            remain_rounds = max(remain_rounds,
                                               int((max_err - i) / i * rounds_already))
                        remain_rounds = min(remain_rounds, max_runs - rounds_already)
                        remain_time   = (time() - start) / rounds_already * remain_rounds
                        print(f'Elapsed: {time()-start:.0f}s  '
                              f'ETA: {remain_time:.0f}s  '
                              f'Rounds: {rounds_already}  '
                              f'Errors: {list(map(int, num_block_err.reshape(-1)))}')
    except KeyboardInterrupt:
        print('Abort')

    print(f'Time elapsed: {time()-start:.1f}s')
    bler     = np.sum(num_block_err, axis=1) / np.sum(num_runs, axis=1)
    ber      = np.sum(num_bit_err,   axis=1) / np.sum(num_runs, axis=1) / k
    ave_iter = np.sum(num_iter,      axis=1) / np.sum(num_runs, axis=1)

    print('  ebno  :' + '\t'.join(map(str, ebno_vec)))
    print('num_runs:' + '\t'.join(map(str, num_runs.reshape(-1))))
    print('ave_bler:' + '\t'.join(map(str, bler)))
    print('ave_ber :' + '\t'.join(map(str, ber)))
    print('ave_iter:' + '\t'.join(map(str, ave_iter)))

    out_path = f'{filename}-AWGN.txt'
    with open(out_path, 'a') as f:
        write = lambda x: f.write(x + '\n')
        write('  ebno  :' + '\t'.join(map(str, ebno_vec)))
        write('num_runs:' + '\t'.join(map(str, num_runs.reshape(-1))))
        write('ave_bler:' + '\t'.join(map(str, bler)))
        write('ave_ber :' + '\t'.join(map(str, ber)))
        write('ave_iter:' + '\t'.join(map(str, ave_iter)))
    print(f'Appended → {out_path}')
