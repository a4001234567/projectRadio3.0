import os
from simulation.modulation.modulation import modulation
from simulation.modulation.demodulation import demodulation
import numpy as np
from time import time, time_ns
from multiprocessing import Pool, cpu_count, freeze_support, Manager
from simulation.modulation.constellation import get_constellation
from fileio.matrix_io import get_reader
from simulation.decoders.minsum import make_Flooding_SMS_Decoder
from simulation.decoders.bp import make_Flooding_BP_Decoder

# ── Simulation targets ────────────────────────────────────────────────────────
# Uncomment exactly one line to select which matrix to simulate.
# Run from the simulation directory:  python mainNew.py
# Output is appended to  <filename>-AWGN.txt

# 0.2 dB steps throughout; switch to 0.05 dB at right end (BER < ~1e-5)
MOORE_EBNO = [2.8, 3.0, 3.2, 3.4, 3.6, 3.8,
              4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4]

BJ_EBNO    = [2.8, 3.0, 3.2, 3.4, 3.6, 3.8,
              4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4]

# BJ  m=5 gamma=12 rho=33 u=0  N=1023  k=821  rate=0.8025  threshold=2.6573 dB
filename = 'matrices/bj_m5_g12_r33_u0.zip';              ebno_vec = BJ_EBNO

# Moore  J=2 L=10 b=101  N=1000  k=800  rate=0.800  threshold=2.5473 dB
#filename = 'matrices/moore_J2_L10_b101_opt.zip';         ebno_vec = MOORE_EBNO

# Moore  J=2 L=10 b=101  N=1000  PEG version
#filename = 'matrices/moore_J2_L10_b101_opt_peg.zip';     ebno_vec = MOORE_EBNO

# Moore  J=2 L=10 b=107  N=1060  k=848  rate=0.800  threshold=2.5473 dB
#filename = 'matrices/moore_J2_L10_b107_opt.zip';         ebno_vec = MOORE_EBNO

# Moore  J=2 L=10 b=107  N=1060  PEG version
#filename = 'matrices/moore_J2_L10_b107_opt_peg.zip';     ebno_vec = MOORE_EBNO

# ── Simulation parameters ─────────────────────────────────────────────────────
max_iter   = 50
max_runs   = int(5e7)
resolution = int(1e5)
conste_name = 'BPSK'
max_err    = 200

def find_info_positions(H):
    """GF(2) row reduction to find pivot columns; return the non-pivot (information) column indices."""
    K, N = H.shape
    work = H.astype(np.uint8).copy()
    pivot_cols = []
    r = 0
    for col in range(N):
        # find pivot row at or below r
        pivot = next((i for i in range(r, K) if work[i, col]), None)
        if pivot is None:
            continue
        work[[r, pivot]] = work[[pivot, r]]
        # eliminate all other rows
        rows_to_elim = np.where(work[:, col])[0]
        rows_to_elim = rows_to_elim[rows_to_elim != r]
        work[rows_to_elim] ^= work[r]
        pivot_cols.append(col)
        r += 1
        if r == K:
            break
    return np.array(sorted(set(range(N)) - set(pivot_cols)), dtype=np.int32)


# ── Setup (runs at import time so workers inherit state) ──────────────────────
H = get_reader()(filename)
K, N = H.shape

print('Finding information bit positions via GF(2) row reduction ...')
info_pos = find_info_positions(H)
k = len(info_pos)
R = k / N          # true rate from GF(2) rank — used in Eb/N0 -> sigma conversion
print(f'H shape: {H.shape}  |  GF(2) rank: {N-k}  |  k: {k}  |  rate: {R:.6f}')

num_block_err = np.zeros((len(ebno_vec),1))
num_bit_err   = np.zeros((len(ebno_vec),1))
num_iter      = np.zeros((len(ebno_vec),1))
num_runs      = np.zeros((len(ebno_vec),1))
constellation_suite = get_constellation(conste_name, N)
m = constellation_suite[-2]; is2D = constellation_suite[-3]
H_matrix = H.astype('bool')
indx, indy = np.nonzero(H)

def init_worker():
    seed = ((1<<32)-1) & (os.getpid() ^ time_ns())
    seed ^= ((1<<32)-1) & (os.getpid() ^ time_ns())
    np.random.seed(seed)

round_per_sim = min(1000, resolution)

def simulate_multi_round(Err_collected):
    fbp = make_Flooding_BP_Decoder(max_iter=max_iter,
            indx=indx, indy=indy, H=K, W=N)
    cur_had_runs, cur_bit_errs, cur_block_errs, cur_iters = [], [], [], []
    for _ in range(round_per_sim):
        # None=skipped, True=no error, False=error
        cur_had_run  = [None for _ in ebno_vec]
        cur_block_err = [0 for _ in ebno_vec]
        cur_bit_err   = [0 for _ in ebno_vec]
        cur_iter      = [0 for _ in ebno_vec]

        x = np.zeros((N,1))
        symbol = modulation(x, N, *constellation_suite)
        n = np.random.randn(N//m,1) + (1j*np.random.randn(N//m,1) if is2D else 0)
        for i_ebno, ebno in enumerate(ebno_vec):
            if Err_collected[i_ebno] >= max_err:
                continue          # cur_had_run[i_ebno] stays None — not counted
            sigma = np.pow(10,-ebno/20) / np.sqrt(2*R*m)
            y = symbol + sigma * n
            llr = demodulation(y, sigma, N, *constellation_suite).reshape(-1)
            x_hat, iter_this_time = fbp(llr=llr)

            cur_iter[i_ebno] = iter_this_time
            if np.any(x_hat):
                cur_had_run[i_ebno] = False
                cur_block_err[i_ebno] += 1
                cur_bit_err[i_ebno]   += np.sum(x_hat[info_pos])
                Err_collected[i_ebno] += 1
            else:
                cur_had_run[i_ebno] = True
        cur_had_runs.append(cur_had_run)
        cur_bit_errs.append(cur_bit_err)
        cur_block_errs.append(cur_block_err)
        cur_iters.append(cur_iter)
    return cur_had_runs, cur_bit_errs, cur_block_errs, cur_iters

if __name__ == '__main__':
    freeze_support()
    start = time()
    num_proc = int(cpu_count() * 0.8)
    print(f'{num_proc} processes')
    try:
        with Manager() as manager:
            Err_collected = manager.list([0 for _ in range(len(ebno_vec))])
            cnt = 0
            with Pool(num_proc, initializer=init_worker) as pool:
                for res in pool.imap_unordered(simulate_multi_round,
                                               (Err_collected for _ in range(max_runs//round_per_sim))):
                    for had_run, bit_err, block_err, iters in zip(*res):
                        for idx, (i_had_run, i_bit_err, i_block_err, i_iter) in enumerate(
                                zip(had_run, bit_err, block_err, iters)):
                            if i_had_run is None:
                                continue          # skipped — do not count
                            num_runs[idx] += 1
                            num_iter[idx] += i_iter
                            if not i_had_run:     # error
                                num_block_err[idx] += 1
                                num_bit_err[idx]   += i_bit_err
                    if all(num_block_err >= max_err):
                        pool.terminate()
                        break
                    cnt += 1
                    if (cnt * round_per_sim) % resolution == 0:
                        rounds_already = cnt * round_per_sim
                        remain_rounds = 0
                        for i in num_block_err.reshape(-1):
                            if i == 0:
                                remain_rounds = max_runs
                                continue
                            remain_rounds = max(remain_rounds,
                                               int((max_err-i)/i * rounds_already))
                        remain_rounds = min(remain_rounds, max_runs - rounds_already)
                        remain_time = (time()-start) / rounds_already * remain_rounds
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
    with open(f'{filename}-AWGN.txt', 'a') as file:
        write = lambda x: file.write(x + '\n')
        write('  ebno  :' + '\t'.join(map(str, ebno_vec)))
        write('num_runs:' + '\t'.join(map(str, num_runs.reshape(-1))))
        write('ave_bler:' + '\t'.join(map(str, bler)))
        write('ave_ber :' + '\t'.join(map(str, ber)))
        write('ave_iter:' + '\t'.join(map(str, ave_iter)))
