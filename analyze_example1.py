"""
Example 1 Analysis: GA-DE threshold + quick simulation sweep
Vandermonde J=13, L=67, block_size=67
  Unpunctured : truncate=0 → 871×4489 (13,67)-regular
  Punctured   : truncate=3 → 832×4288 (13,64)-regular
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from fileio.matrix_io import get_reader
from de.awgn import find_thres_from_books, convert_snr_to_ebno

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GA-DE threshold
# ─────────────────────────────────────────────────────────────────────────────
EX1 = os.path.join(os.path.dirname(__file__),
                   'data', 'essay_examples', 'example_01')

def code_params(H):
    K, N = H.shape
    # GF(2) rank via row-reduction
    work = H.astype(np.uint8).copy()
    pivots = 0
    for col in range(N):
        row = next((i for i in range(pivots, K) if work[i, col]), None)
        if row is None:
            continue
        work[[pivots, row]] = work[[row, pivots]]
        mask = work[:, col].astype(bool)
        mask[pivots] = False
        work[mask] ^= work[pivots]
        pivots += 1
        if pivots == K:
            break
    rank = pivots
    k    = N - rank
    R    = k / N
    return K, N, rank, k, R

def ga_threshold(var_book, chk_book, R):
    thres_ebno = find_thres_from_books(var_book, chk_book, R)
    shannon_snr = np.power(10, -thres_ebno / 10)   # σ² in linear
    # Shannon limit Eb/N0 for rate R, BPSK-AWGN: C = 0.5*log2(1+2R*Eb/N0)
    # at capacity: R = 0.5*log2(1 + 2R * Eb_N0_lin)
    # → Eb_N0_lin = (2^(2R)-1)/(2R)
    shannon_eb_lin = (np.power(2, 2*R) - 1) / (2*R)
    shannon_eb_db  = 10*np.log10(shannon_eb_lin)
    gap_db = thres_ebno - shannon_eb_db
    return thres_ebno, shannon_eb_db, gap_db

print("=" * 60)
print("EXAMPLE 1  —  Vandermonde J=13  L=67  P=67")
print("=" * 60)

for label, fname, truncate in [
        ('Unpunctured (truncate=0)', 'vandermonde_unpunctured.zip', 0),
        ('Punctured   (truncate=3)', 'vandermonde_punctured.zip',   3),
]:
    H = get_reader()(os.path.join(EX1, fname))
    K, N, rank, k, R = code_params(H)
    col_degs = np.unique(H.sum(0))
    row_degs = np.unique(H.sum(1))
    print(f"\n{label}")
    print(f"  Shape     : {K} × {N}")
    print(f"  GF2 rank  : {rank}  →  k={k}  rate={R:.6f}")
    print(f"  Col-deg   : {col_degs}")
    print(f"  Row-deg   : {row_degs}")

    # Build degree books directly from H (exact, no recipe needed)
    col_deg_hist = {}
    for d in H.sum(0).astype(int):
        col_deg_hist[d] = col_deg_hist.get(d, 0) + 1
    row_deg_hist = {}
    for d in H.sum(1).astype(int):
        row_deg_hist[d] = row_deg_hist.get(d, 0) + 1

    thres, shannon, gap = ga_threshold(col_deg_hist, row_deg_hist, R)
    print(f"  GA threshold: {thres:.4f} dB")
    print(f"  Shannon lim : {shannon:.4f} dB  (gap = {gap:.4f} dB)")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Quick simulation sweep — unpunctured matrix only
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("QUICK SIMULATION SWEEP  (unpunctured, BP decoder)")
print("=" * 60)

from simulation.modulation.modulation import modulation
from simulation.modulation.demodulation import demodulation
from simulation.modulation.constellation import get_constellation
from simulation.decoders.bp import make_Flooding_BP_Decoder

H = get_reader()(os.path.join(EX1, 'vandermonde_unpunctured.zip'))
K, N, rank, k, R = code_params(H)
indx, indy = np.nonzero(H)
constellation_suite = get_constellation('BPSK', N)
m_bits = constellation_suite[-2]
is2D   = constellation_suite[-3]

def run_sweep(ebno_vec, n_frames=500, max_iter=50, seed=42):
    np.random.seed(seed)
    fbp = make_Flooding_BP_Decoder(max_iter=max_iter, indx=indx, indy=indy, H=K, W=N)
    block_errs = np.zeros(len(ebno_vec), dtype=int)
    bit_errs   = np.zeros(len(ebno_vec), dtype=int)
    runs       = np.zeros(len(ebno_vec), dtype=int)

    # info positions (non-pivot columns)
    work = H.astype(np.uint8).copy()
    pivots_list = []
    piv = 0
    for col in range(N):
        row = next((i for i in range(piv, K) if work[i, col]), None)
        if row is None:
            continue
        work[[piv, row]] = work[[row, piv]]
        mask = work[:, col].astype(bool); mask[piv] = False
        work[mask] ^= work[piv]
        pivots_list.append(col); piv += 1
        if piv == K: break
    info_pos = np.array(sorted(set(range(N)) - set(pivots_list)), dtype=np.int32)

    x = np.zeros((N, 1))
    for _ in range(n_frames):
        symbol = modulation(x, N, *constellation_suite)
        noise  = np.random.randn(N // m_bits, 1)
        for i_e, ebno in enumerate(ebno_vec):
            sigma = np.power(10, -ebno / 20) / np.sqrt(2 * R * m_bits)
            y   = symbol + sigma * noise
            llr = demodulation(y, sigma, N, *constellation_suite).reshape(-1)
            x_hat, _ = fbp(llr=llr)
            runs[i_e] += 1
            if np.any(x_hat):
                block_errs[i_e] += 1
                bit_errs[i_e]   += int(np.sum(x_hat[info_pos]))

    bler = block_errs / runs
    ber  = bit_errs  / runs / k
    return bler, ber, runs

# Coarse sweep around the expected threshold
ebno_coarse = np.arange(2.0, 6.1, 0.4)
print(f"\nCoarse sweep: {list(np.round(ebno_coarse, 2))} dB  ({500} frames each)")
bler_c, ber_c, runs_c = run_sweep(ebno_coarse, n_frames=500)
print(f"{'EbNo':>8}  {'BLER':>10}  {'BER':>12}  {'BlkErrs':>8}")
for e, bl, be, r in zip(ebno_coarse, bler_c, ber_c, runs_c):
    berr = int(round(bl * r))
    print(f"{e:8.2f}  {bl:10.4e}  {be:12.4e}  {berr:8d}")

# Identify the knee region (first EbNo where BER starts to drop sharply)
valid = (ber_c > 0)
if valid.any():
    knee_idx = np.where(valid)[0][-1]  # last point with errors
    knee_ebno = ebno_coarse[knee_idx]
else:
    knee_ebno = ebno_coarse[-2]

print(f"\nKnee region approximately at {knee_ebno:.1f} dB")

# Fine sweep near the knee
fine_start = max(knee_ebno - 0.4, ebno_coarse[0])
ebno_fine  = np.round(np.arange(fine_start, knee_ebno + 1.0, 0.2), 2)
print(f"\nFine sweep (0.2 dB step): {list(ebno_fine)} dB  ({800} frames each)")
bler_f, ber_f, runs_f = run_sweep(ebno_fine, n_frames=800, seed=123)
print(f"{'EbNo':>8}  {'BLER':>10}  {'BER':>12}  {'BlkErrs':>8}")
for e, bl, be, r in zip(ebno_fine, bler_f, ber_f, runs_f):
    berr = int(round(bl * r))
    print(f"{e:8.2f}  {bl:10.4e}  {be:12.4e}  {berr:8d}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Recommended EbNo range
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RECOMMENDED EbNo RANGE")
print("=" * 60)

# Find region where BER transitions from ~1e-2 to near zero
all_e = np.concatenate([ebno_coarse, ebno_fine])
all_b = np.concatenate([ber_c, ber_f])
order = np.argsort(all_e)
all_e = all_e[order]; all_b = all_b[order]

# Remove duplicates
_, idx = np.unique(all_e, return_index=True)
all_e = all_e[idx]; all_b = all_b[idx]

# High-BER start: first point with BER ≥ 1e-3
high_mask = all_b >= 1e-3
if high_mask.any():
    e_start = all_e[np.where(high_mask)[0][0]]
else:
    e_start = all_e[0]

# Error-floor region end: last point with any errors
err_mask = all_b > 0
if err_mask.any():
    e_cliff = all_e[np.where(err_mask)[0][-1]]
else:
    e_cliff = all_e[-1]

print(f"\nBased on quick sweep (500–800 frames):")
print(f"  BER ≥ 1e-3 region starts at ~{e_start:.1f} dB")
print(f"  Last observed error at       ~{e_cliff:.1f} dB")
print()
print("Suggested simulation EbNo list (targeting BER down to ~1e-8):")

# Build recommended range
e_low  = max(2.0, round(e_start - 0.4, 1))
e_high = e_cliff + 1.0   # extend 1 dB past cliff for fine region

main_pts = list(np.arange(e_low, e_cliff - 0.1, 0.2).round(2))
fine_pts = list(np.arange(e_cliff - 0.1, e_high + 0.01, 0.05).round(2))
recommended = sorted(set(main_pts + fine_pts))
print(f"  {recommended}")
print()
print("  Convention used: 0.2 dB steps in body, 0.05 dB near right boundary")
print("  Run full simulation with max_err=200 to reach BER ~1e-8")
