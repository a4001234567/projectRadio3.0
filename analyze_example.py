"""
Universal example analyser — GA-DE threshold + coarse/fine simulation sweep.

Usage
-----
  python3 analyze_example.py  path/to/matrix1.zip  [path/to/matrix2.zip ...]
                              [--coarse-start 2.0]  [--coarse-end 7.0]
                              [--coarse-step  0.4]
                              [--coarse-frames 1000]  [--fine-frames 1500]
                              [--min-errors 3]
                              [--max-iter 50]
                              [--jobs N]   (parallel workers, default = #matrices)

Outputs a recommended Eb/N0 list for the full simulation run and per-variant
sweep tables.  Range is derived from the worst-case knee across all variants,
with the fine-density region kept compact (knee ± 0.5 dB at 0.05 dB steps).
"""
import io, os, sys, argparse
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from fileio.matrix_io import get_reader
from de.awgn import find_thres_from_books
from simulation.modulation.modulation import modulation
from simulation.modulation.demodulation import demodulation
from simulation.modulation.constellation import get_constellation
from simulation.decoders.bp import make_Flooding_BP_Decoder

# ── helpers ───────────────────────────────────────────────────────────────────

def code_params(H):
    K, N = H.shape
    work = H.astype(np.uint8).copy()
    pivots = 0
    for col in range(N):
        row = next((i for i in range(pivots, K) if work[i, col]), None)
        if row is None:
            continue
        work[[pivots, row]] = work[[row, pivots]]
        mask = work[:, col].astype(bool); mask[pivots] = False
        work[mask] ^= work[pivots]
        pivots += 1
        if pivots == K:
            break
    k = N - pivots
    return K, N, pivots, k, k / N

def info_positions(H):
    K, N = H.shape
    work = H.astype(np.uint8).copy()
    pivots_list, piv = [], 0
    for col in range(N):
        row = next((i for i in range(piv, K) if work[i, col]), None)
        if row is None: continue
        work[[piv, row]] = work[[row, piv]]
        mask = work[:, col].astype(bool); mask[piv] = False
        work[mask] ^= work[piv]
        pivots_list.append(col); piv += 1
        if piv == K: break
    return np.array(sorted(set(range(N)) - set(pivots_list)), dtype=np.int32)

def ga_threshold(H, R):
    col_hist = {}
    for d in H.sum(0).astype(int):
        col_hist[d] = col_hist.get(d, 0) + 1
    row_hist = {}
    for d in H.sum(1).astype(int):
        row_hist[d] = row_hist.get(d, 0) + 1
    thres   = find_thres_from_books(col_hist, row_hist, R)
    shannon = 10 * np.log10((np.power(2, 2*R) - 1) / (2*R))
    return thres, shannon, thres - shannon

def run_sweep(H, ebno_vec, n_frames, max_iter=50, seed=42):
    np.random.seed(seed)
    K, N = H.shape
    _, _, _, k, R = code_params(H)
    info_pos = info_positions(H)
    indx, indy = np.nonzero(H)
    cs     = get_constellation('BPSK', N)
    m_bits = cs[-2]
    fbp    = make_Flooding_BP_Decoder(max_iter=max_iter, indx=indx, indy=indy, H=K, W=N)

    block_errs = np.zeros(len(ebno_vec), dtype=int)
    bit_errs   = np.zeros(len(ebno_vec), dtype=int)
    runs       = np.zeros(len(ebno_vec), dtype=int)
    x = np.zeros((N, 1))

    for _ in range(n_frames):
        symbol = modulation(x, N, *cs)
        noise  = np.random.randn(N // m_bits, 1)
        for i_e, ebno in enumerate(ebno_vec):
            sigma = np.power(10, -ebno / 20) / np.sqrt(2 * R * m_bits)
            y   = symbol + sigma * noise
            llr = demodulation(y, sigma, N, *cs).reshape(-1)
            x_hat, _ = fbp(llr=llr)
            runs[i_e] += 1
            if np.any(x_hat):
                block_errs[i_e] += 1
                bit_errs[i_e]   += int(np.sum(x_hat[info_pos]))

    bler = block_errs / runs
    ber  = bit_errs  / (runs * k)
    return bler, ber, runs, block_errs

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('matrices', nargs='+', help='Matrix zip file paths')
    p.add_argument('--coarse-start',  type=float, default=2.0)
    p.add_argument('--coarse-end',    type=float, default=7.0)
    p.add_argument('--coarse-step',   type=float, default=0.4)
    p.add_argument('--coarse-frames', type=int,   default=1000)
    p.add_argument('--fine-frames',   type=int,   default=1500)
    p.add_argument('--min-errors',    type=int,   default=3,
                   help='Min block errors required to count a point as the knee')
    p.add_argument('--max-iter',      type=int,   default=50)
    p.add_argument('--jobs',          type=int,   default=0,
                   help='Parallel workers (default: number of matrices)')
    return p.parse_args()

def fmt_list(v):
    """Return a clean Python list literal (no np.float64 wrappers)."""
    return '[' + ', '.join(f'{float(x):.2f}' for x in v) + ']'

# ── per-matrix worker (module-level so it is picklable) ───────────────────────

def _analyze_one(packed):
    """Analyze one matrix file; return (output_text, knee)."""
    fpath, args = packed
    buf = io.StringIO()

    def p(s='', **kw):
        kw.pop('flush', None)
        print(s, file=buf, **kw)

    reader = get_reader(output_form='ARRAY')
    ebno_coarse = np.round(np.arange(args.coarse_start,
                                     args.coarse_end + 1e-9,
                                     args.coarse_step), 4)

    label = os.path.basename(fpath)
    p(f"\n{'='*60}")
    p(f"  {label}")
    p(f"{'='*60}")

    H = reader(fpath)
    K, N, rank, k, R = code_params(H)
    p(f"  Shape  : {K} × {N}   rank={rank}   k={k}   rate={R:.6f}")
    p(f"  Col-deg: {np.unique(H.sum(0))}   Row-deg: {np.unique(H.sum(1))}")

    try:
        thres, shannon, gap = ga_threshold(H, R)
        p(f"  GA threshold : {thres:.4f} dB   Shannon : {shannon:.4f} dB   gap : {gap:.4f} dB")
    except Exception as e:
        p(f"  GA threshold : (failed — {e})")

    # coarse sweep
    p(f"\n  Coarse sweep ({args.coarse_frames} frames, step={args.coarse_step} dB):")
    bler_c, ber_c, runs_c, errs_c = run_sweep(H, ebno_coarse, args.coarse_frames,
                                               max_iter=args.max_iter)
    p(f"  {'EbNo':>6}  {'BLER':>10}  {'BER':>12}  {'BlkErr':>7}")
    for e, bl, be, r in zip(ebno_coarse, bler_c, ber_c, runs_c):
        p(f"  {float(e):6.2f}  {bl:10.4e}  {be:12.4e}  {int(round(bl*r)):7d}")

    # knee: last point with >= min_errors block errors
    solid = errs_c >= args.min_errors
    if solid.any():
        knee = float(ebno_coarse[np.where(solid)[0][-1]])
    elif errs_c.any():
        knee = float(ebno_coarse[np.where(errs_c > 0)[0][-1]])
    else:
        knee = float(ebno_coarse[-2])
    p(f"\n  Knee ≈ {knee:.2f} dB  (requires ≥ {args.min_errors} block errors)")

    # fine sweep: knee ± 0.2 dB at 0.05 dB steps
    f_start   = max(args.coarse_start, round(knee - 0.2, 2))
    f_end     = round(knee + 0.2, 2)
    ebno_fine = np.round(np.arange(f_start, f_end + 1e-9, 0.05), 4)
    p(f"  Fine sweep ({args.fine_frames} frames, step=0.05 dB):")
    bler_f, ber_f, runs_f, errs_f = run_sweep(H, ebno_fine, args.fine_frames,
                                               max_iter=args.max_iter, seed=123)
    p(f"  {'EbNo':>6}  {'BLER':>10}  {'BER':>12}  {'BlkErr':>7}")
    for e, bl, be, r in zip(ebno_fine, bler_f, ber_f, runs_f):
        p(f"  {float(e):6.2f}  {bl:10.4e}  {be:12.4e}  {int(round(bl*r)):7d}")

    return buf.getvalue(), knee

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    n_jobs  = args.jobs if args.jobs > 0 else len(args.matrices)
    n_jobs  = min(n_jobs, len(args.matrices))

    packed  = [(fpath, args) for fpath in args.matrices]

    if n_jobs == 1:
        results = [_analyze_one(p) for p in packed]
    else:
        with Pool(n_jobs) as pool:
            results = pool.map(_analyze_one, packed)

    all_knees = []
    for text, knee in results:
        print(text, end='', flush=True)
        all_knees.append(knee)

    # ── recommended range ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RECOMMENDED Eb/N0 RANGE")
    print(f"{'='*60}")

    global_knee = max(all_knees)
    e_low      = max(args.coarse_start, round(global_knee - 1.6, 1))
    e_body_end = round(global_knee - 0.2, 2)
    e_fine_end = round(global_knee + 0.2, 2)

    body_pts    = list(np.arange(e_low,          e_body_end + 1e-9, 0.2).round(2))
    fine_pts    = list(np.arange(e_body_end + 0.05, e_fine_end + 1e-9, 0.05).round(2))
    recommended = sorted({float(x) for x in body_pts + fine_pts})

    print(f"\n  {fmt_list(recommended)}")
    print(f"\n  Convention: 0.2 dB body  →  0.05 dB last ~0.6 dB")
    print(f"  Run full simulation with max_err=200 targeting BER ~1e-8\n")

if __name__ == '__main__':
    main()
