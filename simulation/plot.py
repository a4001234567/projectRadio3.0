"""
AWGN performance plotter — BER and BLER vs Eb/N0.

Usage
-----
    python plot_performance.py

Reads the last simulation run from each *-AWGN.txt file in matrices/.
Produces results.pdf.
"""

import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
]

_pseudo_tick_values = np.array(
    [v * 10**k for k in range(-10, 1) for v in range(2, 10)]
)


def parse_last_run(path):
    """Return (ebno, ber, bler) from the last simulation run in an AWGN result file."""
    text = open(path).read()
    blocks = re.split(r'(?=  ebno  :)', text.strip())
    blocks = [b for b in blocks if b.strip()]
    last = blocks[-1]
    lines = {l.split(':')[0].strip(): l.split(':', 1)[1].strip()
             for l in last.strip().splitlines() if ':' in l}
    ebno = [float(x) for x in lines['ebno'].split()]
    bler = [float(x) for x in lines['ave_bler'].split()]
    ber  = [float(x) for x in lines['ave_ber'].split()]
    return ebno, ber, bler


def plot(ebno, ber, bler, marker, label, color):
    plt.plot(ebno, ber,  marker=marker, linestyle='-',  color=color,
             label=label + ' BER',  markersize=4)
    plt.plot(ebno, bler, marker=marker, linestyle='--', color=color,
             label=label + ' BLER', markersize=4)


def main():
    f, ax = plt.subplots(1, figsize=(9, 5.5))

    # ── 5 reliable simulation results ────────────────────────────────────────
    curves = [
        ('matrices/moore_J2_L10_b101_opt.zip-AWGN.txt',
         'o', 'Moore opt (N=1000, R=0.800, thr=2.55 dB)', colors[0]),
        ('matrices/moore_J2_L10_b101_opt_peg.zip-AWGN.txt',
         's', 'Moore opt+PEG (N=1000, R=0.800)',           colors[1]),
        ('matrices/moore_J2_L10_b101_raw.zip-AWGN.txt',
         '^', 'Moore raw (N=1000, R=0.800, thr=2.75 dB)',  colors[2]),
        ('matrices/bj_m5_g12_r33_u0.zip-AWGN.txt',
         'D', 'BJ g=12 (N=1023, R=0.803, thr=2.66 dB)',   colors[3]),
        ('matrices/vandermonde_J3_L15_p67_wt1.zip-AWGN.txt',
         'x', 'Vand. wt1 J=3 (N=1005, R=0.802, thr=2.67 dB)', colors[4]),
    ]

    for path, marker, label, color in curves:
        ebno, ber, bler = parse_last_run(path)
        plot(ebno, ber, bler, marker, label, color)

    plt.yscale('log')
    plt.ylabel('BER / BLER')
    plt.xlabel('$E_b/N_0$ (dB)')
    ax.set_xlim(2.7, 4.95)
    ax.set_ylim(1e-8, 1)
    plt.grid(axis='y', which='both', linestyle=':')
    plt.grid(axis='x', which='major', linestyle=':')
    plt.legend(fontsize='x-small', loc='lower left', ncol=1)
    plt.xticks([2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8])
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/results.pdf', transparent=True)
    plt.savefig('results/results.png', dpi=150)
    print('Saved results/results.pdf  and  results/results.png')


if __name__ == '__main__':
    main()
