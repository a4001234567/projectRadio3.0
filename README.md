# projectRadio 3.0 — LDPC Code Generation & Simulation

A Python toolkit for constructing, analysing, and simulating LDPC codes — binary and non-binary, over AWGN and BEC channels.

## Structure

```
projectRadio3.0/
├── fileio/              # Matrix I/O (normal/sparse/diff, zlib, binary + NB fields)
├── utils/               # Combinatorics, GF(2) polynomial arithmetic
│
├── codegen/
│   ├── qc/              # QC-LDPC infrastructure (CircularRuler, grid assembly, CR checker)
│   ├── constructions/   # Code constructions
│   │   ├── moore.py         Moore QC-LDPC
│   │   ├── golomb.py        General Golomb matrix search
│   │   ├── bj.py            BJ code over GF(2^m) — binary and NB forms
│   │   ├── vandermonde.py   Vandermonde-form QC-LDPC
│   │   ├── girth_col_search.py  Girth-free column selection
│   │   ├── xiao.py          Xiao algebraic QC (p=20b+1 and p=12b+1)
│   │   ├── ieee80211n.py    IEEE 802.11n standard matrices
│   │   └── nb_ldpc.py       NB-QC-LDPC over GF(2^m)
│   ├── peg/             # Progressive Edge Growth
│   └── analysis/        # CR, MDS, GE rank, cycle counting, degree stats
│
├── de/                  # Density Evolution (BEC exact, AWGN Gaussian Approximation)
│   ├── phi.py           φ/φ⁻¹ lookup tables
│   ├── awgn.py          GA-DE threshold search
│   ├── bec.py           BEC-DE threshold search
│   ├── recipe.py        Degree distribution from masking recipe
│   └── search.py        Exhaustive recipe search with rectangular condition
│
├── simulation/
│   ├── channels/        AWGN (BPSK + QAM), BEC
│   ├── decoders/        BP, Scaled Min-Sum, Peeling (BEC), NB-EMS
│   ├── modulation/      BPSK modulation/demodulation
│   ├── runner.py        AWGN Monte Carlo runner (binary)
│   ├── runner_bec.py    BEC Monte Carlo runner
│   ├── runner_nb.py     NB-LDPC AWGN Monte Carlo runner
│   ├── nb_decoder.py    NB-EMS decoder library
│   └── plot.py          BER/BLER curve plotter
│
└── data/
    ├── matrices/        Generated H matrices (.zip)
    └── results/         Simulation output files
```

## Quick Start

```python
# Build a Moore QC-LDPC matrix optimised via Density Evolution
from codegen.constructions.moore import build_moore_matrix, search_recipe
from codegen.analysis.ge import analyse

recipe, threshold = search_recipe(J=2, L=8, block_size=37, channel='AWGN')
H = build_moore_matrix(J=2, L=8, block_size=37, recipe=recipe, truncate=1)
print(analyse(H))

# Simulate over AWGN with BP decoder
from simulation.decoders.bp import make_Flooding_BP_Decoder
from simulation.channels.awgn import ebno_to_sigma, bpsk_mod, bpsk_llr
import numpy as np

K, N = H.shape
R = 1 - K/N
indx, indy = np.nonzero(H)
decode = make_Flooding_BP_Decoder(max_iter=50, indx=indx, indy=indy, H=K, W=N)

sigma = ebno_to_sigma(4.0, R, bits_per_symbol=1)
y = bpsk_mod(np.zeros(N)) + sigma * np.random.randn(N)
x_hat, iters = decode(bpsk_llr(y, sigma))

# Build a Vandermonde matrix guaranteed girth >= 6
from codegen.constructions.girth_col_search import find_girth_free_cols
from codegen.constructions.vandermonde import build_vandermonde_matrix

cols = find_girth_free_cols(num_rows=3, P=67, max_cols=12, girth=6)
H = build_vandermonde_matrix(J=3, L=12, block_size=67, cols=cols)

# Non-binary BJ code over GF(8)
from codegen.constructions.bj import build_nb_bj_matrix
H_nb = build_nb_bj_matrix(m=3, gamma=4, rho=6)
```

## Requirements

```
pip install -r requirements.txt
```

## Constructions

| Construction | File | Notes |
|---|---|---|
| Moore QC-LDPC | `moore.py` | Golomb ruler doubling, DE-guided masking |
| General Golomb | `golomb.py` | Free per-cell ruler search |
| BJ code | `bj.py` | Over GF(2^m), binary and NB forms |
| Vandermonde | `vandermonde.py` | Simple shift-product structure |
| Girth-free col search | `girth_col_search.py` | Column selection for any target girth |
| Xiao algebraic | `xiao.py` | p=20b+1 and p=12b+1 variants |
| IEEE 802.11n | `ieee80211n.py` | Standard D2 matrices, all rates/lengths |
| NB-QC-LDPC | `nb_ldpc.py` | Vandermonde-like NB construction |

## Decoders

| Decoder | Channel | File |
|---|---|---|
| Flooding Belief Propagation | AWGN | `decoders/bp.py` |
| Scaled Min-Sum (Numba JIT) | AWGN | `decoders/minsum.py` |
| Peeling decoder (Python + JIT) | BEC | `decoders/peeling.py` |
| Extended Min-Sum (NB-EMS) | AWGN | `nb_decoder.py` |
