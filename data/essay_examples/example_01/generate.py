"""
Example 1 — Vandermonde (Theorem 7): punctured vs. unpunctured vs. PEG
Parameters: J=13, L=67, block_size=67
  Unpunctured : truncate=0  →  871 × 4489  (13,67)-regular
  Punctured   : truncate=3  →  832 × 4288  (13,64)-regular
  PEG         : same degree sequence as unpunctured, edges placed by PEG
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from codegen.constructions.vandermonde import build_vandermonde_grid
from codegen.qc.grid import writeNDarray
from codegen.peg.peg import peg as PEG
from fileio.matrix_io import writer

J          = 13
L          = 67
block_size = 67
cols       = list(range(L))   # [0, 1, ..., 66]  (all elements of Z_67)
out        = os.path.dirname(__file__)

grid = build_vandermonde_grid(cols, J, block_size)

# ── 1. Unpunctured ────────────────────────────────────────────────────────────
H_unpunct = writeNDarray(grid, block_size, truncate=0).astype(np.uint8)
print(f'Unpunctured : {H_unpunct.shape}  '
      f'col-deg={np.unique(H_unpunct.sum(0))}  row-deg={np.unique(H_unpunct.sum(1))}')
writer(os.path.join(out, 'vandermonde_unpunctured.zip'), H_unpunct,
       mode='sparse', compress=True,
       comments=[f'Vandermonde J={J} L={L} P={block_size} truncate=0'])

# ── 2. Punctured ──────────────────────────────────────────────────────────────
H_punct = writeNDarray(grid, block_size, truncate=1).astype(np.uint8)
print(f'Punctured   : {H_punct.shape}  '
      f'col-deg={np.unique(H_punct.sum(0))}  row-deg={np.unique(H_punct.sum(1))}')
writer(os.path.join(out, 'vandermonde_punctured.zip'), H_punct,
       mode='sparse', compress=True,
       comments=[f'Vandermonde J={J} L={L} P={block_size} truncate=1'])

# ── 3. PEG (matched to unpunctured degree sequence) ───────────────────────────
nchk, nvar = H_unpunct.shape
degree_seq  = list(H_unpunct.sum(axis=0).astype(int))   # [13, 13, ..., 13]
print(f'Running PEG : {nvar} variable nodes, {nchk} check nodes, '
      f'all column degrees = {degree_seq[0]} ...')
sim_peg = PEG(nvar, nchk, degree_seq)
sim_peg.progressive_edge_growth()
H_peg = sim_peg.H.astype(np.uint8)
print(f'PEG         : {H_peg.shape}  '
      f'col-deg={np.unique(H_peg.sum(0))}  row-deg={np.unique(H_peg.sum(1))}')
writer(os.path.join(out, 'vandermonde_peg.zip'), H_peg,
       mode='sparse', compress=True,
       comments=[f'PEG J={J} L={L} P={block_size} degree-matched to unpunctured Vandermonde'])

print('Done.')
