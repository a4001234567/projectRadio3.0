"""
Example 3 — BR Code vs. Truncated Vandermonde (Equivalence + BP Applicability)
Parameters: J=5, L=37, block_size=37

  Vandermonde : truncate=1  →  180 × 1332  lower companion form
  BR          : S_{36} blocks, no truncate  →  180 × 1332  (same code)
  PEG         : degree-matched to punctured Vandermonde
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from codegen.constructions.vandermonde import build_vandermonde_grid
from codegen.constructions.br import build_br_matrix
from codegen.qc.grid import writeNDarray
from codegen.peg.peg import peg as PEG
from fileio.matrix_io import writer

J          = 5
L          = 37
block_size = 37
cols       = list(range(L))   # [0, 1, ..., 36]  (all elements of Z_37)
out        = os.path.dirname(__file__)

# ── 1. Vandermonde punctured ──────────────────────────────────────────────────
grid   = build_vandermonde_grid(cols, J, block_size)
H_vand = writeNDarray(grid, block_size, truncate=1).astype(np.uint8)
print(f'Vandermonde : {H_vand.shape}  '
      f'col-deg={np.unique(H_vand.sum(0))}  row-deg={np.unique(H_vand.sum(1))}')
writer(os.path.join(out, 'vandermonde_punctured.zip'), H_vand,
       mode='sparse', compress=True,
       comments=[f'Vandermonde J={J} L={L} P={block_size} truncate=1'])

# ── 2. BR form ────────────────────────────────────────────────────────────────
H_br = build_br_matrix(J=J, cols=cols, P=block_size)
print(f'BR          : {H_br.shape}  '
      f'col-deg={np.unique(H_br.sum(0))}  row-deg={np.unique(H_br.sum(1))}')
writer(os.path.join(out, 'br.zip'), H_br,
       mode='sparse', compress=True,
       comments=[f'BR J={J} L={L} P={block_size} S_{{P-1}} blocks'])

# ── 3. PEG (degree-matched to punctured Vandermonde) ─────────────────────────
nchk, nvar = H_vand.shape
degree_seq = list(H_vand.sum(axis=0).astype(int))
print(f'Running PEG : {nvar} variable nodes, {nchk} check nodes ...')
sim_peg = PEG(nvar, nchk, degree_seq)
sim_peg.progressive_edge_growth()
H_peg = sim_peg.H.astype(np.uint8)
print(f'PEG         : {H_peg.shape}  '
      f'col-deg={np.unique(H_peg.sum(0))}  row-deg={np.unique(H_peg.sum(1))}')
writer(os.path.join(out, 'vandermonde_peg.zip'), H_peg,
       mode='sparse', compress=True,
       comments=[f'PEG J={J} L={L} P={block_size} degree-matched to punctured Vandermonde'])

print('Done.')
