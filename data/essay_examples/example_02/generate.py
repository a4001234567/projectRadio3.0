"""
Example 2 — Vandermonde (Column Selection, 6-Cycle Free), all punctured
Parameters: J=3, L=13, block_size=67, truncate=1  →  198 × 858
  Punctured : consecutive cols [0..12]
  6-free    : girth-free cols (girth=6, no 4-cycles), punctured
  PEG       : same degree sequence as punctured 6-free, edges placed by PEG
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from codegen.constructions.vandermonde import build_vandermonde_grid
from codegen.constructions.girth_col_search import find_girth_free_cols
from codegen.qc.grid import writeNDarray
from codegen.peg.peg import peg as PEG
from fileio.matrix_io import writer

J          = 3
L          = 13
block_size = 67
truncate   = 1
out        = os.path.dirname(__file__)

# ── 1. Punctured (consecutive cols) ──────────────────────────────────────
cols_consec = list(range(L))   # [0, 1, ..., 12]
grid_consec = build_vandermonde_grid(cols_consec, J, block_size)
H_punct = writeNDarray(grid_consec, block_size, truncate=truncate).astype(np.uint8)
print(f'Punctured   : {H_punct.shape}  '
      f'col-deg={np.unique(H_punct.sum(0))}  row-deg={np.unique(H_punct.sum(1))}')
writer(os.path.join(out, 'vandermonde_punctured.zip'), H_punct,
       mode='sparse', compress=True,
       comments=[f'Vandermonde J={J} L={L} P={block_size} truncate={truncate}'])

# ── 2. 6-cycle-free column selection, punctured ───────────────────────────
# In a QC-LDPC circulant, two check nodes in the same block-row connect to
# distinct positions in every block-column (one 1 per row in a circulant),
# so they can never share a variable-node neighbor. Consequently any 6-cycle
# must involve check nodes from 3 *distinct* block-rows. With J=3 that means
# all block-rows are used exactly once, and girth=6 (g=3) catches every such
# case exactly — the check is complete for 6-cycle elimination at J=3.
print(f'Searching for {L} girth-6-free columns in Z_{block_size} ...')
cols_free = find_girth_free_cols(num_rows=J, P=block_size, max_cols=L, girth=6)
print(f'Found {len(cols_free)} columns: {sorted(cols_free)}')
grid_free = build_vandermonde_grid(cols_free, J, block_size)
H_free = writeNDarray(grid_free, block_size, truncate=truncate).astype(np.uint8)
print(f'6-free      : {H_free.shape}  '
      f'col-deg={np.unique(H_free.sum(0))}  row-deg={np.unique(H_free.sum(1))}')
writer(os.path.join(out, 'vandermonde_6free.zip'), H_free,
       mode='sparse', compress=True,
       comments=[f'Vandermonde J={J} L={L} P={block_size} truncate={truncate} '
                 f'girth-6-free cols={sorted(cols_free)}'])

# ── 3. PEG (matched to punctured 6-free degree sequence) ─────────────────
nchk, nvar = H_free.shape
degree_seq  = list(H_free.sum(axis=0).astype(int))
print(f'Running PEG : {nvar} variable nodes, {nchk} check nodes, '
      f'col degrees = {sorted(set(degree_seq))} ...')
sim_peg = PEG(nvar, nchk, degree_seq)
sim_peg.progressive_edge_growth()
H_peg = sim_peg.H.astype(np.uint8)
print(f'PEG         : {H_peg.shape}  '
      f'col-deg={np.unique(H_peg.sum(0))}  row-deg={np.unique(H_peg.sum(1))}')
writer(os.path.join(out, 'vandermonde_6free_peg.zip'), H_peg,
       mode='sparse', compress=True,
       comments=[f'PEG J={J} L={L} P={block_size} degree-matched to punctured 6-free Vandermonde'])

print('Done.')
