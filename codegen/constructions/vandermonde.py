"""
Vandermonde-form QC-LDPC parity-check matrix construction.

Block (i, j) is a circulant with shift  cols[j] * i  mod  block_size.

    Row 0 : all identity circulants  (shift = 0)
    Row 1 : shifts = cols[j]
    Row i : shifts = cols[j] * i  mod  block_size

Column generators cols[j] are drawn from Z_block_size* (non-zero).
Use girth_col_search.find_girth_free_cols to pick cols that guarantee
a minimum girth before calling build_vandermonde_matrix.
"""
import random
from typing import List, Optional

import numpy as np

from codegen.qc.ruler import CircularRuler
from codegen.qc.grid import writeNDarray, write_board
from fileio.matrix_io import writer


def build_vandermonde_grid(
    cols: List[int], J: int, block_size: int
) -> List[List[CircularRuler]]:
    """
    Build the J × L ruler grid for the Vandermonde construction.

    grid[i][j] = CircularRuler with single marker  cols[j] * i  mod  block_size.
    Row 0 has shift 0 (identity circulant) for every column.
    """
    return [
        [CircularRuler(block_size, ((cols[j] * i) % block_size,)) for j in range(len(cols))]
        for i in range(J)
    ]


def build_vandermonde_matrix(
    J: int,
    L: int,
    block_size: int,
    cols: Optional[List[int]] = None,
    rand_seed: int = 42,
    filename: str = '',
    truncate: int = 0,
) -> np.ndarray:
    """
    Build a Vandermonde-form QC-LDPC parity-check matrix.

    Parameters
    ----------
    J          : number of row-blocks
    L          : number of column-blocks
    block_size : circulant period P
    cols       : L column generators from Z_P*; randomly sampled if None
    rand_seed  : seed for random column selection
    filename   : if non-empty, write matrix to this file (sparse, compressed)
    truncate   : columns trimmed per block at write time

    Returns
    -------
    H : numpy array of shape (J*block_size, L*block_size)
    """
    if cols is None:
        rng = random.Random(rand_seed)
        cols = rng.sample(range(1, block_size), L)
    assert len(cols) == L, f"Need exactly L={L} column generators, got {len(cols)}"
    assert all(0 < c < block_size for c in cols), "All cols must be in 1..block_size-1"

    grid = build_vandermonde_grid(cols, J, block_size)

    if filename:
        write_board(grid, block_size, filename=filename,
                    truncate=truncate, mode='sparse', compress=True)

    return writeNDarray(grid, block_size, truncate=truncate).astype(np.uint8)


if __name__ == '__main__':
    J, L, P = 3, 19, 67
    H = build_vandermonde_matrix(J, L, P)
    print(f'H shape: {H.shape}  rate: {1 - J/L:.3f}  density: {H.mean():.4f}')
