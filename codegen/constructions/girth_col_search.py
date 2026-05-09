"""
Girth-free column selection for QC-LDPC Vandermonde-form matrices.

For a J-row QC matrix with circulant period P, a set of column shifts
{c_0, ..., c_{L-1}} is girth-2g-free iff no g-tuple of rows (i_0,...,i_{g-1})
and g-tuple of columns (c_{j_0},...,c_{j_{g-1}}) satisfies:

    sum_k  (i_k - i_{k-1}) * c_{j_k}  ≡ 0  (mod P)

`find_girth_free_cols` greedily builds the largest such column set via
backtracking, or finds exactly `max_cols` columns if specified.
"""
from itertools import combinations
from typing import List, Optional
import random

from utils.combinatorics import combination_with_order


def has_conflict(cols: List[int], num_rows: int, girth: int, P: int) -> Optional[tuple]:
    """
    Return a conflicting column tuple if cols contains a girth-cycle, else None.

    A conflict is a g-tuple of columns (g = girth//2) for which some g-tuple
    of row-index differences satisfies the inner-product condition mod P.
    """
    g = girth >> 1
    rows = tuple(range(num_rows))
    for row_seq in combination_with_order(rows, g):
        row_diffs = [row_seq[i] - row_seq[i - 1] for i in range(g)]
        for col_tuple in combinations(cols, g):
            if not sum(d * c for d, c in zip(row_diffs, col_tuple)) % P:
                return col_tuple
    return None


def find_girth_free_cols(
    num_rows: int,
    P: int,
    max_cols: int,
    girth: int = 6,
    choices: Optional[List[int]] = None,
    rand_seed: int = 42,
) -> List[int]:
    """
    Greedily find up to `max_cols` column shifts from Z_P that are girth-free.

    Parameters
    ----------
    num_rows : J — number of row-blocks in the QC matrix
    P        : circulant period (column shifts are drawn from 0..P-1)
    max_cols : target number of columns to select
    girth    : minimum girth to enforce (must be even; default 6)
    choices  : candidate column shifts to search over (default: all 0..P-1)
    rand_seed: random seed for shuffling candidates

    Returns
    -------
    List of selected column shifts (length <= max_cols).
    Call `len(result) < max_cols` to detect partial solutions.
    """
    if choices is None:
        choices = list(range(P))

    rng = random.Random(rand_seed)
    rng.shuffle(choices)

    best: List[int] = []

    def backtrack(already: List[int], remaining: List[int]) -> Optional[List[int]]:
        nonlocal best
        if len(already) == max_cols:
            return already
        if len(already) > len(best):
            best = already.copy()
        for i, col in enumerate(remaining):
            candidate = already + [col]
            if has_conflict(candidate, num_rows, girth, P) is None:
                result = backtrack(candidate, remaining[i + 1:])
                if result is not None:
                    return result
        return None

    result = backtrack([], choices)
    return result if result is not None else best


if __name__ == '__main__':
    num_rows = 3
    girth    = 6
    P        = 67
    max_cols = 25

    try:
        cols = find_girth_free_cols(num_rows, P, max_cols, girth=girth)
        print(f'Found {len(cols)} columns: {cols}')
    except KeyboardInterrupt:
        print('Interrupted')
