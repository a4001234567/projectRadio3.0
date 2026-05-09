"""
Moore-type QC-LDPC parity-check matrix construction.

Block (i, j) in the J×L grid has circulant shift set:
    rulers[i][j] = row_zero[j] << i   (all markers * 2^i mod block_size)

This gives a quasi-cyclic code with (block_size - truncate) × (block_size - truncate)
circulant sub-blocks.

For the diagonal (single-ruler) approach:
    row_zero[j] = base_ruler << j
    => rulers[i][j] = base_ruler << (i + j)
which is valid whenever the base ruler is "legal" for J+L-1 doublings.

Parameters
----------
J          : number of row-blocks (check-node side)
L          : number of column-blocks before truncation (variable-node side)
block_size : circulant period  (typically a Mersenne prime, 2^m-1, or similar odd number)
weight     : markers per block — 1 (single shift) or 2 (Golomb, girth >= 6)
truncate   : columns removed from the right of each block row at write time (default 1)
recipe     : optional masking — {col_type: count} where col_type = (w_0,...,w_{J-1}),
             w_k in {0, 1, ..., weight}.  Columns are consumed left-to-right.
"""

from itertools import combinations
from random import shuffle, sample, seed as _seed
from typing import List, Optional, Dict, Tuple
import numpy as np

from codegen.qc.ruler import CircularRuler, two_side_mod, find_viable_combinations
from codegen.qc.grid import write_board, writeNDarray
from fileio.matrix_io import writer as io_writer
from codegen.analysis.mds import check_mds


def is_valid_block_size(p: int) -> bool:
    """Return True iff p is an odd prime with 2 as a primitive root."""
    if p < 3 or p % 2 == 0:
        return False
    # Miller-Rabin primality (deterministic for p < 3.3e24 with these witnesses)
    d, r = p - 1, 0
    while d % 2 == 0:
        d //= 2; r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a >= p:
            continue
        x = pow(a, d, p)
        if x in (1, p - 1):
            continue
        for _ in range(r - 1):
            x = x * x % p
            if x == p - 1:
                break
        else:
            return False
    # Check ord_p(2) == p-1 by verifying 2^((p-1)/q) != 1 for every prime factor q of p-1
    phi = p - 1
    n = phi
    factors = set()
    f = 2
    while f * f <= n:
        while n % f == 0:
            factors.add(f); n //= f
        f += 1
    if n > 1:
        factors.add(n)
    return all(pow(2, phi // q, p) != 1 for q in factors)


# ── Ruler legality search ─────────────────────────────────────────────────────

def generate_legal_rulers(block_size: int, J: int, weight: int) -> List[CircularRuler]:
    """
    Return all CircularRulers of `weight` markers over Z_{block_size} that are
    legal for the Moore construction with J row-blocks.

    Legality means:
      - weight >= 2: the ruler is Golomb (all pairwise circular distances distinct)
      - The Golomb distance sets of ruler, ruler<<1, ruler<<2, ... ruler<<(J-1)
        are mutually disjoint (ensures girth >= 6 across row doublings).

    For weight == 1 all single-marker rulers are legal (no pair → no bins → no
    conflict); they are returned in order [0, 1, ..., block_size-1].
    """
    legal = []
    for markers in combinations(range(block_size), weight):
        r = CircularRuler(block_size, markers)
        if weight >= 2 and not r.check_golomb():
            continue
        if weight >= 2:
            seen_bins = set(r.bins)
            valid = True
            for i in range(1, J):
                new_bins = {two_side_mod((b << i) % block_size, block_size) for b in r.bins}
                if not new_bins.isdisjoint(seen_bins):
                    valid = False
                    break
                seen_bins.update(new_bins)
            if not valid:
                continue
        legal.append(r)
    return legal


# ── Row-zero construction ─────────────────────────────────────────────────────

def find_moore_row_zero(
    block_size: int, J: int, L: int, weight: int, rand_seed: int = 42
) -> List[CircularRuler]:
    """
    Find L mutually compatible rulers for row-zero via find_viable_combinations.

    Each ruler is legal for J doublings (generate_legal_rulers), and the set of
    L rulers is jointly compatible: no bin collisions between any pair across all
    J row-doublings (find_viable_combinations).

    Returns a list of L CircularRulers.
    """
    _seed(rand_seed)
    legals = generate_legal_rulers(block_size, J, weight)
    if not legals:
        raise ValueError(
            f"No legal weight-{weight} ruler for block_size={block_size}, J={J}. "
            f"Try a larger block_size."
        )
    shuffle(legals)
    row_zero = find_viable_combinations(
        [set() for _ in range(block_size - 1)], legals, L, J
    )
    if not row_zero:
        raise ValueError(
            f"Could not find {L} compatible rulers for block_size={block_size}, "
            f"J={J}, weight={weight}. Try a larger block_size or fewer columns."
        )
    return row_zero


# ── Grid construction ─────────────────────────────────────────────────────────

def build_moore_grid(
    row_zero: List[CircularRuler], J: int
) -> List[List[CircularRuler]]:
    """
    Build the J × L grid: rulers[i][j] = row_zero[j] << i.
    """
    return [[row_zero[j] << i for j in range(len(row_zero))] for i in range(J)]


# ── Post-construction column shift (identity normalisation) ───────────────────

def shift_to_identity(
    grid: List[List[CircularRuler]], block_size: int
) -> List[List[CircularRuler]]:
    """
    Apply a per-column offset so that the row-0 block of every column becomes
    an identity-like circulant (its first/minimum marker shifts to 0).

    Each column j gets its own offset_j = (-min(rulers[0][j].markers)) % block_size,
    applied to all J row-blocks in that column independently.

    Adding a constant mod block_size is a cyclic-group automorphism —
    it preserves girth and all CR / Golomb properties.
    """
    J = len(grid)
    L = len(grid[0])
    offsets = [(-min(grid[0][j].markers)) % block_size for j in range(L)]
    return [
        [
            CircularRuler(block_size, {(m + offsets[j]) % block_size for m in grid[i][j].markers})
            for j in range(L)
        ]
        for i in range(J)
    ]


# ── Masking (recipe application) ──────────────────────────────────────────────

def apply_recipe(
    grid: List[List[CircularRuler]],
    recipe: Dict[Tuple[int, ...], int],
    block_size: int,
) -> List[List[CircularRuler]]:
    """
    Apply a masking recipe to the ruler grid, consuming columns left-to-right.

    recipe: {col_type: count}
        col_type = (w_0, w_1, ..., w_{J-1})  — desired weight per row-block
        count    = number of columns of this type

    Each w_k must be <= len(grid[k][col].markers).  The markers are randomly
    sub-sampled to the requested weight.  A weight of 0 gives a zero block.
    """
    J = len(grid)
    col = 0
    masked = [list(row) for row in grid]  # shallow copy per row
    for col_type, cnt in recipe.items():
        for _ in range(cnt):
            for row_idx, w in enumerate(col_type):
                src = masked[row_idx][col]
                if w == 0:
                    masked[row_idx][col] = CircularRuler(block_size)
                elif w < len(src):
                    masked[row_idx][col] = CircularRuler(
                        block_size, sample(sorted(src.markers), w)
                    )
                # w == len(src): leave as-is
            col += 1
    return masked


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_moore_matrix(
    J: int,
    L: int,
    block_size: int,
    *,
    weight: int = 2,
    truncate: int = 1,
    recipe: Optional[Dict[Tuple[int, ...], int]] = None,
    identity_col: bool = False,
    verify_mds: bool = False,
    rand_seed: int = 42,
    filename: str = "",
    return_array: bool = True,
) -> Optional[np.ndarray]:
    """
    Build (and optionally save) a Moore-type QC-LDPC parity-check matrix.

    Parameters
    ----------
    J, L         : row- and column-block counts
    block_size   : circulant period
    weight       : markers per block (1 or 2)
    truncate     : columns trimmed per block row at write time
    recipe       : masking dict {col_type: count}; None = no masking (all full-weight)
    identity_col : if True, apply a per-column offset after construction so that
                   rulers[0][j] has its minimum marker at 0 for every column j.
                   Each column gets its own independent offset — girth is preserved.
    verify_mds   : if True, check the block MDS property of the marker matrix over
                   GF(2^{block_size-1}).  Skipped by default as it is time-consuming.
    rand_seed    : random seed for ruler search
    filename     : if non-empty, write the matrix to this file via UTILS.IO
    return_array : if True, also return the dense numpy array

    Returns
    -------
    H  : numpy uint8 array of shape (J*(block_size-truncate), L*(block_size-truncate))
         or None if return_array is False
    """
    if not is_valid_block_size(block_size):
        raise ValueError(f"block_size={block_size} must be an odd prime with 2 as a primitive root.")

    # 1. Row-zero
    row_zero = find_moore_row_zero(block_size, J, L, weight, rand_seed)

    # 2. Full J×L grid
    grid = build_moore_grid(row_zero, J)

    # 3. Per-column shift so row-0 blocks become identity-like
    if identity_col:
        grid = shift_to_identity(grid, block_size)

    # 4. Masking
    if recipe is not None:
        grid = apply_recipe(grid, recipe, block_size)

    # 5. Optional block MDS check
    if verify_mds:
        f = tuple(range(block_size))   # (x^p-1)/(x-1) = 1+x+...+x^{p-1}, irreducible over F_2
        markers = [[tuple(sorted(rulers.markers)) for rulers in row] for row in grid]
        if not check_mds(f, markers):
            raise ValueError("Block MDS property not satisfied.")

    # 6. Write to file
    if filename:
        write_board(grid, block_size, filename=filename,
                    truncate=truncate, mode='sparse', compress=True)

    # 6. Dense array
    if return_array:
        H = writeNDarray(grid, block_size, truncate=truncate)
        return H.astype(np.uint8)
    return None


# ── Density-evolution recipe search ──────────────────────────────────────────

def search_recipe(
    J: int,
    L: int,
    block_size: int,
    truncate: int = 1,
    weight: int = 2,
    channel: str = 'AWGN',
    menus: Optional[List[Tuple[int, ...]]] = None,
) -> Tuple[Dict, float]:
    """
    Search for the masking recipe with the lowest decoding threshold via
    density evolution, subject to the rectangular condition.

    Parameters
    ----------
    J, L        : row- and column-block counts
    block_size  : circulant period
    truncate    : truncation (same as used in build_moore_matrix)
    weight      : max weight per block (1 or 2); defines the default menu
    channel     : 'BEC' or 'AWGN'
    menus       : explicit list of column types to consider; if None, uses all
                  tuples in {0,...,weight}^J that are not all-zero

    Returns
    -------
    best_recipe : dict {col_type: count}
    threshold   : corresponding decoding threshold
    """
    from de.bec import findThresForRecipe as BEC_findThresForRecipe
    from de.recipe import isRecipeSatisfyRect, find_composition

    if channel == 'AWGN':
        from de.awgn import findThresForRecipe as GA_findThresForRecipe
        thres_fn = GA_findThresForRecipe
    else:
        thres_fn = BEC_findThresForRecipe

    if menus is None:
        from itertools import product as iproduct
        # Each block is either full weight or zero — no mixed 1&2 per column
        menus = [
            col_type for col_type in iproduct((0, weight), repeat=J)
            if any(w > 0 for w in col_type)
        ]

    best_recipe = None
    best_thres  = float('inf')

    for composition in find_composition(L, len(menus)):
        recipe = {k: v for k, v in zip(menus, composition)}
        if not isRecipeSatisfyRect(recipe):
            continue
        thres = thres_fn(recipe, block_size, truncate)
        if thres < best_thres:
            best_thres  = thres
            best_recipe = {k: v for k, v in recipe.items() if v > 0}

    if best_recipe is None:
        raise ValueError("No recipe satisfying the rectangular condition was found.")

    return best_recipe, best_thres


# ── CLI / demo ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    J = 3; L = 12; block_size = 107; truncate = 1; weight = 2
    recipe = None  # set e.g. {(2, 2, 0): 2, (2, 0, 2): 2, (0, 2, 2): 2, (2,2,2): 6}

    print(f"Moore QC-LDPC  J={J}  L={L}  block={block_size}  w={weight}  trunc={truncate}")
    t0 = time.time()
    H = build_moore_matrix(
        J, L, block_size,
        weight=weight,
        truncate=truncate,
        recipe=recipe,
        filename=f'moore_{J}x{L}_b{block_size}_w{weight}.zip',
    )
    print(f"Shape: {H.shape}  density: {H.sum()/(H.shape[0]*H.shape[1]):.4f}  [{time.time()-t0:.2f}s]")
