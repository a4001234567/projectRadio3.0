# Example 2 — Vandermonde (Column Selection, 6-Cycle Free), Punctured

Compares three punctured Vandermonde constructions to show the effect of
algebraic column selection on short-cycle counts. All variants share the same
[858, 660, ?] parameters, isolating the impact of construction method on
cycle distribution.

## Parameters

| Parameter | Value |
|---|---|
| J (row-blocks) | 3 |
| L (col-blocks) | 13 |
| P / block_size | 67 |
| truncate | 1 |
| tsize = P − truncate | 66 |
| Matrix shape (m × n) | 198 × 858 |
| Linear code | [858, 660, ?] |
| Code rate | 10/13 |

All three variants are punctured (truncate=1) and share the same [858, 660, ?]
parameters and rate 10/13. GF(2) rank = 198 = m (full rank) for all variants.

## Variants

### 1. Punctured
Standard Vandermonde with consecutive column generators [0, 1, …, 12].
No girth guarantee. Serves as the baseline.

### 2. 6-Cycle Free
Column generators chosen by backtracking girth-free search (`girth_col_search`,
`girth=6`).

Selected generators: `[3, 7, 15, 17, 25, 29, 34, 38, 45, 46, 49, 50, 57]`

### 3. PEG
Progressive Edge Growth using the same degree sequence as the 6-free variant
(mixed col-degrees 2–3 matching the punctured structure). Edges placed to
maximise local girth but without the algebraic column-selection guarantee.

## Cycle Count Comparison

| | Punctured | 6-free | PEG |
|---|---|---|---|
| **6-cycles** | 4,554 | **0** | 745 |
| **8-cycles** | 126,813 | 51,447 | 44,816 |
| **10-cycles** | 1,318,871 | 655,128 | 725,446 |

The 6-free column selection eliminates all 6-cycles and halves the 8/10-cycle
counts compared to the plain punctured baseline. PEG achieves the fewest
8-cycles but does not guarantee 6-cycle freedom.
