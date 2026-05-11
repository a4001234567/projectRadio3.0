# Example 3 — BR Form vs. Punctured Vandermonde (Equivalence + BP Applicability)

Shows that the BR companion-matrix construction and the punctured Vandermonde
construction define the **same code** via an explicit algebraic transformation,
then contrasts their Tanner-graph structures to motivate why only the Vandermonde
representation is suitable for BP decoding.

## Parameters

| Parameter | Value |
|---|---|
| J (row-blocks) | 5 |
| L (col-blocks) | 37 |
| P / block_size | 37 (prime) |
| truncate (Vandermonde) | 1 |
| Block size N = P − 1 | 36 |
| Matrix shape (m × n) | 180 × 1332 |
| Linear code | [1332, 1152, ?] |
| Design rate | 32/37 ≈ 0.8649 |

## Variants

### 1. Vandermonde (punctured)
Standard punctured Vandermonde with consecutive column generators [0, 1, …, 36]
and truncate=1. Each block (i, j) is the top-left 36×36 sub-matrix of the
lower circulant shift raised to the power `i·j mod 37`. Column degrees: 4–5.

### 2. BR form
Companion-matrix construction: block (i, j) = S^{i·j mod 37}, where S is the
36×36 lower companion matrix (subdiagonal = 1, last column = 1 over F₂).
No truncation needed — S_{N−1} is already equivalent to the punctured P_N.
Column degrees: 5–40 (one column per S^k block is all-ones, causing spikes).

### 3. PEG
Progressive Edge Growth using the same degree sequence as the punctured
Vandermonde. Serves as a reference for a graph-coded design with matched
degree distribution but no algebraic structure.

## Algebraic Equivalence

The transformation matrix **T** (J·N × J·N block matrix) satisfies **T · H_BR = H_Vand** over F₂:

- T[0, 0] = I
- T[i, 0] = J (all-ones) for i > 0
- T[i, i] = −Q = Q (over F₂) for i > 0, where Q = J − I
- All other blocks = 0

The key identity underlying this is **Q · S^k = J − P^k_punct** for all k = 1…P−1.
T is non-singular because rank(Q) = N = 36 over F₂ (N even).

See `verify.py` for a full computational proof (three independent checks).

## Cycle Count Comparison

| | Vandermonde (punctured) | BR | PEG |
|---|---|---|---|
| **col-deg range** | 4–5 | 5–40 | 4–5 |
| **GF(2) rank** | 180 | 180 | 180 |
| **k (dimension)** | 1152 | 1152 | 1152 |
| **6-cycles** | 431,064 | 1,438,899,480 | 467,037 |
| **8-cycles** | — (timeout) | — (timeout) | — (timeout) |

Vandermonde and BR produce the **same code** (identical GF(2) rank and dimension)
but entirely different Tanner graphs. The BR matrix has column degrees up to 40
(each S^k block contains one all-ones column), making it unsuitable for
belief-propagation decoding. The punctured Vandermonde retains the sparse,
low-degree structure required for practical BP decoders.
