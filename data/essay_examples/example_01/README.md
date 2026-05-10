# Example 1 — Vandermonde (Theorem 7, Punctured vs. Unpunctured vs. PEG)

Demonstrates Theorem 7 using a Vandermonde-based parity-check matrix.
Compares three variants — punctured, unpunctured, and PEG-refined — to show
that BER/BLER performance is roughly equivalent across all three.

## Parameters

J=13, L=67, block_size=67, cols=[0..66]

| Variant | truncate (τ) | H shape | Regularity |
|---|---|---|---|
| Unpunctured | 0 | 871 × 4489 | (13,67)-regular |
| Punctured   | 1 | 858 × 4422 | col-deg 12–13, row-deg 66–67 |
| PEG         | — | 871 × 4489 | degree-matched to unpunctured |
