"""
Apply PEG (Progressive Edge Growth) edge reassignment to an existing matrix.

Usage
-----
    python apply_peg.py <input.zip> <output.zip>

Example (reproduce the project result):
    python apply_peg.py ../task3_moore/output/moore_J2_L10_b101_opt.zip \
                        output/moore_J2_L10_b101_opt_peg.zip

PEG preserves the degree distribution of the input matrix while greedily
maximising local girth.  The output matrix has the same shape and column
weights as the input; its GF(2) rank (and hence code rate) is unchanged.

Note: PEG is only meaningful for full-rank matrices (e.g. Moore matrices).
BJ matrices have many GF(2)-redundant rows, so PEG is not applicable there.
"""

import sys, os

from fileio.matrix_io import get_reader, writer
from codegen.peg.peg import PEG_LIKE

os.makedirs(os.path.join(os.path.dirname(__file__), 'output'), exist_ok=True)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    filein, fileout = sys.argv[1], sys.argv[2]
    print(f"Reading  : {filein}")
    H = get_reader()(filein)
    print(f"H shape  : {H.shape}")

    print("Running PEG edge reassignment ...")
    H_peg = PEG_LIKE(H)

    writer(fileout, H_peg, mode='sparse', compress=True,
           comments=[f'PEG-processed  source={os.path.basename(filein)}'])
    print(f"Saved    : {fileout}")
