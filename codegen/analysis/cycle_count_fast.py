import numpy as np
from fileio.matrix_io import get_reader
import argparse

def count_n_cycles(check_matrix, n, short=False):
    """
    Count n-cycles in an LDPC check matrix.
    Optimized: set-based path membership + backtracking + np.nonzero graph build.
    """
    num_rows, num_cols = check_matrix.shape
    total = num_rows + num_cols

    # Build adjacency: lists for iteration, sets for O(1) membership/closure check
    adj_list = [[] for _ in range(total)]
    adj_set  = [set() for _ in range(total)]
    rs, cs = np.nonzero(check_matrix)
    for r, c in zip(rs.tolist(), cs.tolist()):
        v = num_rows + c
        adj_list[r].append(v)
        adj_list[v].append(r)
        adj_set[r].add(v)
        adj_set[v].add(r)

    visited     = set()   # fully-processed check nodes (avoids double-counting)
    in_path     = set()   # nodes currently on the DFS path
    cycle_count = [0]
    target      = [0]     # start node of current DFS

    def dfs(node, depth):
        if depth == n - 1:
            if target[0] in adj_set[node]:
                cycle_count[0] += 1
                if short:
                    raise StopIteration
            return
        in_path.add(node)
        for nb in adj_list[node]:
            if nb not in visited and nb not in in_path:
                dfs(nb, depth + 1)
        in_path.discard(node)

    for node in range(num_rows):
        target[0] = node
        in_path.add(node)
        try:
            for nb in adj_list[node]:
                dfs(nb, 1)
        except StopIteration:
            return True
        in_path.discard(node)
        visited.add(node)

    return cycle_count[0] // 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count cycles of a file (fast)")
    parser.add_argument('filename', type=str, help="input filename")
    parser.add_argument('-n', type=int, default=6, help="cycle length")
    args = parser.parse_args()
    reader = get_reader(output_form='ARRAY')
    H = reader(args.filename)
    print(f'Shape: {H.shape}')
    print(f'{args.n}-cycles: {count_n_cycles(H, args.n)}')
