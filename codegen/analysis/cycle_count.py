import numpy as np
from fileio.matrix_io import get_reader
import argparse

def count_n_cycles(check_matrix, n, short=False):
    """
    Count n-cycles in an LDPC check matrix.

    Parameters:
        check_matrix (np.ndarray): Binary 0/1 check matrix.
        n (int): Length of the cycle to detect.

    Returns:
        int: Number of n-cycles detected.
    """
    def dfs(node, depth, path):
        # If the cycle is closed and of the correct length
        if depth == n-1:
            if path[0] in graph[node]:
                return 1
            return 0

        count = 0
        for neighbor in graph[node]:
            if neighbor in visited:
                continue
            if neighbor not in path:
                count += dfs(neighbor, depth+1, path + [neighbor])
        return count

    # Step 1: Build a bipartite graph from the check matrix
    num_rows, num_cols = check_matrix.shape
    graph = {i: [] for i in range(num_rows)}  # Row nodes as integers
    graph.update({num_rows + j: [] for j in range(num_cols)})  # Column nodes as integers

    for i in range(num_rows):
        for j in range(num_cols):
            if check_matrix[i, j]:
                graph[i].append(num_rows + j)
                graph[num_rows + j].append(i)
    #print(graph[0])

    # Step 2: Perform DFS to count cycles starting from check nodes only
    visited = set()
    cycle_count = 0

    for node in range(num_rows):  # Only start from check nodes (row nodes)
        cycle_count += dfs(node, 0, [node])
        visited.add(node)  # Mark the node fully processed
        if short and cycle_count:
            return True

    # Each cycle is counted multiple times (once per starting node)
    return cycle_count//2

if __name__ == "__main__":
    check_matrix = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 1, 1, 0],
    ])

    parser = argparse.ArgumentParser(description="Count cycles of a file")
    parser.add_argument('filename', type=str, help="input filename")

    args = parser.parse_args()
    reader = get_reader()
    H = reader(args.filename)
    #print(H.shape)
    l = 6
    while True:
        print(l)
        print(count_n_cycles(H,l,short=False))
        a = input()
        if 'q' in a:
            exit()
        l += 2
    l = 4
    while True:
        if count_n_cycles(H,l,short=True):
            print(f'Minimum cycle lengh is {l}')
            exit(0)
        l += 2

