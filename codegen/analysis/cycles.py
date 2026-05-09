#!/usr/bin/env python3
from typing import List, Tuple, Set, Dict

def countQCcycles(mat:List[List[Tuple[int,...]]], block_size: int, cycle_length: int) -> List[List[int]]:
    assert cycle_length & 1 == 0, "Cycle length must be even"
    result = [[0 for _ in row] for row in mat]
    mat_processed = [[((entry[0]+block_size)%block_size if entry else -1) for entry in row] for row in mat]
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if not len(mat[i][j]):
                continue
            result[i][j] = _DFS(mat_processed, block_size, (i,j), cycle_length, set(), (i,j), 0)
    return result

def _DFS(mat:List[List[int]], block_size: int, start_node:Tuple[int,int], ttl:int, curPath:Set[Tuple[int,int]], LastNode:Tuple[int,int], pastValues:int) -> int:
    if 0 == ttl:
        return 1 if (LastNode == start_node and pastValues%block_size == 0) else 0
    cnt = 0
    H = len(mat); W = len(mat[0])
    # Enumerate all possible next nodes;
    row, col = LastNode
    curNodeValue = mat[row][col]
    if len(curPath)&1: # odd length path, going vertical
        candidates = [((i, col), 0) for i in range(H) if mat[i][col] != -1 and i != row]
    else: # even length path, going horizontal
        candidates = [((row, j), mat[row][j] - curNodeValue) for j in range(W) if mat[row][j] != -1 and j != col]
    #print(candidates)
    for nextNode,diff in candidates:
        #if nextNode in curPath: continue
        cnt += _DFS(mat, block_size, start_node, ttl-1, curPath | {nextNode}, nextNode, pastValues + diff)
    return cnt

# Example usage
if __name__ == "__main__":
    mat = [[(10,),(20,)],[(10,),(20,)]]
    mat = [[(58,),(13,),(141,),(126,),(115,),(72,),(53,),(280,),(45,),(303,),(217,),(17,),(16,),(48,),(112,),(120,)],
    [(116,),(26,),(282,),(252,),(230,),(144,),(106,),(243,),(90,),(289,),(117,),(34,),(32,),(96,),(224,),(240,)],
    [(174,),(39,),(106,),(61,),(28,),(216,),(159,),(206,),(135,),(275,),(17,),(51,),(48,),(144,),(19,),(43,)],
    [(232,),(52,),(247,),(187,),(143,),(288,),(212,),(169,),(180,),(261,),(234,),(68,),(64,),(192,),(131,),(163,)]
    ]
    block_size = 317
    print(countQCcycles(mat, block_size, 6))