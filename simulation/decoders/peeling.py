from queue import Queue
import numpy as np
from numba import jit
from numba.typed import List
from numba import types
from typing import Tuple, Any, TypeVar
from numpy.typing import NDArray
import time

def peeling_decoder(mat_H:NDArray[np.int64], vec_y:NDArray[np.bool]) -> int:
    """"
    Peeling decoder for LDPC codes
    :param mat_H: parity check matrix
    :param vec_y: received vector status, 1 for unknown/erased bits, 0 for known bits
    :return: number of remaining unknown bits
    """
    # preprocessing
    # declaring the arrays:
    H,W = mat_H.shape
    num_of_unknown_bits = vec_y.sum()
    num_of_nodes = 0
    unknown_var_nodes = set(np.nonzero(vec_y)[0])
    cur_rows = [set() for _ in range(W)]
    unknown_var_nodes_per_chk = [set() for _ in range(H)]
    for chk in range(len(mat_H)):
        for var in mat_H[chk].nonzero()[0]:
            if var not in unknown_var_nodes:
                continue
            cur_rows[var].add(chk)
            unknown_var_nodes_per_chk[chk].add(var)

    peelable = Queue()
    for chk in range(H):
        if 1 == len(unknown_var_nodes_per_chk[chk]):
            peelable.put(chk)

    while not peelable.empty():
        cur_row = peelable.get()
        if not len(unknown_var_nodes_per_chk[cur_row]):
            continue
        peeled_var = unknown_var_nodes_per_chk[cur_row].pop()
        for chk in cur_rows[peeled_var]:
            unknown_var_nodes_per_chk[chk].discard(peeled_var)
            if 1 == len(unknown_var_nodes_per_chk[chk]):
                peelable.put(chk)
        unknown_var_nodes.remove(peeled_var)
    return len(unknown_var_nodes)

@jit(nopython=True,cache=True,nogil=True)
def jitpeeling_decoder(mat_H:NDArray[np.int64], vec_y:NDArray[np.bool]) -> Tuple[int,int]:
    """"
    Peeling decoder for LDPC codes
    :param mat_H: parity check matrix
    :param vec_y: received vector status, 1 for unknown/erased bits, 0 for known bits
    :return: number of remaining unknown bits
    """
    # preprocessing
    # declaring the arrays:
    H,W = mat_H.shape
    unknown_var_nodes = set(np.nonzero(vec_y)[0])
    cur_rows = [List.empty_list(types.int64) for _ in range(W)]
    unknown_var_nodes_per_chk = [List.empty_list(types.int64) for _ in range(H)]
    for chk in range(len(mat_H)):
        for var in mat_H[chk].nonzero()[0]:
            if var not in unknown_var_nodes:
                continue
            cur_rows[var].append(chk)
            unknown_var_nodes_per_chk[chk].append(var)

    peelable = list()
    for chk in range(H):
        if 1 == len(unknown_var_nodes_per_chk[chk]):
            peelable.append(chk)

    cnt = 0
    while len(peelable):
        cur_row = peelable.pop()
        if not len(unknown_var_nodes_per_chk[cur_row]):
            continue
        cnt += 1
        peeled_var = unknown_var_nodes_per_chk[cur_row].pop()
        for chk in cur_rows[peeled_var]:
            if not len(unknown_var_nodes_per_chk[chk]):
                continue
            unknown_var_nodes_per_chk[chk].remove(peeled_var)
            if 1 == len(unknown_var_nodes_per_chk[chk]):
                peelable.append(chk)
        unknown_var_nodes.remove(peeled_var)
    return cnt, len(unknown_var_nodes)


if __name__ == '__main__':
    # Testing!
    Mat_H = np.array([[1,1,0,1,0,0,0],
                      [0,1,1,0,1,0,0],
                      [0,0,1,1,0,1,0],
                      [0,0,0,1,1,0,1]])
    Vec_y = np.array([0,0,0,1,1,1,1])
    assert peeling_decoder(Mat_H, Vec_y) == 0
    Vec_y = np.array([1,0,0,1,1,1,0])
    assert peeling_decoder(Mat_H, Vec_y) == 0
    Mat_H = np.array([[0,1,0,1,0,0,0],
                      [0,1,1,0,1,0,0],
                      [0,0,1,1,0,1,0],
                      [0,0,0,1,1,0,1]])
    assert peeling_decoder(Mat_H, Vec_y) == 1
    Vec_y = np.array([1,1,1,1,1,1,1])
    assert peeling_decoder(Mat_H, Vec_y) == 7
    print("All tests passed!")
    Mat_H = np.array([[1,1,0,1,0,0,0],
                      [0,1,1,0,1,0,0],
                      [0,0,1,1,0,1,0],
                      [0,0,0,1,1,0,1]])
    Vec_y = np.array([0,0,0,1,1,1,1])
    begin = time.time()
    for _ in range(1000000):
        assert peeling_decoder(Mat_H, Vec_y) == 0
    print("Time elapsed:", time.time()-begin)

    Mat_H = np.array([[1,1,0,1,0,0,0],
                      [0,1,1,0,1,0,0],
                      [0,0,1,1,0,1,0],
                      [0,0,0,1,1,0,1]])
    Vec_y = np.array([0,0,0,1,1,1,1])
    assert jitpeeling_decoder(Mat_H, Vec_y)[1] == 0
    Vec_y = np.array([1,0,0,1,1,1,0])
    assert jitpeeling_decoder(Mat_H, Vec_y)[1] == 0
    Mat_H = np.array([[0,1,0,1,0,0,0],
                      [0,1,1,0,1,0,0],
                      [0,0,1,1,0,1,0],
                      [0,0,0,1,1,0,1]])
    assert jitpeeling_decoder(Mat_H, Vec_y)[1] == 1
    Vec_y = np.array([1,1,1,1,1,1,1])
    assert jitpeeling_decoder(Mat_H, Vec_y)[1] == 7
    print("JIT: All tests passed!")
    Mat_H = np.array([[1,1,0,1,0,0,0],
                      [0,1,1,0,1,0,0],
                      [0,0,1,1,0,1,0],
                      [0,0,0,1,1,0,1]])
    Vec_y = np.array([0,0,0,1,1,1,1])
    begin = time.time()
    for _ in range(1000000):
        assert jitpeeling_decoder(Mat_H, Vec_y)[1] == 0
    print("JIT Time elapsed:", time.time()-begin)
