import numpy as np
import time
from utils.combinatorics import cnt_nzeros, nonzero
from itertools import combinations
from typing import Tuple, List, Iterable
#@numba.njit
def binary_repr(array):
    #return np.sum(1<<np.where(array)[0])
    return sum(1<<i for i,val in enumerate(array) if val)

#@numba.jit(nopython=False)
def check_CR(Mat:np.ndarray)->bool:
    h,w = Mat.shape
    if h>w:
        Mat = Mat.T
    binary_reprs = [binary_repr(i) for i in Mat]   
    for (idx_i,row_i),(idx_j,row_j) in combinations(enumerate(binary_reprs),2):
        if cnt_nzeros(row_i&row_j) > 1:
            print(idx_i,idx_j)
            return False
    return True

class CR_checker:
    '''
    A class to check if a matrix is a circular ruler.

    '''
    def __init__(self,H:int,W:int):
        self.shape = H,W
        if H > W:
            self.shape = W,H
            self.transpose = True
        else:
            self.transpose = False
        self.binary_repr = [0 for _ in range(W)]
    def __enter__(self):
        return self
    def __setitem__(self,coord:Tuple[int,int],value:int):
        if 1 != value:
            return
        if not self.transpose:
            x,y = coord
        else:
            y,x = coord
        assert 0 <= x < self.shape[0] and 0 <= y < self.shape[1], f'Index out of range: {coord}'
        self.binary_repr[x] |= 1<<y
    def __exit__(self,exc_type,exc_value,traceback):
        binary_reprs = self.binary_repr
        for (idx_i,row_i),(idx_j,row_j) in combinations(enumerate(binary_reprs),2):
            if cnt_nzeros(row_i&row_j) > 1:
                nonzeroPositions = ','.join(map(str, nonzero((row_i&row_j))))
                if not self.transpose:
                    raise ValueError(f'CR condition not satisfied!, see row {idx_i} and row {idx_j} and columns {nonzeroPositions}')
                else: raise ValueError(f'CR condition not satisfied!, see column {idx_i} and column {idx_j} and rows {nonzeroPositions}')

def test_binary_repr():
    assert binary_repr([0,1,1]) == 6
    assert binary_repr([1,0,0]) == 1
    a = binary_repr([0,0,1,1])
    b = binary_repr([1,0,1,0])
    assert cnt_nzeros(a) == 2
    assert cnt_nzeros(b) == 2
    assert cnt_nzeros(a&b) == 1

def test_check_CR():
    matrix = np.array([[0,0,1],[0,1,0],[1,0,0]])
    assert check_CR(matrix)
    matrix = np.array([[0,1,1],[1,1,1]])
    assert not check_CR(matrix)
    matrix = np.array([[0,1,1],[1,1,1]]).T
    assert not check_CR(matrix)
    assert check_CR(np.identity(3000))
    with CR_checker(3,3) as checker:
        checker[0,2] = 1
        checker[1,1] = 1
        checker[2,0] = 1
    with CR_checker(3,3) as checker:
        checker[0,0] = 1
        checker[0,2] = 1
        checker[1,1] = 1
        checker[2,0] = 1
