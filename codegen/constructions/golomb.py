from collections.abc import Iterable
from itertools import combinations,product,takewhile
import numpy as np
from utils.combinatorics import factorial, combination_with_order, binomial
from functools import lru_cache
from fileio.matrix_io import Writer
from typing import Set, List, Tuple
from codegen.analysis.cr import check_CR, CR_checker
from random import shuffle
import time
import sys
import re
from multiprocessing import Pool

@lru_cache(maxsize=65536)
def calculate_char(x: int,y: int,length: int)->int:
    if x < y:x,y = y,x
    return min(x-y,length+y-x)

@lru_cache(maxsize=256)
def two_side_mod(x,modp):
    return min(x,modp-x)

@lru_cache(maxsize=65536)
def calculate_characteristic(ruler_row: List['CircularRuler'], cnt_rows: int) -> Tuple[set]:
    result = [ruler.bins for ruler in ruler_row]
    for rulerA,rulerB in combinations(ruler_row,2):
        result.append(rulerA-rulerB)

    return tuple(result)


def check_collide(charA,charB):
    for sA,sB in zip(charA,charB):
        if not sA.isdisjoint(sB):
            return True
    return False

def add_char(charA,charB):
    return [sA.union(sB) for sA,sB in zip(charA,charB)]

def find_viable_combinations(cur_characteristic,remaining_choices,cap,cnt_rows):
    #print(cur_characteristic)
    if cap == 0:
        return []
    for idx,choices in enumerate(remaining_choices,start=1):
        cur_char = calculate_characteristic(choices,cnt_rows)
        #print(choices)
        #print(cur_char)
        if check_collide(cur_characteristic,cur_char):
            continue
        res = find_viable_combinations(add_char(cur_characteristic,cur_char),remaining_choices[idx:],cap-1,cnt_rows)
        if res != False:
            res.append(choices)
            return res
    return False

class CircularRuler:
    __slots__ = ('length', 'markers', '_binary_repr', '_is_golomb', '_bins')
    def __init__(self,length:int,markers:Iterable=None):
        self.length = length
        self.markers = {i%self.length for i in markers} if markers else set()
        self._binary_repr = None
        self._is_golomb = None
        self._bins = set()

    def add(self,element):
        self.markers.add(element%self.length)
        self._is_golomb = None

    @property
    def bins(self)->set:
        if not self._bins:
            self.check_golomb()
        return self._bins

    @property
    def binary_repr(self) -> int:
        if self._binary_repr is None:
            self._binary_repr = sum(1<<i for i in self.markers)
        return self._binary_repr

    def __int__(self) -> int:
        return self.binary_repr

    def __contains__(self,element):
        return element in self.markers

    def check_golomb(self) -> bool:
        if self._is_golomb is None:
            self._bins = set()
            bins = self._bins
            for x,y in combinations(self.markers,2):
                distance = calculate_char(x,y,self.length)
                if distance in bins:
                    self._is_golomb = False
                    break
                bins.add(distance)
            else:
                self._is_golomb = True
        return self._is_golomb 

    def __repr__(self):
        return '+'.join(f'x^{marker}' for marker in sorted(self.markers))
        return f'golomb ruler of {",".join(map(str,sorted(self.markers)))}:mod {self.length}'

    def __lshift__(self,value: int) -> 'CircularRuler':
        if not self.markers:
            return CircularRuler(self.length)
        new_markers = {(i << value) for i in self.markers}
        new_ruler = CircularRuler(self.length, new_markers)
        new_ruler._is_golomb = self._is_golomb
        return new_ruler

    def visualize(self,writer_obj,x,y,limit=None):
        if not limit:
            limit = self.length
        else:
            limit = min(self.length,limit)
        for nx in range(limit):
            for marker in self.markers:
                new_y = (nx+marker)%self.length
                if new_y >= limit: continue
                writer_obj[x+nx,y+new_y] = 1
        return writer_obj

    def __len__(self):
        return len(self.markers)

    def __sub__(self,another_ruler: 'CircularRuler') -> set:
        result = set()
        f = lambda x:x+self.length if x < 0 else x
        for x in self.markers:
            for y in another_ruler.markers:
                result.add(f(x-y)) # x1-y1 = x2-y2 will not happen, otherwise x1-x2 = y1-y2, for 2 golomb rulers
        return result

def transpose2D(matrix:List[List[int]])->List[List[int]]:
    return [list(i) for i in zip(*matrix)]

def draw_board(j,l,block_size,rulers:List[List['CircularRuler']],board=None,truncate=0):
    tsize = block_size-truncate
    if not board: board = np.zeros((j*tsize,l*tsize))
    for x,y in product(range(j),range(l)):
        board = rulers[x][y].visualize(board,x*tsize,y*tsize,limit=tsize)
    return board

def generate_legal_rulers(block_size: int, j: int, weight: int) -> List[CircularRuler]:
    legal_rulers = []
    marker_combinations = tuple(map(lambda x:CircularRuler(block_size,x),combinations(range(block_size),weight)))
    marker_combinations = list(takewhile(lambda x:x.check_golomb(),marker_combinations))
    shuffle(marker_combinations)
    marker_combinations = tuple(marker_combinations[:500])
    for markers_col in combination_with_order(marker_combinations,j):
        #for head in marker_combinations:
        #markers_col = [head,head<<1]
        cur_char = set()
        for ruler in markers_col:
            if not ruler.bins.isdisjoint(cur_char):
                break
            cur_char.update(ruler.bins)
        else:
            legal_rulers.append(markers_col)
    return legal_rulers

def find_golomb_matrix(j:int,l:int,block_size:int):
    legals = generate_legal_rulers(block_size,j,2)
    shuffle(legals)
    legals = legals[:5000]
    char_length = binomial(j,2) + j
    res = find_viable_combinations([set() for _ in range(char_length)],legals,l,j)
    res = transpose2D(res)
    print(res)
    truncate = 1
    H = j*(block_size-truncate)
    W = l*(block_size-truncate)
    with CR_checker(H,W) as checker:
        draw_board(j,l,block_size,res,checker,truncate=truncate)
    with Writer(H = H, W = W, filename=f'j={j},l={l},p={block_size}.zip',mode='sparse',compress=True) as writer:
        draw_board(j,l,block_size,res,writer,truncate=truncate)
    

if __name__ == '__main__':
    find_golomb_matrix(4,79,711)