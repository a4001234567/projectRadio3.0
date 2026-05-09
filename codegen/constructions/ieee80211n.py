from collections.abc import Iterable
from itertools import combinations,product
import numpy as np
from utils.combinatorics import factorial
from functools import lru_cache
from typing import Set, List, Tuple
from codegen.analysis.cr import check_CR
from random import shuffle
import time
import sys
import re
from multiprocessing import Pool
from fileio.matrix_io import _neglect, _splitby, read_matrix, writer

@lru_cache(maxsize=65536)
def calculate_char(x: int,y: int,length: int)->int:
    if x < y:x,y = y,x
    return min(x-y,length+y-x)

@lru_cache(maxsize=256)
def two_side_mod(x,modp):
    return min(x,modp-x)

@lru_cache(maxsize=65536)
def calculate_characteristic(ruler: 'CircularRuler', cnt_rows: int) -> Tuple[set]:
    result = [ruler.bins]
    shifted_ruler = ruler
    for _ in range(1,cnt_rows):
        shifted_ruler = shifted_ruler << 1
        result.append(shifted_ruler-ruler)
    return tuple(result)

def check_collide(charA,charB):
    for sA,sB in zip(charA,charB):
        if not sA.isdisjoint(sB):
            return True
    return False

def add_char(charA,charB):
    return [sA.union(sB) for sA,sB in zip(charA,charB)]

def find_viable_combinations(cur_characteristic,remaining_choices,cap,cnt_rows):
    if cap == 0:
        return []
    for idx,choices in enumerate(remaining_choices,start=1):
        cur_char = calculate_characteristic(choices,cnt_rows)
        if check_collide(cur_characteristic,cur_char):
            continue
        res = find_viable_combinations(add_char(cur_characteristic,cur_char),remaining_choices[idx:],cap-1,cnt_rows)
        if res == False:
            continue
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
        self._bins = {}

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
            seen_distances = set()
            for x,y in combinations(self.markers,2):
                distance = calculate_char(x,y,self.length)
                if distance in seen_distances:
                    self._is_golomb = False
                    break
                seen_distances.add(distance)
            else:
                self._is_golomb = True
                self._bins = seen_distances
        return self._is_golomb 

    def __repr__(self):
        return '+'.join(f'x^{marker}' for marker in sorted(self.markers))
        return f'golomb ruler of {",".join(map(str,sorted(self.markers)))}:mod {self.length}'

    def __lshift__(self,value: int) -> 'CircularRuler':
        if not self.markers:
            return CircularRuler(self.length)
        #f = lambda x:x if x < self.length else x-self.length
        new_markers = {(i << value) for i in self.markers}
        new_ruler = CircularRuler(self.length, new_markers)
        new_ruler._is_golomb = self._is_golomb
        #new_ruler._bins = {f(i<<1) for i in self._bins}
        return new_ruler

    def visualize(self,obj,x,y,limit=None):
        if not limit:
            limit = self.length
        for nx in range(min(self.length,limit)):
            for ny in range(min(self.length,limit)):
                if ((ny-nx)+self.length)%self.length in self.markers:
                    obj[x+nx,y+ny] = 1
                else:
                    obj[x+nx,y+ny] = 0
        return obj

    def __len__(self):
        return len(self.markers)

    def __sub__(self,another_ruler: 'CircularRuler') -> set:
        result = set()
        f = lambda x:x+self.length if x < 0 else x
        for x in self.markers:
            for y in another_ruler.markers:
                result.add(f(x-y)) # x1-y1 = x2-y2 will not happen, otherwise x1-x2 = y1-y2, for 2 golomb rulers
        return result

def draw_board(rulers,block_size,truncate=0):
    j,l = len(rulers),len(rulers[0])
    tsize = block_size-truncate
    board = np.zeros((j*tsize,l*tsize))
    for x,y in product(range(j),range(l)):
        board = (rulers[x][y]).visualize(board,x*tsize,y*tsize,limit=tsize)
    return board

def generate_legal_rulers(block_size: int, j: int, weight: int) -> List[CircularRuler]:
    legal_rulers = []
    for markers in combinations(range(block_size),weight):
        ruler = CircularRuler(block_size, markers)
        if not ruler.check_golomb():
            continue
        is_valid = True
        shifted_markers = set(ruler.bins)
        for i in range(1,j):
            new_markers = {two_side_mod((m<<i) % block_size,block_size) for m in ruler.bins}
            if not new_markers.isdisjoint(shifted_markers):
                is_valid = False
                break
            shifted_markers.update(new_markers)

        if is_valid:
            legal_rulers.append(ruler)

    return legal_rulers

def find_golomb_matrix(j:int,l:int,block_size:int):
    assert 2*j <= block_size, 'No Solution'
    assert 2*l <= block_size, 'No Solution'
    legals = generate_legal_rulers(block_size,j,2)
    shuffle(legals)
    res = find_viable_combinations([set() for _ in range(block_size-1)],legals,l,j)
    board = draw_board(j,l,block_size,res,truncate=1)
    assert check_CR(board)
    represents = ''
    #represents += f'>j={j};l={l};p={block_size}//Matrix of size ({j}*{block_size})*({l}*{block_size})\n'
    #represents += f'>Polys:{";".join(map(str,res))}\n'
    #represents += f'>CR condition checked\n'
    #represents += f'>Untruncated Matrix\n'
    represents += '\n'.join(map(lambda line:'['+' '.join(map(lambda x:str(int(x)),line))+']',board))+']\n'
    #represents += f'>Truncated Matrix\n'
    #board = draw_board(j,l,block_size,res,truncate=1)
    #represents += '['+','.join(map(lambda line:'['+','.join(map(lambda x:str(int(x)),line))+']',board))+']\n'
    with open(f'{j}*{l}*{block_size}.txt','w') as file:
        file.write(represents)

def func(task):
        task_start = time.time()
        print(f'Calculating {task}')
        find_golomb_matrix(*task)
        print(f'Use {time.time()-task_start:.3f}s')
        return 1

def generate_rulers(markers, block_size):
    result = []
    for rows in markers:
        result.append([])
        for mat_markers in rows:
            result[-1].append(CircularRuler(block_size,mat_markers))
    return result

if __name__ == '__main__':
    ruler1 = CircularRuler(7,markers=(0,2,6))
    assert ruler1.check_golomb()
    ruler1.add(11)
    assert not ruler1.check_golomb()
    ruler1.markers.remove(4)
    ruler2 = CircularRuler(7,markers=(0,1,3))
    ruler2<<=1
    assert (0 in ruler2) and (2 in ruler2) and (6 in ruler2) and (not 1 in ruler2)
    assert (3 in ruler2.bins) and (2 in ruler2.bins) and (1 in ruler2.bins)

    N = '1944'; rate = '3/4'
    name = f'D2_{N}b_R{rate.replace("/","")}'

    if N == '1944':
        Z = 81
        if rate == '1/2':
            string = """[57 -1 -1 -1 50 -1 11 -1 50 -1 79 -1 1 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
                      3 -1 28 -1 0 -1 -1 -1 55 7 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1;
                    30 -1 -1 -1 24 37 -1 -1 56 14 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1;
                    62 53 -1 -1 53 -1 -1 3 35 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1;
                    40 -1 -1 20 66 -1 -1 22 28 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1;
                    0 -1 -1 -1 8 -1 42 -1 50 -1 -1 8 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1;
                    69 79 79 -1 -1 -1 56 -1 52 -1 -1 -1 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1;
                    65 -1 -1 -1 38 57 -1 -1 72 -1 27 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1;
                    64 -1 -1 -1 14 52 -1 -1 30 -1 -1 32 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1;
                    -1 45 -1 70 0 -1 -1 -1 77 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1;
                    2 56 -1 57 35 -1 -1 -1 -1 -1 12 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0;
                    24 -1 61 -1 60 -1 -1 27 51 -1 -1 16 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0"""
        elif rate == '2/3':
            string = """61 75 4 63 56 -1 -1 -1 -1 -1 -1 8 -1 2 17 25 1 0 -1 -1 -1 -1 -1 -1;
                    56 74 77 20 -1 -1 -1 64 24 4 67 -1 7 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1;
                    28 21 68 10 7 14 65 -1 -1 -1 23 -1 -1 -1 75 -1 -1 -1 0 0 -1 -1 -1 -1;
                    48 38 43 78 76 -1 -1 -1 -1 5 36 -1 15 72 -1 -1 -1 -1 -1 0 0 -1 -1 -1;
                    40 2 53 25 -1 52 62 -1 20 -1 -1 44 -1 -1 -1 -1 0 -1 -1 -1 0 0 -1 -1;
                    69 23 64 10 22 -1 21 -1 -1 -1 -1 -1 68 23 29 -1 -1 -1 -1 -1 -1 0 0 -1;
                    12 0 68 20 55 61 -1 40 -1 -1 -1 52 -1 -1 -1 44 -1 -1 -1 -1 -1 -1 0 0;
                    58 8 34 64 78 -1 -1 11 78 24 -1 -1 -1 -1 -1 58 1 -1 -1 -1 -1 -1 -1 0"""
        elif rate == '3/4':
            string = """[48 29 28 39 9 61 -1 -1 -1 63 45 80 -1 -1 -1 37 32 22 1 0 -1 -1 -1 -1;
                    4 49 42 48 11 30 -1 -1 -1 49 17 41 37 15 -1 54 -1 -1 -1 0 0 -1 -1 -1;
                    35 76 78 51 37 35 21 -1 17 64 -1 -1 -1 59 7 -1 -1 32 -1 -1 0 0 -1 -1;
                    9 65 44 9 54 56 73 34 42 -1 -1 -1 35 -1 -1 -1 46 39 0 -1 -1 0 0 -1;
                    3 62 7 80 68 26 -1 80 55 -1 36 -1 26 -1 9 -1 72 -1 -1 -1 -1 -1 0 0;
                    26 75 33 21 69 59 3 38 -1 -1 -1 35 -1 62 36 26 -1 -1 1 -1 -1 -1 -1 0"""
        elif rate == '5/6':
            string = """13 48 80 66 4 74 7 30 76 52 37 60 -1 49 73 31 74 73 23 -1 1 0 -1 -1;
                    69 63 74 56 64 77 57 65 6 16 51 -1 64 -1 68 9 48 62 54 27 -1 0 0 -1;
                    51 15 0 80 24 25 42 54 44 71 71 9 67 35 -1 58 -1 29 -1 53 0 -1 0 0;
                    16 29 36 41 44 56 59 37 50 24 -1 65 4 65 52 -1 4 -1 73 52 1 -1 -1 0"""
    elif N == '1296':
        Z = 54
        if rate == '1/2':
            string = """[40 -1 -1 -1 22 -1 49 23 43 -1 -1 -1 1 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
                    50 1 -1 -1 48 35 -1 -1 13 -1 30 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1;
                    39 50 -1 -1 4 -1 2 -1 -1 -1 -1 49 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1;
                    33 -1 -1 38 37 -1 -1 4 1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1;
                    45 -1 -1 -1 0 22 -1 -1 20 42 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1;
                    51 -1 -1 48 35 -1 -1 -1 44 -1 18 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1;
                    47 11 -1 -1 -1 17 -1 -1 51 -1 -1 -1 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1;
                    5 -1 25 -1 6 -1 45 -1 13 40 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1;
                    33 -1 -1 34 24 -1 -1 -1 23 -1 -1 46 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1;
                    1 -1 27 -1 1 -1 -1 -1 38 -1 44 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1;
                    -1 18 -1 -1 23 -1 -1 8 0 35 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0;
                    49 -1 17 -1 30 -1 -1 -1 34 -1 -1 19 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0]"""
        elif rate == '2/3':
            string = """[39 31 22 43 -1 40 4 -1 11 -1 -1 50 -1 -1 -1 6 1 0 -1 -1 -1 -1 -1 -1;
                    25 52 41 2 6 -1 14 -1 34 -1 -1 -1 24 -1 37 -1 -1 0 0 -1 -1 -1 -1 -1;
                    43 31 29 0 21 -1 28 -1 -1 2 -1 -1 7 -1 17 -1 -1 -1 0 0 -1 -1 -1 -1;
                    20 33 48 -1 4 13 -1 26 -1 -1 22 -1 -1 46 42 -1 -1 -1 -1 0 0 -1 -1 -1;
                    45 7 18 51 12 25 -1 -1 -1 50 -1 -1 5 -1 -1 -1 0 -1 -1 -1 0 0 -1 -1;
                    35 40 32 16 5 -1 -1 18 -1 -1 43 51 -1 32 -1 -1 -1 -1 -1 -1 -1 0 0 -1;
                    9 24 13 22 28 -1 -1 37 -1 -1 25 -1 -1 52 -1 13 -1 -1 -1 -1 -1 -1 0 0;
                    32 22 4 21 16 -1 -1 -1 27 28 -1 38 -1 -1 -1 8 1 -1 -1 -1 -1 -1 -1 0]"""
        elif rate == '3/4':
            string = """[39 40 51 41 3 29 8 36 -1 14 -1 6 -1 33 -1 11 -1 4 1 0 -1 -1 -1 -1;
                    48 21 47 9 48 35 51 -1 38 -1 28 -1 34 -1 50 -1 50 -1 -1 0 0 -1 -1 -1;
                    30 39 28 42 50 39 5 17 -1 6 -1 18 -1 20 -1 15 -1 40 -1 -1 0 0 -1 -1;
                    29 0 1 43 36 30 47 -1 49 -1 47 -1 3 -1 35 -1 34 -1 0 -1 -1 0 0 -1;
                    1 32 11 23 10 44 12 7 -1 48 -1 4 -1 9 -1 17 -1 16 -1 -1 -1 -1 0 0;
                    13 7 15 47 23 16 47 -1 43 -1 29 -1 52 -1 2 -1 53 -1 1 -1 -1 -1 -1 0]"""
        elif rate == '5/6':
            string = """48 29 37 52 2 16 6 14 53 31 34 5 18 42 53 31 45 -1 46 52 1 0 -1 -1;
                    17 4 30 7 43 11 24 6 14 21 6 39 17 40 47 7 15 41 19 -1 -1 0 0 -1;
                    7 2 51 31 46 23 16 11 53 40 10 7 46 53 33 35 -1 25 35 38 0 -1 0 0;
                    19 48 41 1 10 7 36 47 5 29 52 52 31 10 26 6 3 2 -1 51 1 -1 -1 0 """
    elif N == '648':
        Z = 27
        if rate == '1/2':
            string = """[0 -1 -1 -1 0 0 -1 -1 0 -1 -1 0 1 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
                    22 0 -1 -1 17 -1 0 0 12 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1;
                    6 -1 0 -1 10 -1 -1 -1 24 -1 0 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1;
                    2 -1 -1 0 20 -1 -1 -1 25 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1;
                    23 -1 -1 -1 3 -1 -1 -1 0 -1 9 11 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1;
                    24 -1 23 1 17 -1 3 -1 10 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1;
                    25 -1 -1 -1 8 -1 -1 -1 7 18 -1 -1 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1;
                    13 24 -1 -1 0 -1 8 -1 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1;
                    7 20 -1 16 22 10 -1 -1 23 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1;
                    11 -1 -1 -1 19 -1 -1 -1 13 -1 3 17 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1;
                    25 -1 8 -1 23 18 -1 14 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0;
                    3 -1 -1 -1 16 -1 -1 2 25 5 -1 -1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0];"""
        elif rate == '2/3':
            string = """[25 26 14 -1 20 -1 2 -1 4 -1 -1 8 -1 16 -1 18 1 0 -1 -1 -1 -1 -1 -1;
                    10 9 15 11 -1 0 -1 1 -1 -1 18 -1 8 -1 10 -1 -1 0 0 -1 -1 -1 -1 -1;
                    16 2 20 26 21 -1 6 -1 1 26 -1 7 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1;
                    10 13 5 0 -1 3 -1 7 -1 -1 26 -1 -1 13 -1 16 -1 -1 -1 0 0 -1 -1 -1;
                    23 14 24 -1 12 -1 19 -1 17 -1 -1 -1 20 -1 21 -1 0 -1 -1 -1 0 0 -1 -1;
                    6 22 9 20 -1 25 -1 17 -1 8 -1 14 -1 18 -1 -1 -1 -1 -1 -1 -1 0 0 -1;
                    14 23 21 11 20 -1 24 -1 18 -1 19 -1 -1 -1 -1 22 -1 -1 -1 -1 -1 -1 0 0;
                    17 11 11 20 -1 21 -1 26 -1 3 -1 -1 18 -1 26 -1 1 -1 -1 -1 -1 -1 -1 0]"""
        elif rate == '3/4':
            string = """[16 17 22 24 9 3 14 -1 4 2 7 -1 26 -1 2 -1 21 -1 1 0 -1 -1 -1 -1;
                    25 12 12 3 3 26 6 21 -1 15 22 -1 15 -1 4 -1 -1 16 -1 0 0 -1 -1 -1;
                    25 18 26 16 22 23 9 -1 0 -1 4 -1 4 -1 8 23 11 -1 -1 -1 0 0 -1 -1;
                    9 7 0 1 17 -1 -1 7 3 -1 3 23 -1 16 -1 -1 21 -1 0 -1 -1 0 0 -1;
                    24 5 26 7 1 -1 -1 15 24 15 -1 8 -1 13 -1 13 -1 11 -1 -1 -1 -1 0 0;
                    2 2 19 14 24 1 15 19 -1 21 -1 2 -1 24 -1 3 -1 2 1 -1 -1 -1 -1 0]"""
        elif rate == '5/6':
            string = """[17 13 8 21 9 3 18 12 10 0 4 15 19 2 5 10 26 19 13 13 1 0 -1 -1;
                    3 12 11 14 11 25 5 18 0 9 2 26 26 10 24 7 14 20 4 2 -1 0 0 -1;
                    22 16 4 3 10 21 12 5 21 14 19 5 -1 8 5 18 11 5 5 15 0 -1 0 0;
                    7 7 14 14 4 16 16 24 24 10 1 7 15 6 10 26 8 18 21 14 1 -1 -1 0]"""


    markers = []
    splitter = _splitby(' ')
    negletter = _neglect(*tuple("';{}[]\""))
    for line in string.split('\n'):
        markers.append(list())
        for item in splitter(negletter(line.lstrip())):
            markers[-1].append([] if '-' in item else [int(item)])
    print(markers)
    res = generate_rulers(markers,Z)
    board = draw_board(res,Z)
    assert check_CR(board)
    writer(f'{name}.txt',board)
    exit()
