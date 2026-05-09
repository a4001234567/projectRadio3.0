from itertools import product
from typing import List
import numpy as np

from codegen.qc.ruler import CircularRuler
from codegen.analysis.cr import CR_checker
from fileio.matrix_io import Writer


def write_board(rulers:List[List[CircularRuler]],block_size:int,filename:str='',truncate:int=0,**kwargs):
    '''
    Write the rulers to a file, while checking CR condition first.
    params:
        rulers: List[List[CircularRuler]]: the rulers to be written
        block_size: int: the size of each block
        filename: str: the name of the file to be written
        truncate: int: the number of characters to be truncated
        kwargs: dict: other parameters for the Writer
    '''
    j,l = len(rulers),len(rulers[0])
    tsize = block_size-truncate
    H = j*tsize; W = l*tsize
    with CR_checker(H,W) as writer_obj:
        for x,y in product(range(j),range(l)):
            writer_obj = (rulers[x][y]).visualize(writer_obj,x*tsize,y*tsize,limit=tsize)
    if not filename:
        return
    with Writer(H,W,filename,**kwargs) as writer_obj:
        for x,y in product(range(j),range(l)):
            writer_obj = (rulers[x][y]).visualize(writer_obj,x*tsize,y*tsize,limit=tsize)

def write_board_diag(rulers:List[List[CircularRuler]],block_size:int,filename:str='',truncate:int=0,**kwargs):
    '''
    Write the rulers to a file, while checking CR condition first.
    params:
        rulers: List[List[CircularRuler]]: the rulers to be written
        block_size: int: the size of each block
        filename: str: the name of the file to be written
        truncate: int: the number of characters to be truncated
        kwargs: dict: other parameters for the Writer
    '''
    j,l = len(rulers),len(rulers[0])
    tsize = block_size-truncate
    H = j*tsize; W = l*tsize
    with CR_checker(H,W) as writer_obj:
        for x,y in product(range(j),range(l)):
            writer_obj = (rulers[x][y]).visualize(writer_obj,x*tsize,y*tsize,limit=tsize)
    if not filename:
        return
    with Writer(H,W,filename,**kwargs) as writer_obj:
        for x,y in product(range(j),range(l)):
            writer_obj = (rulers[x][y]).visualize(writer_obj,x*tsize,y*tsize,limit=tsize)


def writeNDarray(rulers:List[List[CircularRuler]],block_size:int,truncate:int=0,**kwargs):
    '''
    Write the rulers to a numpy ND array, while checking CR condition first.
    params:
        rulers: List[List[CircularRuler]]: the rulers to be written
        block_size: int: the size of each block
        truncate: int: the number of characters to be truncated
        kwargs: dict: other parameters for the Writer
    '''
    j,l = len(rulers),len(rulers[0])
    tsize = block_size-truncate
    H = j*tsize; W = l*tsize
    result = np.zeros((H,W),dtype=np.int64)
    for x,y in product(range(j),range(l)):
        result = (rulers[x][y]).visualize(result,x*tsize,y*tsize,limit=tsize)
    return result


def generate_rulers(markers, block_size):
    result = []
    for rows in markers:
        result.append([])
        for mat_markers in rows:
            result[-1].append(CircularRuler(block_size,mat_markers))
    return result
