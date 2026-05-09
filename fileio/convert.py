from fileio.matrix_io import writer, _splitter
import numpy as np

with open('compMac/s2.94.494') as file:
    lines = file.__iter__()
    W,H = map(int,_splitter(next(lines).rstrip()))
    board = np.zeros((H,W),dtype=int)
    for idxRow,line in enumerate(lines):
        if not line:continue
        assert idxRow < H
        for idxCol in map(int,_splitter(line.rstrip())):
            if idxCol == 0:continue
            board[idxRow,idxCol-1] = 1
    writer('compMac/s2.94.494.zip',board,mode='sparse',compress=True)
