#!/usr/bin/env python
# 
# Copyright 2013 IIT Bombay.
# Author: Manu T S
# 
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

import numpy as np
from fileio.matrix_io import get_reader, writer
import sys
#from functool import partial
try:
    from tqdm import tqdm
except ImportError:
    def temp(feed,*arg,**args):
        return feed
    tqdm = temp

import numpy as np

class peg():

    """
    Progressive edge growth algorithm for generating
    LDPC matrices. The algorithm is obtained from [1]
    """

    def __init__(self, nvar, nchk, degree_sequence):
        self.degree_sequence = degree_sequence
        self.nvar = nvar
        self.nchk = nchk
        self.H: np.ndarray = np.zeros((nchk, nvar), dtype = np.int32)
        self.sym_degrees: np.ndarray = np.zeros(nvar, dtype = np.int32)
        self.chk_degrees: np.ndarray = np.zeros(nchk, dtype = np.int32)
        self.var_adj: tuple[set[int]] = tuple((set() for _ in range(nvar)))
        self.chk_adj: tuple[set[int]] = tuple((set() for _ in range(nchk)))

    def grow_edge(self, var:int, chk:int):
        self.var_adj[var].add(chk)
        self.chk_adj[chk].add(var)
        self.H[chk, var] = 1
        self.sym_degrees[var] += 1
        self.chk_degrees[chk] += 1

    def bfs(self, var):
        var_list = {var}

        cur_chk_list = set();new_chk_list = set()
        chk_Q = set();var_Q = {var}
        while True:
            for _var in var_Q:
                for i in self.var_adj[_var] - cur_chk_list:
                    new_chk_list.add(i)
                    chk_Q.add(i)
            var_Q = set()
            for _chk in chk_Q:
                for j in self.chk_adj[_chk] - var_list:
                    var_list.add(j)
                    var_Q.add(j)
            chk_Q = set()
            if len(new_chk_list) == self.nchk or new_chk_list == cur_chk_list:
                return self.find_smallest_chkQ(cur_chk_list)
            cur_chk_list = new_chk_list.copy()
    
    def find_smallest_chkQ(self, cur_chk_set):
        return min(tuple((i for i in range(self.nchk) if i not in cur_chk_set)), key=lambda x:self.chk_degrees[x])

    def progressive_edge_growth(self):
        for var in tqdm(range(self.nvar),desc='Edge Growing:'):
            for k in range(self.degree_sequence[var]):
                if k == 0:
                    smallest_degree_chk = self.find_smallest_chkQ(set())
                    self.grow_edge(var, smallest_degree_chk)
                else:
                    chk = self.bfs(var)
                    self.grow_edge(var, chk)

"""
References

"Regular and Irregular Progressive-Edge Growth Tanner Graphs",
Xiao-Yu Hu, Evangelos Eleftheriou and Dieter M. Arnold.
IEEE Transactions on Information Theory, January 2005.
"""

def PEG_LIKE(H_matrix:np.ndarray)->np.ndarray:
    nchk, nvar = H_matrix.shape
    degrees = H_matrix.sum(axis=0)
    sim_peg = peg(nvar, nchk, degrees)
    sim_peg.progressive_edge_growth()
    resH = sim_peg.H
    assert resH.shape == H_matrix.shape
    print(type(resH.sum(axis=0)))
    print(resH.sum(axis=0).shape)
    print(degrees.shape)
    print(type(degrees))
    #print(resH.sum(axis=0))
    #print(degrees)
    assert np.all(resH.sum(axis=0) == degrees)
    return resH

if __name__ == "__main__":
    reader = get_reader()
    assert sys.argv.__len__() >= 3
    _, filein, fileout, *_ = sys.argv
    H_matrix = reader(filein)
    H_like = PEG_LIKE(H_matrix)
    if fileout.endswith('.zip'):
        writer(fileout,H_like,mode='sparse',comments=f'PEG_processed {filein}',compress=True)
    else:
        writer(fileout,H_like,mode='sparse',comments=f'PEG_processed {filein}')
