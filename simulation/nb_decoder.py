#
#  Pocket SDR Python Library - NB-LDPC Decoder Functions
#
#  References:
#  [1] BeiDou Navigation Satellite System Signal In Space Interface Control
#      Document Open Service Signal B1C (Version 1.0), December, 2017
#  [2] E.Li et al., Trellis-based Extended Min-Sum algorithm for non-binary LDPC
#      codes and its hardware structure, IEEE Trans. on Communications, 2013
#
#  Author:
#  T.TAKASU
#
#  History:
#  2024-01-25  1.0  new
#
from math import *
import numpy as np
import sympy
import numba
#from numba.extending import register_jitable
from typing import List, Mapping

# By defining the character of the field:q, and the primitive polynomial,
# We generate the GF(q^n) field, alongside with their parameters.

# GF(q^n) parameters -----------------------------------------------------------
# The char of the field
q = 2
# The primitive polynomial, defined using sympy
x = sympy.symbols('x')
primitive_poly = sympy.poly(x**7+x+1, domain='GF(2)') # The generator polynomial
N_GF = primitive_poly.degree() # The degree of the generator polynomial
Q_GF = 1 << N_GF # The number of elements in GF(q^n)

NM_EMS = 4 # LLR truncation size of EMS
ERR_PROB = 0.9 # error probability of input codes
MAX_ITER = 15

# GF(q) tables -----------------------------------------------------------------
GF_VEC = [] # The vectors of GF(q), representing with binary format

def poly_to_int(poly):
    coeffs = [c % 2 for c in poly.all_coeffs()][::-1]
    return sum(b << i for i, b in enumerate(coeffs))

curElement = sympy.poly(1, x, domain='GF(2)')
for i in range(1, Q_GF):
    GF_VEC.append(poly_to_int(curElement))
    curElement = curElement * sympy.poly(x, domain='GF(2)')
    curElement = curElement % primitive_poly

GF_POW = [0 for _ in range(Q_GF)]
for i in range(1, Q_GF):
    GF_POW[GF_VEC[i-1]] = i - 1

'''
GF_VEC = ( # power -> vector ([1])
     1,  2,  4,  8, 16, 32,  3,  6, 12, 24, 48, 35,  5, 10, 20, 40,
    19, 38, 15, 30, 60, 59, 53, 41, 17, 34,  7, 14, 28, 56, 51, 37,
     9, 18, 36, 11, 22, 44, 27, 54, 47, 29, 58, 55, 45, 25, 50, 39,
    13, 26, 52, 43, 21, 42, 23, 46, 31, 62, 63, 61, 57, 49, 33)

GF_POW = ( # vector -> power ([1])
     0,  0,  1,  6,  2, 12,  7, 26,  3, 32, 13, 35,  8, 48, 27, 18,
     4, 24, 33, 16, 14, 52, 36, 54,  9, 45, 49, 38, 28, 41, 19, 56,
     5, 62, 25, 11, 34, 31, 17, 47, 15, 23, 53, 51, 37, 44, 55, 40,
    10, 61, 46, 30, 50, 22, 39, 43, 29, 60, 42, 21, 20, 59, 57, 58)
'''

# initialize GF(q) table -------------------------------------------------------
GF_MUL = [] # multiply
def init_table():
    global GF_MUL
    if len(GF_MUL):
        return
    GF_MUL = np.zeros((Q_GF, Q_GF), dtype='uint8')
    for i in range(1, Q_GF):
        for j in range(1, Q_GF):
            GF_MUL[i][j] = GF_VEC[(GF_POW[i] + GF_POW[j]) % (Q_GF - 1)]

# convert binary codes to GF(q) codes ------------------------------------------
#@numba.jit(nopython=True, nogil=True)
def bin2gf(syms):
    n = len(syms) // N_GF
    code = np.zeros(n, dtype='uint8')
    for i in range(n):
        for j in range(N_GF):
            code[i] = (code[i] << 1) + syms[i*N_GF+j]
    return code

# convert GF(q) codes to binary codes ------------------------------------------
#@numba.jit(nopython=True, nogil=True)
def gf2bin(code):
    n = len(code)
    syms = np.zeros(n * N_GF, dtype='uint8')
    for i in range(n):
        for j in range(N_GF):
            syms[i*N_GF+j] = (code[i] >> (N_GF - 1 - j)) & 1
    return syms

# Tanner graph edges -----------------------------------------------------------
def graph_edge(H_idx, H_ele):
    ie, je, he = [], [], []
    for i in range(len(H_idx)):
        for j in range(len(H_idx[i])):
            ie.append(i)
            je.append(H_idx[i][j])
            he.append(H_ele[i][j])
    return ie, je, he, len(he) # CN-index, VN-index, H_ij of edges

# initialize LLR ---------------------------------------------------------------
def init_LLR(code, err_prob):
    L = np.zeros((len(code), Q_GF), dtype='float32')
    for i in range(len(code)):
        for j in range(Q_GF):
            nerr = bin(code[i] ^ j).count('1')
            L[i][j] = -log(err_prob) * nerr
    return L

# parity check -----------------------------------------------------------------
#@numba.jit(nopython=True, nogil=True)
def check_parity(ie, je, he, m, code):
    s = np.zeros(m, dtype='uint8')
    for i in range(len(ie)):
        s[ie[i]] ^= GF_MUL[he[i]][code[je[i]]]
    return np.all(s == 0)

# permute VN->CN message -------------------------------------------------------
def permute_V2C(h, V2C):
    V2C_p = np.zeros(Q_GF, dtype='float32')
    for i in range(Q_GF):
        V2C_p[GF_MUL[h][i]] = V2C[i]
    return V2C_p

def permute_V2C(h, V2C):
    V2C_p = np.zeros(Q_GF, dtype='float32')
    np.put(V2C_p, GF_MUL[h], V2C)
    return V2C_p


# permute CN->VN message -------------------------------------------------------
def permute_C2V(h, C2V):
    C2V_p = np.zeros(Q_GF, dtype='float32')
    for i in range(Q_GF):
        C2V_p[i] = C2V[GF_MUL[h][i]]
    return C2V_p

def permute_C2V(h, C2V):
    return C2V[GF_MUL[h]].copy()

# extended-min-sum (EMS) of LLRs ([2]) -----------------------------------------
def ext_min_sum(L1, L2):
    if len(L1) == 0:
        return L2
    idx1 = np.argsort(L1)
    idx2 = np.argsort(L2)
    maxL = L1[idx1[NM_EMS-1]] + L2[idx2[NM_EMS-1]]
    Ls = np.full(Q_GF, maxL, dtype='float32')
    
    for i in idx1[:NM_EMS]:
        for j in idx2[:NM_EMS]:
            if L1[i] + L2[j] < Ls[i^j]:
                Ls[i^j] = L1[i] + L2[j]
    return Ls

def ext_min_sum(L1:np.ndarray, L2:np.ndarray)->np.ndarray:
    idx1 = np.argpartition(L1, NM_EMS)[:NM_EMS]
    idx2 = np.argpartition(L2, NM_EMS)[:NM_EMS]
    l1 = L1[idx1]
    l2 = L2[idx2]
    maxL = l1[-1]+l2[-1]
    Ls = np.full(Q_GF, maxL, dtype='float64')
    np.minimum.at(Ls, np.bitwise_xor.outer(idx1, idx2).ravel(), np.add.outer(l1, l2).ravel())
    return Ls.copy()

cnt = 5000

@numba.jit(nopython=False)
def ext_min_sum(L1:np.ndarray, L2:np.ndarray)->np.ndarray:
    L1 = L1.copy()
    L2 = L2.copy()
    idx1 = np.argsort(L1)[:NM_EMS]
    idx2 = np.argsort(L2)[:NM_EMS]
    maxL = L1[idx1[-1]] + L2[idx2[-1]]
    Ls = np.full(Q_GF, maxL, dtype='float64')
    
    for i in idx1:
        for j in idx2:
            if L1[i] + L2[j] < Ls[i^j]:
                Ls[i^j] = L1[i] + L2[j]
    return Ls.copy()

#@numba.jit(nopython=True)
def allButOneSum(sumFunc, vals):
    '''
    This function computes the sum of all elements in vals except for the one,
    enumerating the excluded element, where the sum function is set by sumFunc.
    Input:
        sumFunc: the function to compute the sum
        vals: the list of values to be summed
    Output:
        the sum of all elements in vals except for the one, enumerating the excluded element,
        i.e., the kth element in output is the sum of all elements in vals except for the kth element.
    '''
    left = [vals[0]]
    n = len(vals)
    for i in range(1, len(vals)):
        left.append(sumFunc(left[i-1], vals[i]))
    right = [vals[-1]]
    for i in range(len(vals)-2, -1, -1):
        right.append(sumFunc(vals[i], right[-1]))
    right.reverse()
    result = [right[1]] + \
             [sumFunc(left[i-1], right[i+1]) for i in range(1, n-1)] + \
             [left[-2]]
    return result

# decode NB-LDPC ---------------------------------------------------------------
def decode_NB_LDPC(H_idx, H_ele, m, n, syms):
    
    # initialize GF(q) tables
    init_table()
    
    # convert binary codes to GF(q) codes
    code = bin2gf(syms)
    
    # Tanner graph edges
    #ie, je, he, ne = graph_edge(H_idx, H_ele)
    ie, je = H_idx
    he = H_ele
    ne = len(he)
    V2C = np.zeros((ne, Q_GF), dtype='float32')
    C2V = np.zeros((ne, Q_GF), dtype='float32')
    
    # initialize LLR and VN->CN messages
    L = init_LLR(code, ERR_PROB)
    for i in range(ne):
        V2C[i] = permute_V2C(he[i], L[je[i]])
    
    jeValInv:Mapping[int,List[int]]= dict()
    for idx, val in enumerate(je):
        if val not in jeValInv:
            jeValInv[val] = []
        jeValInv[val].append(idx)
    
    ieValInv:Mapping[int,List[int]]= dict()
    for idx, val in enumerate(ie):
        if val not in ieValInv:
            ieValInv[val] = []
        ieValInv[val].append(idx)

    for iter in range(MAX_ITER):
        print(iter)
        print(code[:10])
        # parity check
        if check_parity(ie, je, he, m, code):
            syms_dec = gf2bin(code)
            nerr = np.count_nonzero(syms_dec ^ syms)
            return syms_dec[:m*N_GF], nerr
        print('Stage 0')

        for sharedI, iIndex in ieValInv.items():
            C2V[iIndex] = allButOneSum(ext_min_sum,V2C[iIndex])
        for i in range(ne):
            C2V[i] -= np.min(C2V[i])
            C2V[i] = permute_C2V(he[i], C2V[i])
        print('Stage 1')
        
        # update variable nodes
        for i in range(ne):
            Ls = L[je[i]].copy()
            for j in jeValInv[je[i]]:
                if i != j:
                    Ls += C2V[j]
            Ls -= np.min(Ls)
            V2C[i] = permute_V2C(he[i], Ls)
        print('Stage 2')
        
        # update LLR and GF(q) codes
        for i in range(n):
            for j in jeValInv[i]:
                L[i] += C2V[j]
            L[i] -= np.min(L[i])
            code[i] = np.argmin(L[i])
        print('Stage 3')
    
    return gf2bin(code)[:m*N_GF], -1

@numba.jit(nopython=True)
def initLLR(naiveLLR, codeLength)->np.ndarray:
    '''
    Initialize the LLR matrix from the naiveLLR vector.
    naiveLLR: the naive LLR vector, which is a array representing the LLR of each bit
    codeLength: the length of the code
    return: the initialized LLR matrix, which is a 2D array of shape (codeLength, Q_GF)
    normalized so the minimum value in each row is 0.
    '''
    L = np.zeros((codeLength, Q_GF), dtype='float64')
    for i in range(codeLength):
        for j in range(Q_GF):
            for bit in range(N_GF):
                if j & (1 << bit):
                    L[i,j] += naiveLLR[i*N_GF + bit,0]
        L[i] -= np.min(L[i])
    return L

@numba.jit(nopython=True)
def countNonzeros(code:np.ndarray)->int:
    '''
    Count the number of non-zero elements in the code.
    code: the code to be counted
    return: the number of non-zero elements in the code
    '''
    count = 0
    for i in range(len(code)):
        codeSymbol = code[i]
        while codeSymbol:
            count += 1
            codeSymbol &= codeSymbol - 1
    return count

def simulateOneRound(ie, je, he, ieValInv:Mapping[int,List[int]], jeValInv:Mapping[int,List[int]], ebno, H, W):
    ne = len(he)
    V2C = np.zeros((ne, Q_GF), dtype='float64')
    C2V = np.zeros((ne, Q_GF), dtype='float64')
    R = (1 - H/W)
    codeX = np.zeros(((W * N_GF),1))
    symbolX = 1 - 2*codeX
    noise = np.random.randn(W*N_GF,1)
    sigma = np.pow(10, -ebno/20)/R
    y = symbolX + sigma * noise
    print(y.shape)
    naiveLLR = 2. * y / np.pow(sigma, 2)
    print(naiveLLR.shape)
    L = initLLR(naiveLLR, W)
    for i in range(ne):
        V2C[i] = permute_V2C(he[i], L[je[i]])
    code = np.zeros(W, dtype='uint8')

    for iter in range(MAX_ITER):
        print(f'iteration:{iter}')
        for sharedI, iIndex in ieValInv.items():
            C2V[iIndex] = allButOneSum(ext_min_sum,V2C[iIndex])
        for i in range(ne):
            C2V[i] -= np.min(C2V[i])
            C2V[i] = permute_C2V(he[i], C2V[i])
        
        for i in range(ne):
            Ls = L[je[i]].copy()
            for j in jeValInv[je[i]]:
                if i != j:
                    Ls += C2V[j]
            Ls -= np.min(Ls)
            V2C[i] = permute_V2C(he[i], Ls)
        
        # update LLR and GF(q) codes
        for i in range(W):
            for j in jeValInv[i]:
                L[i] += C2V[j]
            L[i] -= np.min(L[i])
            code[i] = np.argmin(L[i])
        if not np.any(code): # We used symmetric property, by enforcing the original code to be 0.
            return iter, False, 0
    if np.any(code):
        isBlockError = True
        numBitError = countNonzeros(code)
        return max_iter, isBlockError, numBitError
    else:
        return max_iter, False, 0


if __name__ == "__main__":
    from fileio.matrix_io import read_matrix as reader
    HcheckMat = reader('Example0.txt')
    H_idx = np.where(HcheckMat)
    H_ele = HcheckMat[H_idx]
    H, W = HcheckMat.shape
    ie, je = H_idx
    he = H_ele
    ne = len(he)
    jeValInv:Mapping[int,List[int]]= dict()
    for idx, val in enumerate(je):
        if val not in jeValInv:
            jeValInv[val] = []
        jeValInv[val].append(idx)
    
    ieValInv:Mapping[int,List[int]]= dict()
    for idx, val in enumerate(ie):
        if val not in ieValInv:
            ieValInv[val] = []
        ieValInv[val].append(idx)
    res = simulateOneRound(ie, je, he, H_idx, H_ele, 7.0, H, W)
    print(res)
    exit()
    symbols = np.zeros((W * N_GF), dtype='uint8')
    for i in range(60):
        symbols[i] = 1
    
    max_iter = 1000
    max_errs = 100
    collected_blerrs = 0
    collected_symerrs = 0
    for _ in range(max_iter):
        res = decode_NB_LDPC(H_idx, H_ele, H, W, symbols)
        print(res)
        exit()