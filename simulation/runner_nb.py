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
from typing import List, Mapping
from fileio.matrix_io import read_matrix as reader
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, freeze_support, Manager
from time import time

# By defining the character of the field:q, and the primitive polynomial,
# We generate the GF(q^n) field, alongside with their parameters.

# Configuration parameters ------------------------------------------------------
# The number of errors to be collected
max_err = 100
# The file name of the check matrix
filename = "Example0.txt"
# The primitive polynomial
x = sympy.symbols('x')
primitive_poly = sympy.poly(x**7+x+1, domain='GF(2)') # The generator polynomial
#primitive_poly = sympy.poly(x**6+x**5+1, domain='GF(2)') # The generator polynomial
# The iteration number of the decoder
MAX_ITER = 10
# The number of rounds to be simulated
MAX_RUNS = 1e3
# Ebno vector
ebno_vec = [5.2]
# LLR truncation size of EMS
NM_EMS = 32
# The proportion of #process/#cpu
ratio = 0.5
# The resolution of the simulation, every resolution rounds, the simulation will be printed
resolution = 100

N_GF = primitive_poly.degree() # The degree of the generator polynomial
Q_GF = 1 << N_GF # The number of elements in GF(q^n)
MAX_RUNS = int(MAX_RUNS)

# GF(q) tables -----------------------------------------------------------------
GF_VEC = [] # The vectors of GF(q), representing with binary format

HcheckMat = reader('Example0.txt')
H_idx = np.where(HcheckMat)
H_ele = HcheckMat[H_idx]
H, W = HcheckMat.shape
R = (1 - H/W)
print(f'{filename} loaded')
print(f'R = {R}')
print(f'Q_GF = {Q_GF}')
print(f'Matrix Size = {H}x{W}')
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
init_table()

def permute_V2C(h, V2C):
    V2C_p = np.zeros(Q_GF, dtype='float64')
    np.put(V2C_p, GF_MUL[h], V2C)
    return V2C_p

@numba.jit(nopython=True)
def permute_V2C(h, V2C):
    V2C_p = np.zeros(Q_GF, dtype='float64')
    for i in range(Q_GF):
        V2C_p[GF_MUL[h][i]] = V2C[i]
    return V2C_p

def permute_C2V(h, C2V):
    return C2V[GF_MUL[h]].copy()

@numba.jit(nopython=True)
def permute_C2V(h, C2V):
    C2V_p = np.zeros(Q_GF, dtype='float64')
    for i in range(Q_GF):
        C2V_p[i] = C2V[GF_MUL[h][i]]
    return C2V_p

# extended-min-sum (EMS) of LLRs ([2]) -----------------------------------------
@numba.jit(nopython=True)
def ext_min_sum(L1, L2):
    if len(L1) == 0:
        return L2
    idx1 = np.argsort(L1)
    idx2 = np.argsort(L2)
    maxL = L1[idx1[NM_EMS-1]] + L2[idx2[NM_EMS-1]]
    Ls = np.full(Q_GF, maxL, dtype='float64')
    
    for i in idx1[:NM_EMS]:
        for j in idx2[:NM_EMS]:
            if L1[i] + L2[j] < Ls[i^j]:
                Ls[i^j] = L1[i] + L2[j]
    return Ls

def eext_min_sum(L1:np.ndarray, L2:np.ndarray)->np.ndarray:
    idx1 = np.argpartition(L1, NM_EMS)[:NM_EMS]
    idx2 = np.argpartition(L2, NM_EMS)[:NM_EMS]
    l1 = L1[idx1]
    l2 = L2[idx2]
    maxL = l1[-1]+l2[-1]
    Ls = np.full(Q_GF, maxL, dtype='float64')
    np.minimum.at(Ls, np.bitwise_xor.outer(idx1, idx2).ravel(), np.add.outer(l1, l2).ravel())
    return Ls.copy()

#@numba.jit(nopython=False)
def eext_min_sum(L1:np.ndarray, L2:np.ndarray)->np.ndarray:
    #L1 = L1.copy()
    #L2 = L2.copy()
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

@numba.jit(nopython=True, nogil=True)
def check_parity(ie, je, he, m, code):
    s = np.zeros(m, dtype='uint8')
    for i in range(len(ie)):
        s[ie[i]] ^= GF_MUL[he[i]][code[je[i]]]
    return np.sum(s != 0)

def simulateOneRound(ie, je, he, ieValInv:Mapping[int,List[int]], jeValInv:Mapping[int,List[int]], ebno, H, W):
    ne = len(he)
    V2C = np.zeros((ne, Q_GF), dtype='float64')
    C2V = np.zeros((ne, Q_GF), dtype='float64')
    R = (1 - H/W)
    codeX = np.zeros(((W * N_GF),1))
    symbolX = 1 - 2*codeX
    noise = np.random.randn(W*N_GF,1)
    sigma = np.pow(10, -ebno/20)/np.sqrt(2*R)
    y = symbolX + sigma * noise
    naiveLLR = 2. * y / np.pow(sigma, 2)
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
        
        for sharedI, iIndex in jeValInv.items():
            V2C[iIndex] = allButOneSum(lambda x, y: x + y, C2V[iIndex])
        for i in range(ne):
            V2C[i] -= np.min(V2C[i])
            V2C[i] = permute_V2C(he[i], V2C[i])
        # update LLR and GF(q) codes
        for i in range(W):
            for j in jeValInv[i]:
                L[i] += C2V[j]
            L[i] -= np.min(L[i])
            code[i] = np.argmin(L[i])
        if not np.any(code): # We used symmetric property, by enforcing the original code to be 0.
            return iter, False, 0
        print(np.sum(code != 0))
        print(check_parity(ie, je, he, W, code))
    if np.any(code):
        isBlockError = True
        numSymbolError = np.sum(code != 0)
        #numBitError = countNonzeros(code)
        return MAX_ITER, isBlockError, numSymbolError
    else:
        return MAX_ITER, False, 0

def doIterDecode(C2V, V2C, L, ieValInv:Mapping[int,List[int]], jeValInv:Mapping[int,List[int]], W, H, he, code):
    for iter in range(MAX_ITER):
        print(f'iteration:{iter}')
        for sharedI, iIndex in ieValInv.items():
            C2V[iIndex] = allButOneSum(ext_min_sum,V2C[iIndex])
        for i in range(ne):
            C2V[i] -= np.min(C2V[i])
            C2V[i] = permute_C2V(he[i], C2V[i])
        
        for sharedI, iIndex in jeValInv.items():
            V2C[iIndex] = allButOneSum(lambda x, y: x + y, C2V[iIndex])
        for i in range(ne):
            V2C[i] -= np.min(V2C[i])
            V2C[i] = permute_V2C(he[i], V2C[i])
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
        numSymbolError = np.sum(code != 0)
        #numBitError = countNonzeros(code)
        return MAX_ITER, isBlockError, numSymbolError
    else:
        return MAX_ITER, False, 0

round_per_sim = 100
def simulate_multi_round(Err_collected):
    cur_had_runs = []
    cur_bit_errs = []
    cur_block_errs = []
    cur_iters = []
    for _ in range(round_per_sim):
        cur_had_run = [None for _ in ebno_vec]
        cur_block_err = [0 for _ in ebno_vec]
        cur_bit_err = [0 for _ in ebno_vec]
        cur_iter = [0 for _ in ebno_vec]
        ne = len(he)
        V2C = np.zeros((ne, Q_GF), dtype='float64')
        C2V = np.zeros((ne, Q_GF), dtype='float64')
        #R = (1 - H/W)
        codeX = np.zeros(((W * N_GF),1))
        symbolX = 1 - 2*codeX
        noise = np.random.randn(W*N_GF,1)
        for i_ebno,ebno in enumerate(ebno_vec):
            if Err_collected[i_ebno] >= max_err:
                continue
            sigma = np.pow(10, -ebno/20)/np.sqrt(2*R)
            y = symbolX + sigma * noise
            naiveLLR = 2. * y / np.pow(sigma, 2)
            L = initLLR(naiveLLR, W)
            for i in range(ne):
                V2C[i] = permute_V2C(he[i], L[je[i]])
            code = np.zeros(W, dtype='uint8')

            iterComple, hasBlockError, numSymbolError = doIterDecode(C2V, V2C, L, ieValInv, jeValInv, W, H, he, code)


            cur_iter[i_ebno] = iterComple
            if hasBlockError:
                cur_had_run[i_ebno] = False
                cur_block_err[i_ebno] += 1
                cur_bit_err[i_ebno] += numSymbolError
                Err_collected[i_ebno] += 1
            else:
                cur_had_run[i_ebno] = True
        cur_had_runs.append(cur_had_run)
        cur_bit_errs.append(cur_bit_err)
        cur_block_errs.append(cur_block_err)
        cur_iters.append(cur_iter)
    return cur_had_runs, cur_bit_errs, cur_block_errs, cur_iters

if __name__ == '__main__':
    num_block_err = np.zeros((len(ebno_vec),1))
    num_bit_err = np.zeros((len(ebno_vec),1))
    num_iter = np.zeros((len(ebno_vec),1))
    num_runs = np.zeros((len(ebno_vec),1))
    #mp.set_start_method("spawn")
    freeze_support()
    start = time()
    num_proc = int(cpu_count()*ratio)
    try:
        with Manager() as manager:
            Err_collected = manager.list([0 for _ in range(len(ebno_vec))])
            cnt = 0
            with Pool(num_proc) as pool:
                for res in pool.imap_unordered(simulate_multi_round, (Err_collected for _ in range(MAX_RUNS//round_per_sim))):
                    for had_run, bit_err, block_err, iters in zip(*res):
                        for idx,(i_had_run, i_bit_err, i_block_err, i_iter) in enumerate(zip(had_run, bit_err, block_err, iters)):
                            if i_had_run == False:
                                if num_block_err[idx] >= max_err:
                                    continue
                                num_block_err[idx] += 1
                                num_bit_err[idx] += i_bit_err
                                num_iter[idx] += i_iter
                                num_runs[idx] += 1
                            if i_had_run == True: # No error
                                num_runs[idx] += 1
                                num_iter[idx] += i_iter
                    if all(num_block_err >= max_err):
                        pool.terminate()
                        break
                    cnt += 1
                    if (cnt*round_per_sim)%resolution == 0:
                        rounds_already = cnt * round_per_sim
                        remain_rounds = 0
                        for i in num_block_err.reshape(-1):
                            if 0 == i:
                                remain_rounds = MAX_RUNS
                                continue
                            remain_rounds = max(remain_rounds, int((max_err-i)/i*rounds_already))
                        remain_rounds = min(remain_rounds,MAX_RUNS - rounds_already)
                        remain_time = (time()-start)/rounds_already*remain_rounds
                        print(f'Time elapsed: {time()-start}')
                        print(f'Expected Time Remaining: {remain_time} seconds')
                        print(f'Already ran {rounds_already} round of simulation')
                        print(f'Collected Errors: '+"\t".join(map(str,num_block_err)))
    except KeyboardInterrupt:
        print('Abort')


    print(f'Time elapsed: {time()-start}')

    print('BLER simulation is finished')
    bler = np.sum(num_block_err,axis=1)/np.sum(num_runs,axis=1)
    ber = np.sum(num_bit_err,axis=1)/np.sum(num_runs,axis=1)/W
    ave_iter = np.sum(num_iter,axis=1)/np.sum(num_runs,axis=1)
    print(' ebno   :'+'\t'.join(map(str,ebno_vec)))
    print('num_runs:'+'\t'.join(map(str,num_runs.reshape(-1))))
    print('ave_bler:'+'\t'.join(map(str,bler)))
    print('ave_ber :'+'\t'.join(map(str,ber)))
    print('ave_iter:'+'\t'.join(map(str,ave_iter)))
    with open(f'{filename}_output.txt','a') as file:
        write = lambda x:file.write(x+'\n')
        write(' ebno   :'+'\t'.join(map(str,ebno_vec)))
        write('num_runs:'+'\t'.join(map(str,num_runs.reshape(-1))))
        write('ave_bler:'+'\t'.join(map(str,bler)))
        write('ave_ber :'+'\t'.join(map(str,ber)))
        write('ave_iter:'+'\t'.join(map(str,ave_iter)))