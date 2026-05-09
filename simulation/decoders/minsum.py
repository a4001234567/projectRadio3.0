from numba import njit
import numpy as np
from typing import List, Tuple

#eps 
#THRES = 1.0 - 2*eps
tanh = np.tanh;arctanh = np.arctanh;clip = np.clip
copyto=np.copyto;addat = np.add.at;mulat = np.multiply.at

@njit
def min1_min2_per_check(abs_v, indx, indy, min1, min2, C):
    min1.fill(1e30)
    min2.fill(1e30)
    for t in range(C):
        _val = abs_v[t]
        indices_x = indx[t]
        if _val < min1[indices_x]:
            min2[indices_x] = min1[indices_x]
            min1[indices_x] = _val
        elif _val < min2[indices_x]:
            min2[indices_x] = _val
    return min1, min2

def make_Flooding_SMS_Decoder(max_iter:int,
        indx: np.ndarray,
        indy: np.ndarray,
        H: int,
        W: int,
        alpha:float=1.):
    C = indx.shape[0]
    min1 = np.empty(H)
    min2 = np.empty(H)
    prods = np.ones(H)
    def Flooding_SMS_decoder(llr: np.ndarray) -> Tuple[np.ndarray, int]:
        llr = llr.reshape(-1)
        sums = llr.copy()
        v2c = sums[indy].copy()
        for t in range(max_iter):
            mulat(prods,indx,np.sign(v2c))
            sign = prods[indx] / np.sign(v2c)
            abs_v = np.abs(v2c)
            nonlocal min1, min2
            min1, min2 = min1_min2_per_check(abs_v, indx, indy, min1, min2, C)
            use_min1 = (abs_v != min1[indx])
            mag = np.where(use_min1, min1[indx], min2[indx])
            c2v = alpha * sign * mag
            copyto(sums, llr)
            addat(sums, indy, c2v)
            x_hat = (sums < 0.)
            if not np.any(x_hat):
                prods.fill(1.)
                break
            v2c = sums[indy] - c2v
            prods.fill(1.)
        return x_hat.reshape(-1,1), t+1
    return Flooding_SMS_decoder
